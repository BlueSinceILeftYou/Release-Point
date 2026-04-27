import os
import cv2
import numpy as np
import copy
import csv 
import os
from image_registration import cross_correlation_shifts
from src.utils import draw_ball_curve, fill_lost_tracking
from src.FrameInfo import FrameInfo
from src.tunneling import draw_tunnel_zone

_DIVERGE_THRESHOLD = 40
_DIVERGE_CONSECUTIVE = 3

def _save_mse_log(mse_log, outputPath):
    base = os.path.splitext(outputPath)[0]
    
    path = base + "_mse.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mse_log[0].keys())
        writer.writeheader()
        writer.writerows(mse_log)
    
    print(f"MSE log saved to {path}")

def compute_masked_mse(ref_image, corrected_image, ref_ball=None, overlay_ball=None):
    # Compute MSE between reference image and transformed image
    mask = np.ones(ref_image.shape[:2], dtype=np.uint8)

    if ref_ball is not None:
        cv2.circle(mask, ref_ball, _BALL_MASK_RADIUS, 0, -1)

    if overlay_ball is not None:
        cv2.circle(mask, overlay_ball, _BALL_MASK_RADIUS, 0, -1)

    mask_bool = mask.astype(bool)

    diff = ref_image.astype(float) - corrected_image.astype(float)
    diff = diff ** 2 

    return diff[mask_bool].mean()

def generate_overlay(video_frames, width, height, fps, outputPath, registration_type="orb", registration_threshold=0.75, debug_keypoints=False):
    output_dir = os.path.dirname(os.path.abspath(outputPath))
    print("Saving overlay result to", outputPath)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(outputPath, codec, fps / 2, (width, height))

    frame_lists = sorted(video_frames, key=len, reverse=True)
    balls_in_curves = [[] for i in range(len(frame_lists))]
    shifts = {}

    divergence_frame = None
    divergence_curve_len = None
    diverge_run = 0

    mse_log = []

    # Take the longest frames as background
    for idx, base_frame in enumerate(frame_lists[0]):
        # Overlay frames
        background_frame = base_frame.frame.copy()
        for list_idx, frameList in enumerate(frame_lists[1:]):
            if idx < len(frameList):
                overlay_frame = frameList[idx]
            else:
                overlay_frame = frameList[len(frameList) - 1]

            alpha = 1.0 / (list_idx + 2)
            beta = 1.0 - alpha
            if registration_type == "cross_correlation":
                corrected_frame, mse = image_registration(
                    background_frame, overlay_frame, shifts, list_idx, width, height
                )
            else:
                corrected_frame, ball_pos, mse = new_image_registration(
                    background_frame, overlay_frame, registration_threshold, shifts, list_idx, width, height,
                    ref_ball=base_frame.ball, ref_ball_in_frame=base_frame.ball_in_frame,
                    draw_keypoints=debug_keypoints, output_dir=output_dir
                )
                overlay_frame.ball = ball_pos

            background_frame = cv2.addWeighted(
                corrected_frame, alpha, background_frame, beta, 0
            )

            # Prepare balls to draw
            if overlay_frame.ball_in_frame:
                balls_in_curves[list_idx + 1].append(
                    [
                        overlay_frame.ball[0],
                        overlay_frame.ball[1],
                        overlay_frame.ball_color,
                    ]
                )

            mse_log.append({
                "frame_idx": idx,
                "video_pair": list_idx + 1,   # 0 is base, so overlay pairs start at 1
                "registration_type": registration_type,
                "mse": mse,
                "ref_ball": base_frame.ball if base_frame.ball_in_frame else None,
                "overlay_ball": overlay_frame.ball if overlay_frame.ball_in_frame else None,
            })

        if base_frame.ball_in_frame:
            balls_in_curves[0].append(
                [base_frame.ball[0], base_frame.ball[1], base_frame.ball_color]
            )

        # Emphasize base frame
        base_frame_weight = 0.55
        background_frame = cv2.addWeighted(
            base_frame.frame,
            base_frame_weight,
            background_frame,
            1 - base_frame_weight,
            0,
        )

        _save_mse_log(mse_log, outputPath)

        # Track divergence using transformed ball positions
        if divergence_frame is None:
            current_positions = [c[-1][:2] for c in balls_in_curves if c]
            if len(current_positions) >= 2:
                max_dist = max(
                    ((current_positions[i][0] - current_positions[j][0]) ** 2
                     + (current_positions[i][1] - current_positions[j][1]) ** 2) ** 0.5
                    for i in range(len(current_positions))
                    for j in range(i + 1, len(current_positions))
                )
                if max_dist > _DIVERGE_THRESHOLD:
                    diverge_run += 1
                    if diverge_run >= _DIVERGE_CONSECUTIVE:
                        divergence_frame = idx - _DIVERGE_CONSECUTIVE + 1
                        divergence_curve_len = (
                            max(len(c) for c in balls_in_curves) - _DIVERGE_CONSECUTIVE + 1
                        )
                        print(f"Pitch divergence at frame {divergence_frame}")
                else:
                    diverge_run = 0

        # Draw tunnel zone behind trajectories
        current_positions = [c[-1][:2] for c in balls_in_curves if c]
        background_frame = draw_tunnel_zone(
            background_frame, current_positions, idx, divergence_frame
        )

        # Draw transparent curve and non-transparent balls
        for trajectory in balls_in_curves:
            background_frame = draw_ball_curve(background_frame, trajectory, divergence_curve_len)

        result_frame = cv2.cvtColor(background_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("result_frame", result_frame)
        out.write(result_frame)
        if cv2.waitKey(60) & 0xFF == ord("q"):
            break

# ORB detector init
_orb = cv2.ORB_create(nfeatures=1000)
# brute force matcher init
_bf = cv2.BFMatcher(cv2.NORM_HAMMING)

_BALL_MASK_RADIUS = 40
# Fraction of frame height to mask at top and bottom to exclude scorebug/broadcast overlays
_SCOREBUG_MASK_FRACTION = 0.15

def new_image_registration(ref_image, offset_image, threshold, transforms, list_idx, width, height,
                           ref_ball=None, ref_ball_in_frame=False, draw_keypoints=False, output_dir="."):
    """references opencv feature matching and homography estimation tutorial: 
        https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

    If draw_keypoints=True, opens a debug window showing the Lowe-ratio-filtered matches
    between the two frames (only on the first call when the homography is computed).
    """
    
    gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    gray_offset = cv2.cvtColor(offset_image.frame, cv2.COLOR_BGR2GRAY)

    if list_idx not in transforms:
        # Build masks that block ball regions so ball keypoints don't contaminate the homography
        # also for later similarity comparison 
        ref_mask = np.ones_like(gray_ref, dtype=np.uint8) * 255
        offset_mask = np.ones_like(gray_offset, dtype=np.uint8) * 255

        # Mask top and bottom strips to exclude static broadcast scorebug overlays
        scorebug_h = int(gray_ref.shape[0] * _SCOREBUG_MASK_FRACTION)
        ref_mask[:scorebug_h, :] = 0
        ref_mask[-scorebug_h:, :] = 0
        offset_mask[:scorebug_h, :] = 0
        offset_mask[-scorebug_h:, :] = 0

        if ref_ball_in_frame and ref_ball is not None:
            cv2.circle(ref_mask, ref_ball, _BALL_MASK_RADIUS, 0, -1)
        if offset_image.ball_in_frame:
            cv2.circle(offset_mask, offset_image.ball, _BALL_MASK_RADIUS, 0, -1)

        # Find key points with ORB
        kp1, des1 = _orb.detectAndCompute(gray_ref, ref_mask)
        kp2, des2 = _orb.detectAndCompute(gray_offset, offset_mask)

        # Use BF matcher to match the two kps
        matches = _bf.knnMatch(des1, des2, k=2)
        val_patches = []
        for pairs in matches:
            if len(pairs) == 2:
                m, n = pairs
                # for each kp, if distance is close enough, consider it as a match
                if m.distance < threshold * n.distance:
                    val_patches.append(m)

        # compute the homography matrix with RANSAC
        ref_pts = np.float32([kp1[m.queryIdx].pt for m in val_patches])
        offset_pts = np.float32([kp2[m.trainIdx].pt for m in val_patches])

        M, inliers = cv2.findHomography(offset_pts, ref_pts, cv2.RANSAC, 5.0)

        if draw_keypoints:
            # convert back to color for the image
            ref_bgr = cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR)
            offset_bgr = cv2.cvtColor(offset_image.frame, cv2.COLOR_RGB2BGR)

            def _show_and_save(img, title, filename):
                # fit display
                max_w = 1920
                if img.shape[1] > max_w:
                    scale = max_w / img.shape[1]
                    img = cv2.resize(img, (max_w, int(img.shape[0] * scale)))
                cv2.imwrite(filename, img)
                print(f"[debug] saved {filename}")
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(title, img.shape[1], img.shape[0])
                cv2.imshow(title, img)
                cv2.waitKey(0)
                cv2.destroyWindow(title)

            #  all detected keypoints (rich: circle size = scale, line = orientation)
            _YELLOW = (0, 255, 255)  # BGR yellow
            ref_kp_img = cv2.drawKeypoints(
                ref_bgr, kp1, None, color=_YELLOW,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            offset_kp_img = cv2.drawKeypoints(
                offset_bgr, kp2, None, color=_YELLOW,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            for img, label in [(ref_kp_img, f"Reference  ({len(kp1)} kp)"), (offset_kp_img, f"Overlay  ({len(kp2)} kp)")]:
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            all_kp_img = np.hstack([ref_kp_img, offset_kp_img])
            _show_and_save(all_kp_img, f"All keypoints (pair {list_idx})", os.path.join(output_dir, f"debug_kp_pair{list_idx}.jpg"))

            # matching keypoints
            inlier_matches = [val_patches[i] for i, flag in enumerate(inliers) if flag]
            match_img = cv2.drawMatches(
                ref_bgr, kp1,
                offset_bgr, kp2,
                inlier_matches, None,
                matchColor=_YELLOW,
                singlePointColor=_YELLOW,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            cv2.putText(match_img, f"{len(inlier_matches)} inlier matches  |  threshold={threshold}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            _show_and_save(match_img, f"Inlier matches (pair {list_idx})", os.path.join(output_dir, f"debug_matches_pair{list_idx}.jpg"))

        # cache the homography — computed once per video pair
        transforms[list_idx] = M

    M = transforms[list_idx]

    # transform ball
    ball_pt = np.float32([[offset_image.ball[0], offset_image.ball[1]]]).reshape(-1, 1, 2)
    transformed_ball_pt = cv2.perspectiveTransform(ball_pt, M) 
    ball_pos = (int(transformed_ball_pt[0, 0, 0]), int(transformed_ball_pt[0, 0, 1]))

    # correct image
    corrected_image = cv2.warpPerspective(offset_image.frame, M, (width, height))

    mse = compute_masked_mse(
        ref_image,
        corrected_image,
        ref_ball=ref_ball if ref_ball_in_frame else None,
        overlay_ball=ball_pos  # already transformed to ref frame coords
    )

    return corrected_image, ball_pos, mse

def image_registration(ref_image, offset_image, shifts, list_idx, width, height):
    # The shift is calculated once for each video and stored
    if list_idx not in shifts:
        xoff, yoff = cross_correlation_shifts(
            ref_image[:, :, 0], offset_image.frame[:, :, 0]
        )
        shifts[list_idx] = (xoff, yoff)
    else:
        xoff, yoff = shifts[list_idx]

    offset_image.ball = tuple(
        [offset_image.ball[0] - int(xoff), offset_image.ball[1] - int(yoff)]
    )
    matrix = np.float32([[1, 0, -xoff], [0, 1, -yoff]])
    corrected_image = cv2.warpAffine(offset_image.frame, matrix, (width, height))

    mse = compute_masked_mse(
        ref_image,
        corrected_image,
        overlay_ball=offset_image.ball  
    )

    return corrected_image, mse
