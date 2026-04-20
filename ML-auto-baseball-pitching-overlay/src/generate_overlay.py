import cv2
import numpy as np
import copy
from image_registration import cross_correlation_shifts
from src.utils import draw_ball_curve, fill_lost_tracking
from src.FrameInfo import FrameInfo
from src.tunneling import draw_tunnel_zone

_DIVERGE_THRESHOLD = 40
_DIVERGE_CONSECUTIVE = 3


def generate_overlay(video_frames, width, height, fps, outputPath, registration_type="orb", registration_threshold=0.75):
    print("Saving overlay result to", outputPath)
    codec = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(outputPath, codec, fps / 2, (width, height))

    frame_lists = sorted(video_frames, key=len, reverse=True)
    balls_in_curves = [[] for i in range(len(frame_lists))]
    shifts = {}

    divergence_frame = None
    divergence_curve_len = None
    diverge_run = 0

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
                corrected_frame = image_registration(
                    background_frame, overlay_frame, shifts, list_idx, width, height
                )
            else:
                corrected_frame, ball_pos = new_image_registration(
                    background_frame, overlay_frame, registration_threshold, shifts, list_idx, width, height
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

def new_image_registration(ref_image, offset_image, threshold, transforms, list_idx, width, height):
    """references opencv feature matching and homography estimation tutorial: 
        https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    """
    
    gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    gray_offset = cv2.cvtColor(offset_image.frame, cv2.COLOR_BGR2GRAY)
     
    # Find key points with ORB
    kp1, des1 = _orb.detectAndCompute(gray_ref, None)
    kp2, des2 = _orb.detectAndCompute(gray_offset, None)

    # Use BF matcher to match the two kps\
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
    
    # add homography matrix to the list of transforms
    transforms[list_idx] = M

    # transform ball
    ball_pt = np.float32([[offset_image.ball[0], offset_image.ball[1]]]).reshape(-1, 1, 2)
    transformed_ball_pt = cv2.perspectiveTransform(ball_pt, M) 
    ball_pos = (int(transformed_ball_pt[0, 0, 0]), int(transformed_ball_pt[0, 0, 1]))

    # correct image
    corrected_image = cv2.warpPerspective(offset_image.frame, M, (width, height))

    return corrected_image, ball_pos

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

    return corrected_image
