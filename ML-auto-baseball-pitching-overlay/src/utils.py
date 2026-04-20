import cv2
import copy
import numpy as np
from src.FrameInfo import FrameInfo


TUNNEL_COLOR = (240, 240, 240)


def draw_ball_curve(frame, trajectory, divergence_idx=None):
    trajectory_weight = 0.7

    if not len(trajectory):
        return frame

    pitch_color = trajectory[-1][2]
    split = (
        divergence_idx
        if divergence_idx is not None and 0 < divergence_idx < len(trajectory)
        else len(trajectory)
    )

    def _draw_segment(f, points, color):
        if len(points) < 2:
            return f
        tmp = f.copy()
        pts = np.array(points, dtype="int32")
        cv2.polylines(tmp, [pts], False, color, 22, lineType=cv2.LINE_AA)
        return cv2.addWeighted(tmp, trajectory_weight, f, 1 - trajectory_weight, 0)

    pre_pts = [[p[0], p[1]] for p in trajectory[:split]]
    # overlap by one point so segments connect cleanly
    post_pts = [[p[0], p[1]] for p in trajectory[max(0, split - 1) :]]

    frame = _draw_segment(frame, pre_pts, TUNNEL_COLOR)
    frame = _draw_segment(frame, post_pts, pitch_color)

    last_ball = (trajectory[-1][0], trajectory[-1][1])
    cv2.circle(frame, last_ball, 13, (255, 255, 255), -1)
    return frame


def fill_lost_tracking(frame_list):
    balls_x = [frame.ball[0] for frame in frame_list if frame.ball_in_frame]
    balls_y = [frame.ball[1] for frame in frame_list if frame.ball_in_frame]

    # print(balls_x)
    # print(balls_y)

    # Get the polynomial equation
    curve = np.polyfit(balls_x, balls_y, 2)
    poly = np.poly1d(curve)

    lost_sections = []
    in_lost = False
    frame_count = 0

    # Get the sections where the ball is lost tracked
    for idx, frame in enumerate(frame_list):
        if frame.ball_lost_tracking and frame_count == 0:
            in_lost = True
            lost_sections.append([])

        if in_lost and not (frame.ball_lost_tracking):
            in_lost = False
            frame_count = 0

        if in_lost:
            lost_sections[-1].append(idx)
            frame_count += 1

    # Modify the frames in lost section with the approximated ball
    for lost_section in lost_sections:
        if lost_section:
            prev_frame = frame_list[lost_section[0] - 1]
            last_frame = frame_list[lost_section[-1] + 1]
            color = prev_frame.ball_color

            lost_idx = [frame_list[i] for i in lost_section]

            # Speed is the x difference for each frame
            diff = last_frame.ball[0] - prev_frame.ball[0]
            speed = int(diff / (len(lost_idx) + 1))

            for idx, frame in enumerate(lost_idx):
                x = prev_frame.ball[0] + (speed * (idx + 1))
                y = int(poly(x))
                frame.ball_in_frame = True
                frame.ball = (x, y)
                frame.ball_color = color
                # print('Fill', x, y)


def distance(x, y):
    temp = (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
    return temp ** (0.5)
