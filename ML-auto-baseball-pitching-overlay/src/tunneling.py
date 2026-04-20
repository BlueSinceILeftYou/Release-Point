import cv2
import numpy as np


def draw_tunnel_zone(frame, current_balls, frame_idx, divergence_frame, burst_duration=20):
    valid_balls = [b for b in current_balls if b is not None]
    if len(valid_balls) < 2:
        return frame

    pts = np.float32(valid_balls).reshape(-1, 1, 2)
    (cx, cy), radius = cv2.minEnclosingCircle(pts)
    cx, cy, radius = int(cx), int(cy), int(radius)

    if divergence_frame is None or frame_idx < divergence_frame:
        return _draw_tunneling(frame, cx, cy, radius, frame_idx)

    frames_past = frame_idx - divergence_frame
    if frames_past <= burst_duration:
        return _draw_burst(frame, cx, cy, radius, frames_past, burst_duration)

    return frame


def _draw_tunneling(frame, cx, cy, radius, frame_idx):
    pulse = 0.5 + 0.15 * np.sin(frame_idx * 0.4)
    r = radius + 12

    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), r, (255, 255, 180), -1)
    frame = cv2.addWeighted(overlay, 0.15 * pulse, frame, 1 - 0.15 * pulse, 0)

    for i, extra in enumerate([0, 4, 8]):
        ring_alpha = float(0.7 * (1 - i / 3) * pulse)
        ring_overlay = frame.copy()
        cv2.circle(ring_overlay, (cx, cy), r + extra, (255, 220, 50), 2)
        frame = cv2.addWeighted(ring_overlay, ring_alpha, frame, 1 - ring_alpha, 0)

    return frame


def _draw_burst(frame, cx, cy, base_radius, frames_past, burst_duration):
    progress = frames_past / burst_duration
    burst_r = int(base_radius + 12 + progress * 60)
    alpha = float(0.5 * (1 - progress) ** 2)

    if alpha < 0.01:
        return frame

    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), burst_r, (255, 255, 180), -1)
    frame = cv2.addWeighted(overlay, alpha * 0.4, frame, 1 - alpha * 0.4, 0)

    ring_overlay = frame.copy()
    cv2.circle(ring_overlay, (cx, cy), burst_r, (255, 220, 50), 3)
    frame = cv2.addWeighted(ring_overlay, alpha, frame, 1 - alpha, 0)

    return frame
