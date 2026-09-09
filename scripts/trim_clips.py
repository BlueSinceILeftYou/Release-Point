from pathlib import Path 
import cv2


HEAD_SECONDS = 3
SCENE_THRESHOLD = .3                            # Bhattacharyya distance bounded [0, 1]
_SCRIPTS_DIR = Path(__file__).parent
INPUT_DIR = _SCRIPTS_DIR / "../data/videos"
OUTPUT_DIR = _SCRIPTS_DIR / "../data/videos/processed"

def detect_scene_change(cap, head_seconds, fps):
    """
        this function aims to find the first scene change
        with the goal to cut

        returns list of frames where distance is above threshold
    """
    frames_to_read = int(head_seconds * fps)

    ret, frame = cap.read()

    # init first
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])

    change_list = []
    for i in range(1, frames_to_read):
        ret, frame = cap.read()
        if not ret:
            print("Can't receieve...")

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        
        # Bhattacharyya distance between the two frames
        dist = cv2.compareHist(curr_hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)

        if dist > SCENE_THRESHOLD:
            change_list.append(i)

        prev_hist = curr_hist

    return change_list

def find_trime_frame(scene_changes):
    """ Wrapper that returns last secene index from detect scene chganges
        returns 0 if there are no scene changes
    """
    change_index = 0
    if len(scene_changes) > 0:
        # find frame and offset by 1 for thumbnail change
        change_index = scene_changes[-1] + 1 

    return change_index

def trim_and_save(input_path, trim_frame, output_path):
    """
        Given input file and index for beginning trim, save copy to output file
        that begins and trim_frame
    """
    cap = cv2.VideoCapture(str(input_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if trim_frame > 0.8 * total_frames:
        print(f"  WARNING: trim frame {trim_frame} is past 80% of {total_frames} total frames -- skipping.")
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, trim_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()


def main():
    """
        Trim videos in /data/videos and put them into /data/videos/processed
        TO-DO: Add additional functionality to find optimal Bhattacharrya distance/seconds
    """
    video_files = list(INPUT_DIR.rglob('*.mp4'))

    print(f"Found {len(video_files)} video(s). Scanning first {HEAD_SECONDS}s, threshold={SCENE_THRESHOLD}")
    for vid in video_files:
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS)

        scene_changes = detect_scene_change(cap, HEAD_SECONDS, fps)
        cap.release()
        print(f"\n{vid.name}: {len(scene_changes)} scene change(s) detected")

        trim_frame = find_trime_frame(scene_changes)
        print(f"  trimming at frame {trim_frame}")

        output_path = OUTPUT_DIR / vid.relative_to(INPUT_DIR)
        trim_and_save(vid, trim_frame, output_path)
        print(f"  saved to {output_path}")

if __name__ == "__main__":
    main()