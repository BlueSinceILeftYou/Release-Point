from pathlib import Path 
import cv2


HEAD_SECONDS = 3
SCENE_THRESHOLD = 3
INPUT_DIR = Path("../data/videos")
OUTPUT_DIR = Path("../data/videos/processed")

def detect_scene_change(cap, HEAD_SECONDS, fps):
    """
        this function aims to find the first scene change
        with the goal to cut

        returns list of frames where distance is above threshold
    """
    frames_to_read = HEAD_SECONDS * fps

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

    return change_list

def trim_and_save(input_path, trim_frame, OUTPUT_PATH):
    pass

def main():
    """
        Trim the videos
    """
    video_files = list(Path(INPUT_DIR).glob('*.mp4'))
    
    for vid in video_files:
        # cam view
        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS)       # get FPS

        scene_changes = detect_scene_change(cap, HEAD_SECONDS, fps)
        print("")
        
