import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO

# Loading the frames in videos
def load_video_frames(video_path, frame_skip=5):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n Video Info: {video_path.name} — {total_frames} frames | {fps:.1f} fps | {width}x{height}")

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            frames.append(frame)
        count += 1
    cap.release()
    print(f"Loaded {len(frames)} frames.")
    return frames

# Short cut detection feature
def detect_hard_cuts(frames, threshold=0.6):
    cut_count = 0
    prev_hist = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold:
                cut_count += 1
        prev_hist = hist
    return cut_count

# Motion analysis features
def compute_average_motion(frames):
    motions = []
    for i in range(1, len(frames)):
        prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motions.append(np.mean(mag))
    return float(np.mean(motions)) if motions else 0.0

# Object vs Person detection feature
def object_vs_person_ratio(frames, model_path='yolov8n.pt', sample_rate=10):
    model = YOLO(model_path)
    person_count = 0
    object_count = 0
    for i in range(0, len(frames), sample_rate):
        results = model.predict(source=frames[i], verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for cls in r.boxes.cls:
                cls_id = int(cls.item())
                if cls_id == 0:
                    person_count += 1
                else:
                    object_count += 1
    if object_count == 0:
        return float('inf') if person_count > 0 else 0.0
    return person_count / object_count


def extract_features(video_path):
    frames = load_video_frames(video_path)
    if len(frames) < 2:
        print("Not enough frames to analyze.")
        return {
            "hard_cuts": 0,
            "average_motion": 0.0,
            "person_object_ratio": 0.0
        }

    print("Detecting hard cuts...")
    cuts = detect_hard_cuts(frames)

    print("Calculating average motion...")
    motion = compute_average_motion(frames)

    print("Running object vs person detection...")
    pop_ratio = object_vs_person_ratio(frames)

    return {
        "hard_cuts": cuts,
        "average_motion": motion,
        "person_object_ratio": pop_ratio
    }


def main():
    folder = Path(r"C:\Users\aksha\Desktop\White Panda\Video Feature Extraction\Videos")

    if not folder.exists():
        print("Folder not found:", folder)
        return

    video_files = [f for f in folder.iterdir() if f.suffix.lower() == ".mp4" and f.stem.startswith("SampleVideo")]

    if not video_files:
        print("No sample videos found.")
        return

    for video in video_files:
        print(f"\n Processing: {video.name}")
        try:
            features = extract_features(video)
            print(json.dumps(features, indent=2))
        except Exception as e:
            print(f"Error processing {video.name}: {e}")

if __name__ == "__main__":
    main()
