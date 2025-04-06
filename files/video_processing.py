import os
import cv2

from ultralytics import YOLO
YOLO("yolov8n.pt")  # Downloads weights if not already available

# === CONFIG ===
VIDEO_FOLDER = "database"
FRAME_INTERVAL_SEC = 1  # Sample every 1 second
MODEL_PATH = "yolov8n.pt"

# === Setup ===

model = YOLO(MODEL_PATH)

def process_video(video_path):
    video = cv2.VideoCapture(video_path)
    OUTPUT_DIR = "database/extrac_frames/{video_path}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * FRAME_INTERVAL_SEC)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video loaded: {video_path}")
    print(f"[INFO] FPS: {fps}, Total frames: {frame_count}")

    frame_idx = 0
    saved_count = 0

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        if frame_idx % frame_interval == 0:
            print(f"[INFO] Processing frame {frame_idx}")
            results = model.predict(source=frame, conf=0.4, classes=[0], verbose=False)  # class 0 = person

            for i, result in enumerate(results):
                boxes = result.boxes.xyxy.cpu().numpy()
                for j, box in enumerate(boxes):
                    save_path = os.path.join(OUTPUT_DIR, f"frame{frame_idx}_person{j}.jpg")
                    cv2.imwrite(save_path, frame)
                    saved_count += 1

        frame_idx += 1

    video.release()
    print(f"[DONE] Saved {saved_count} cropped person images to {OUTPUT_DIR}")

# Process all videos in the folder
for video_file in os.listdir(VIDEO_FOLDER):
    if video_file.lower().endswith((".mp4", ".avi", ".mov")):
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        process_video(video_path)