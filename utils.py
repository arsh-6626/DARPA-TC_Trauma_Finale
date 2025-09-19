import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import re

def extract_cropped_human_frames(video_path, num_frames=8, model_path='./best_body.pt', class_name='person'):

    import logging
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    
    # Load YOLO model with all messaging suppressed
    model = YOLO(model_path, verbose=True)
    # model.predictor.args.verbose = False
    # model.predictor.args.hide_labels = True
    # model.predictor.args.hide_conf = True
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    cropped_frames = []
    frame_idx = 0
    extracted = 0

    while extracted < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            # Run YOLO detection
            results = model(frame)
            for result in results:
                boxes = result.boxes
                for box, cls_id in zip(boxes.xyxy, boxes.cls):
                    # Filter only humans
                    if model.names[int(cls_id)] == class_name:
                        x1, y1, x2, y2 = map(int, box)
                        cropped = frame[y1:y2, x1:x2]

                        # Convert to PIL
                        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                        cropped_frames.append(pil_img)
                        extracted += 1
                        break  # Only one human per frame

        frame_idx += 1

    cap.release()

    # If fewer than num_frames, pad by duplicating last frame
    if len(cropped_frames) == 0:
        # fallback: return black images if no detections at all
        cropped_frames = [Image.new("RGB", (224, 224), (0, 0, 0)) for _ in range(num_frames)]
    elif len(cropped_frames) < num_frames:
        last_frame = cropped_frames[-1]
        while len(cropped_frames) < num_frames:
            cropped_frames.append(last_frame.copy())

    return cropped_frames

def extract_face_human_frames(video_path, num_frames=8, model_path="/home/uasdtu/Downloads/yolov8x-face-lindevs.pt", class_name='person'):
    # Load YOLO model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    cropped_frames = []
    frame_idx = 0
    extracted = 0

    while extracted < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            # Run YOLO detection
            results = model(frame, conf = 0.4)
            for result in results:
                boxes = result.boxes
                for box, cls_id in zip(boxes.xyxy, boxes.cls):
                    # Filter only humans
                    if model.names[int(cls_id)] == class_name:
                        x1, y1, x2, y2 = map(int, box)
                        cropped = frame[y1:y2, x1:x2]

                        # Convert to PIL
                        pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                        cropped_frames.append(pil_img)
                        extracted += 1
                        break  # Only one human per frame

        frame_idx += 1

    cap.release()

    # If fewer than num_frames, pad by duplicating last frame
    if len(cropped_frames) == 0:
        # fallback: return black images if no detections at all
        cropped_frames = [Image.new("RGB", (224, 224), (0, 0, 0)) for _ in range(num_frames)]
    elif len(cropped_frames) < num_frames:
        last_frame = cropped_frames[-1]
        while len(cropped_frames) < num_frames:
            cropped_frames.append(last_frame.copy())

    return cropped_frames

def check_tags(output: str) -> bool:
    pattern = r'<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>'
    match = re.search(pattern, output)
    return bool(match)

def parse_to_dict(output: str) -> dict:
    output = output.strip().strip('{}') #stripping of any spaces front and back | extracting content in the curly braces
    items = output.split(',')
    result = {}
    for item in items:
        if ':' in item:
            key, value = item.split(':', 1)
            # remove extra spaces and surrounding quotes
            key = key.strip().strip('"').strip("'")
            value = value.strip().strip('"').strip("'")
            result[key] = value
    return result

def clean_output(report: dict, classes: list)-> dict:
    cleaned_report = {}
    for key, value in report.items():
        try:
            match = next((cls for cls in classes if cls.lower() in value.lower()))
            cleaned_report[key] = match if match else value
        except:
            cleaned_report[key] = value
    return cleaned_report

def load_frames_from_directory(video: str)->list:
    frames_paths = sorted([
                os.path.join(video, f) 
                for f in os.listdir(video) 
                if os.path.splitext(f)[1].lower() in [".jpg", ".png", ".jpeg"]
            ])
    frames = [Image.open(f).convert("RGB") for f in frames_paths]
    return frames

def pad_frames(frames: list)->list:
    if len(frames)%8 !=0:
        pad_len = 8 - (len(frames) % 8)
        frames.extend([frames[-1].copy() for _ in range(pad_len)])
    return frames

    