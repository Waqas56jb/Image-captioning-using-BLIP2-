import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import cvlib as cv
from cvlib.object_detection import draw_bbox
from sklearn.cluster import KMeans
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms
from tqdm import tqdm

# === GPU Optimization Check ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# === Load YOLOv8 (pretrained) ===
yolo_model = YOLO("yolov8x.pt")
yolo_model.to(device)

# === Load BLIP model for zero-shot captioning ===
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# === Gender Detection (OpenCV DNN) ===
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
gender_labels = ['Male', 'Female']

# === Hair Color Estimation ===
def estimate_hair_color(face_img):
    face_img = cv2.resize(face_img, (50, 50))
    pixels = face_img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, random_state=42).fit(pixels)
    r, g, b = kmeans.cluster_centers_[0]
    if r > 150 and g > 100 and b < 100:
        return "blonde"
    elif r < 90 and g < 90 and b < 90:
        return "black"
    elif r > 180 and g > 180 and b > 180:
        return "gray"
    elif r > 130 and g > 80 and b < 100:
        return "brown"
    else:
        return "unknown"

# === Analyze Single Image ===
def analyze_image(image_path):
    results = {
        'image_path': image_path,
        'cowboy_hat': False,
        'bikini': False,
        'horse': False,
        'tractor': False,
        'one_man': False,
        'one_woman': False,
        'hair_color': 'unknown',
        'face_count': 0,
        'caption': ''
    }

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot read image: {image_path}")
        return results

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # === YOLO Detection ===
    yolo_result = yolo_model(image_path, verbose=False, device=0 if device == "cuda" else "cpu")[0]
    for box in yolo_result.boxes.data.tolist():
        cls_id = int(box[5])
        confidence = box[4]
        if confidence > 0.5:
            cls_name = yolo_model.names[cls_id].lower()
            if 'hat' in cls_name and 'cowboy' in cls_name:
                results['cowboy_hat'] = True
            if 'bikini' in cls_name or 'swimsuit' in cls_name:
                results['bikini'] = True
            if 'horse' in cls_name:
                results['horse'] = True
            if 'tractor' in cls_name:
                results['tractor'] = True

    # === Face Detection and Gender Classification ===
    faces, confidences = cv.detect_face(image)
    threshold = 0.8
    genders = []
    for face, conf in zip(faces, confidences):
        if conf < threshold:
            continue
        x1, y1, x2, y2 = face
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (104.0, 177.0, 123.0))
        gender_net.setInput(blob)
        preds = gender_net.forward()
        gender = gender_labels[preds[0].argmax()]
        genders.append(gender)

        # Only estimate hair color for the first detected face
        if results['hair_color'] == 'unknown':
            results['hair_color'] = estimate_hair_color(face_crop)

    results['face_count'] = len(genders)
    results['one_man'] = genders.count('Male') == 1 and len(genders) == 1
    results['one_woman'] = genders.count('Female') == 1 and len(genders) == 1

    # === BLIP Captioning ===
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True).lower()
    results['caption'] = caption

    # === Refine using caption ===
    if 'cowboy hat' in caption or ('hat' in caption and 'cowboy' in caption):
        results['cowboy_hat'] = True
    if 'bikini' in caption or 'swimsuit' in caption:
        results['bikini'] = True
    if 'horse' in caption:
        results['horse'] = True
    if 'tractor' in caption:
        results['tractor'] = True

    return results

# === Batch Process & Save Results ===
def analyze_and_save(images, csv_path="detection_results.csv"):
    existing_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    all_results = []

    for image_path in tqdm(images, desc="🔍 Analyzing"):
        if not existing_df.empty and image_path in existing_df['image_path'].values:
            print(f"⏭️ Skipping (already processed): {image_path}")
            continue

        result = analyze_image(image_path)
        all_results.append(result)

    if all_results:
        df = pd.DataFrame(all_results)
        mode = 'a' if os.path.exists(csv_path) else 'w'
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, index=False, mode=mode, header=header)
        print("📄 Results saved to CSV.")
    else:
        print("👍 No new images to process.")

# === MAIN ===
if __name__ == "__main__":
    image_paths = ["./output/lizcox04.jpg"]  # Replace or extend with your images
    analyze_and_save(image_paths)
