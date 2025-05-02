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
from tqdm import tqdm
import instaloader
import requests
from urllib.parse import urlparse

# === GPU Optimization Check ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# === Load Models ===
def load_models():
    # YOLOv8
    yolo_model = YOLO("yolov8x.pt")
    yolo_model.to(device)
    
    # BLIP for captioning
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    # Gender detection
    gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
    gender_labels = ['Male', 'Female']
    
    # NSFW model
    deploy_prototxt = './nsfw_model/deploy.prototxt'
    caffe_model = './nsfw_model/resnet_50_1by2_nsfw.caffemodel'
    nsfw_net = cv2.dnn.readNetFromCaffe(deploy_prototxt, caffe_model)
    
    return {
        'yolo': yolo_model,
        'blip_processor': processor,
        'blip_model': blip_model,
        'gender_net': gender_net,
        'gender_labels': gender_labels,
        'nsfw_net': nsfw_net
    }

models = load_models()

# === Instagram Profile Picture Download ===
def extract_username(insta_url):
    path = urlparse(insta_url).path
    parts = path.strip("/").split("/")
    return parts[0] if parts else None

def download_profile_pic(username, output_dir="output"):
    L = instaloader.Instaloader()
    try:
        profile = instaloader.Profile.from_username(L.context, username)
        profile_pic_url = profile.profile_pic_url

        os.makedirs(output_dir, exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(profile_pic_url, headers=headers)

        if response.status_code == 200:
            filepath = os.path.join(output_dir, f"{username}.jpg")
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"✅ Profile picture saved as: {filepath}")
            return filepath
        else:
            print(f"❌ Failed to download image for {username}, status: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Error processing {username}: {e}")
        return None

# === NSFW Detection ===
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unreadable: {image_path}")
    image_resized = cv2.resize(image, (224, 224))
    image_resized = image_resized.astype(np.float32) / 255.0
    blob = cv2.dnn.blobFromImage(image_resized, scalefactor=1.0, size=(224, 224),
                                 mean=(104.0, 117.0, 123.0), swapRB=True)
    return blob

def get_nsfw_category(score):
    if score < 0.3:
        return "Safe"
    elif score < 0.6:
        return "Suggestive"
    else:
        return "Explicit"

def predict_nsfw(image_path):
    try:
        blob = preprocess_image(image_path)
        models['nsfw_net'].setInput(blob)
        output = models['nsfw_net'].forward()
        nsfw_score = float(output[0][1])
        nsfw_category = get_nsfw_category(nsfw_score)
        return nsfw_score, nsfw_category
    except Exception as e:
        print(f"NSFW detection error: {e}")
        return None, str(e)

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

# === Main Image Analysis ===
def analyze_image(image_path):
    results = {
        'image_path': image_path,
        'username': os.path.splitext(os.path.basename(image_path))[0],
        'cowboy_hat': False,
        'bikini': False,
        'horse': False,
        'tractor': False,
        'one_man': False,
        'one_woman': False,
        'hair_color': 'unknown',
        'face_count': 0,
        'caption': '',
        'nsfw_score': None,
        'nsfw_category': None
    }

    # First run NSFW detection
    nsfw_score, nsfw_category = predict_nsfw(image_path)
    results['nsfw_score'] = nsfw_score
    results['nsfw_category'] = nsfw_category

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot read image: {image_path}")
        return results

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # === YOLO Detection ===
    yolo_result = models['yolo'](image_path, verbose=False, device=0 if device == "cuda" else "cpu")[0]
    for box in yolo_result.boxes.data.tolist():
        cls_id = int(box[5])
        confidence = box[4]
        if confidence > 0.5:
            cls_name = models['yolo'].names[cls_id].lower()
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
        models['gender_net'].setInput(blob)
        preds = models['gender_net'].forward()
        gender = models['gender_labels'][preds[0].argmax()]
        genders.append(gender)

        # Only estimate hair color for the first detected face
        if results['hair_color'] == 'unknown':
            results['hair_color'] = estimate_hair_color(face_crop)

    results['face_count'] = len(genders)
    results['one_man'] = genders.count('Male') == 1 and len(genders) == 1
    results['one_woman'] = genders.count('Female') == 1 and len(genders) == 1

    # === BLIP Captioning ===
    raw_image = Image.open(image_path).convert("RGB")
    inputs = models['blip_processor'](raw_image, return_tensors="pt").to(device)
    out = models['blip_model'].generate(**inputs)
    caption = models['blip_processor'].decode(out[0], skip_special_tokens=True).lower()
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

# === Process CSV File ===
def process_csv_file(csv_path):
    output_csv = os.path.join(os.path.dirname(csv_path), 'analysis_results.csv')
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    urls = df['User Homepage'].dropna().unique()
    
    results = []
    
    for url in tqdm(urls, desc="Processing Instagram URLs"):
        username = extract_username(url)
        if not username:
            print(f"❌ Could not extract username from URL: {url}")
            continue
            
        print(f"🔍 Processing profile: {username}")
        
        # Download profile picture
        image_path = download_profile_pic(username)
        if not image_path:
            continue
            
        # Analyze the image
        try:
            result = analyze_image(image_path)
            results.append(result)
        except Exception as e:
            print(f"⚠️ Error analyzing image {image_path}: {e}")
            continue
    
    # Save results to CSV
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"✅ Analysis results saved to: {output_csv}")
    else:
        print("⚠️ No valid results to save")