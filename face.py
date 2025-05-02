import os
import cv2
import numpy as np
import pandas as pd
import requests
import instaloader
from urllib.parse import urlparse
from ultralytics import YOLO
import cvlib as cv
from cvlib.object_detection import draw_bbox
from sklearn.cluster import KMeans
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms
from tqdm import tqdm

# === Configuration ===
OUTPUT_DIR = "output"
CSV_RESULTS = "analysis_results.csv"
COUNTRY_ITEMS = ['cowboy hat', 'horse', 'tractor', 'boots', 'jeans', 'western shirt']

# === GPU Optimization ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# === Initialize Models ===
def initialize_models():
    """Load all required models"""
    models = {}
    
    # YOLOv8 for object detection
    models['yolo'] = YOLO("yolov8x.pt").to(device)
    
    # BLIP for image captioning
    models['processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    models['blip'] = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base").to(device)
    
    # Gender detection
    models['gender_net'] = cv2.dnn.readNetFromCaffe(
        'gender_deploy.prototxt', 'gender_net.caffemodel')
    models['gender_labels'] = ['Male', 'Female']
    
    # NSFW detection
    models['nsfw_net'] = cv2.dnn.readNetFromCaffe(
        './nsfw_model/deploy.prototxt', 
        './nsfw_model/resnet_50_1by2_nsfw.caffemodel')
    
    return models

# === Instagram Profile Download ===
def extract_username(insta_url):
    """Extract username from Instagram URL"""
    path = urlparse(insta_url).path
    parts = path.strip("/").split("/")
    return parts[0] if parts else None

def download_profile_pic(username, output_dir=OUTPUT_DIR):
    """Download Instagram profile picture"""
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
            return filepath
        else:
            print(f"❌ Failed to download image for {username}")
            return None

    except Exception as e:
        print(f"⚠️ Error processing {username}: {e}")
        return None

# === Image Analysis ===
def get_nsfw_category(score):
    """Categorize NSFW score"""
    if score < 0.3:
        return "Safe"
    elif score < 0.6:
        return "Suggestive"
    else:
        return "Explicit"

def predict_nsfw(image_path, nsfw_net):
    """Predict NSFW content"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {'nsfw_score': 0, 'nsfw_category': 'Error: Image not readable'}
            
        image_resized = cv2.resize(image, (224, 224))
        image_resized = image_resized.astype(np.float32) / 255.0
        blob = cv2.dnn.blobFromImage(
            image_resized, scalefactor=1.0, size=(224, 224),
            mean=(104.0, 117.0, 123.0), swapRB=True)
        
        nsfw_net.setInput(blob)
        output = nsfw_net.forward()
        nsfw_score = float(output[0][1])
        
        return {
            'nsfw_score': round(nsfw_score, 4),
            'nsfw_category': get_nsfw_category(nsfw_score)
        }
    except Exception as e:
        return {
            'nsfw_score': 0,
            'nsfw_category': f'Error: {str(e)}'
        }

def estimate_hair_color(face_img):
    """Estimate hair color from face image"""
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

def analyze_image(image_path, models):
    """Analyze image for all required features"""
    results = {
        'image_path': image_path,
        'username': os.path.splitext(os.path.basename(image_path))[0],
        'country_items': [],
        'country_item_count': 0,
        'cowboy_hat': False,
        'horse': False,
        'tractor': False,
        'one_man': False,
        'one_woman': False,
        'face_count': 0,
        'hair_color': 'unknown',
        'caption': '',
        'nsfw_score': 0,
        'nsfw_category': 'Safe'
    }
    
    # Add NSFW analysis
    nsfw_result = predict_nsfw(image_path, models['nsfw_net'])
    results.update(nsfw_result)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot read image: {image_path}")
        return results

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # === YOLO Object Detection ===
    yolo_result = models['yolo'](image_path, verbose=False, 
                               device=0 if device == "cuda" else "cpu")[0]
    detected_items = set()
    
    for box in yolo_result.boxes.data.tolist():
        cls_id = int(box[5])
        confidence = box[4]
        if confidence > 0.5:
            cls_name = models['yolo'].names[cls_id].lower()
            detected_items.add(cls_name)
            
            # Check for specific country items
            if 'hat' in cls_name and 'cowboy' in cls_name:
                results['cowboy_hat'] = True
                results['country_items'].append('cowboy hat')
            if 'horse' in cls_name:
                results['horse'] = True
                results['country_items'].append('horse')
            if 'tractor' in cls_name:
                results['tractor'] = True
                results['country_items'].append('tractor')
    
    results['country_item_count'] = len(results['country_items'])

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
            
        # Gender detection
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (104.0, 177.0, 123.0))
        models['gender_net'].setInput(blob)
        preds = models['gender_net'].forward()
        gender = models['gender_labels'][preds[0].argmax()]
        genders.append(gender)

        # Hair color estimation (first face only)
        if results['hair_color'] == 'unknown':
            results['hair_color'] = estimate_hair_color(face_crop)

    results['face_count'] = len(genders)
    results['one_man'] = genders.count('Male') == 1 and len(genders) == 1
    results['one_woman'] = genders.count('Female') == 1 and len(genders) == 1

    # === BLIP Captioning ===
    try:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = models['processor'](raw_image, return_tensors="pt").to(device)
        out = models['blip'].generate(**inputs)
        caption = models['processor'].decode(out[0], skip_special_tokens=True).lower()
        results['caption'] = caption

        # Refine detections using caption
        for item in COUNTRY_ITEMS:
            if item in caption and item not in results['country_items']:
                results['country_items'].append(item)
                results['country_item_count'] += 1
                
                if item == 'cowboy hat':
                    results['cowboy_hat'] = True
                elif item == 'horse':
                    results['horse'] = True
                elif item == 'tractor':
                    results['tractor'] = True
    except Exception as e:
        results['caption'] = f"Error: {str(e)}"

    return results

# === Main Processing Function ===
def process_instagram_profiles(url_csv, output_csv=CSV_RESULTS):
    """Process Instagram profiles from CSV and save results"""
    # Initialize all models
    print("⏳ Loading AI models...")
    models = initialize_models()
    
    # Read input CSV
    df = pd.read_csv(url_csv)
    urls = df['User Homepage'].dropna().unique()
    
    results = []
    
    for url in tqdm(urls, desc="🔍 Processing Profiles"):
        username = extract_username(url)
        if not username:
            print(f"❌ Could not extract username from URL: {url}")
            continue
            
        # Download profile picture
        image_path = download_profile_pic(username)
        if not image_path:
            continue
            
        # Analyze image
        analysis = analyze_image(image_path, models)
        results.append(analysis)
        
        # Save incremental results
        pd.DataFrame(results).to_csv(output_csv, index=False)
    
    print(f"✅ Analysis complete! Results saved to {output_csv}")

# === Main Execution ===
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process profiles from CSV
    process_instagram_profiles("samples.csv")