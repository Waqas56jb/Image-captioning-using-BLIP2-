import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from nudenet import NudeDetector
from ultralytics import YOLO
import requests
import os
import json

# Initialize models
def init_models():
    # BLIP-2 for image understanding
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    
    # YOLOv8 for object detection (cowboy hats, people, etc.)
    yolo_model = YOLO("yolov8n.pt")  # Auto-downloads
    
    # NSFW/Suggestive detector
    nsfw_detector = NudeDetector()  # Auto-downloads
    
    return processor, blip_model, yolo_model, nsfw_detector

# Analyze image
def analyze_image(image_url, processor, blip_model, yolo_model, nsfw_detector):
    # Download image
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    
    # --- BLIP-2 Analysis ---
    # Question: "Is there a cowboy hat, country outfit, or horse? How many people are visible?"
    inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
    generated_ids = blip_model.generate(**inputs, max_new_tokens=50)
    blip_description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # --- YOLOv8 Object Detection ---
    yolo_results = yolo_model(image)
    detected_objects = []
    for result in yolo_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]
            detected_objects.append(label)
    
    # --- NSFW/Suggestive Detection ---
    nsfw_results = nsfw_detector.detect(image_url)
    suggestive_score = len([r for r in nsfw_results if r["class"] in ["FEMALE_BREAST_EXPOSED", "MALE_GENITALIA_EXPOSED"]])
    
    # --- Final Output ---
    output = {
        "image_url": image_url,
        "blip_description": blip_description,
        "detected_objects": list(set(detected_objects)),
        "suggestive_score": suggestive_score,
        "is_cowboy": "cowboy hat" in blip_description.lower() or "cowboy" in blip_description.lower(),
        "people_count": sum(1 for obj in detected_objects if obj == "person"),
        "country_items": any(item in detected_objects for item in ["horse", "cow", "tractor"])
    }
    
    return output

# Main function
def main():
    # Initialize models (auto-download on first run)
    processor, blip_model, yolo_model, nsfw_detector = init_models()
    
    # Example image URLs (replace with your list)
    image_urls = [
        "https://images.pexels.com/photos/868097/pexels-photo-868097.jpeg?cs=srgb&dl=adventure-backpack-climb-868097.jpg&fm=jpg",
        "https://tse4.mm.bing.net/th?id=OIP.vohEdwvjB_CH8wMVOZYSlQHaE8&pid=Api&P=0&h=220"
    ]
    
    results = []
    for url in image_urls:
        print(f"Analyzing: {url}")
        results.append(analyze_image(url, processor, blip_model, yolo_model, nsfw_detector))
    
    # Save results
    os.makedirs("output", exist_ok=True)
    with open("output/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Analysis complete! Check /output/results.json")

if __name__ == "__main__":
    main()