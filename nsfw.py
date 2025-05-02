import cv2
import numpy as np
import pandas as pd

# Load the OpenNSFW model
deploy_prototxt = './nsfw_model/deploy.prototxt'
caffe_model = './nsfw_model/resnet_50_1by2_nsfw.caffemodel'
net = cv2.dnn.readNetFromCaffe(deploy_prototxt, caffe_model)

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found or unreadable: {image_path}")
    image_resized = cv2.resize(image, (224, 224))
    image_resized = image_resized.astype(np.float32) / 255.0
    blob = cv2.dnn.blobFromImage(image_resized, scalefactor=1.0, size=(224, 224),
                                 mean=(104.0, 117.0, 123.0), swapRB=True)
    return blob

# Function to categorize NSFW score
def get_nsfw_category(score):
    if score < 0.3:
        return "Safe"
    elif score < 0.6:
        return "Suggestive"
    else:
        return "Explicit"

# Function to predict NSFW score
def predict_nsfw(image_path):
    try:
        blob = preprocess_image(image_path)
        net.setInput(blob)
        output = net.forward()
        nsfw_score = float(output[0][1])
        nsfw_category = get_nsfw_category(nsfw_score)
        return {
            'image_path': image_path,
            'nsfw_score': round(nsfw_score, 4),
            'nsfw_category': nsfw_category
        }
    except Exception as e:
        return {
            'image_path': image_path,
            'nsfw_score': 'error',
            'nsfw_category': str(e)
        }

# Input image paths
image_paths = ['./output/jenna_waldrip.jpg']  # You can replace or load dynamically from CSV

# Analyze and save
results = [predict_nsfw(image_path) for image_path in image_paths]
df = pd.DataFrame(results)
df.to_csv('image_analysis_results.csv', index=False)

print("✅ All results saved to 'image_analysis_results.csv'")
