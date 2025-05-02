import instaloader
import os
import requests
import pandas as pd
from urllib.parse import urlparse

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
        else:
            print(f"❌ Failed to download image for {username}, status: {response.status_code}")

    except Exception as e:
        print(f"⚠️ Error processing {username}: {e}")

# ✅ Read Instagram URLs from samples.csv
df = pd.read_csv("samples.csv")

# ✅ Loop through the URLs and download profile pics
for url in df['User Homepage'].dropna():
    username = extract_username(url)
    if username:
        print(f"🔍 Extracting profile picture for: {username}")
        download_profile_pic(username)
    else:
        print(f"❌ Could not extract username from URL: {url}")
