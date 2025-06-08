#!/usr/bin/env python3
"""
Script untuk download model FaceNet secara manual
"""

import os
import requests
import torch
from tqdm import tqdm

def get_torch_cache_dir():
    """Get torch cache directory"""
    cache_dir = torch.hub.get_dir()
    checkpoints_dir = os.path.join(cache_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    return checkpoints_dir

def download_file(url, filepath, desc):
    """Download file with progress bar"""
    print(f"📥 Downloading {desc}...")
    print(f"URL: {url}")
    print(f"To: {filepath}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ Downloaded {desc} successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {desc}: {e}")
        return False

def verify_file(filepath, min_size_mb=25):
    """Verify downloaded file"""
    if not os.path.exists(filepath):
        return False, "File not found"
    
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    if size_mb < min_size_mb:
        return False, f"File too small ({size_mb:.1f}MB, expected >{min_size_mb}MB)"
    
    return True, f"File OK ({size_mb:.1f}MB)"

def main():
    print("🚀 FaceNet Model Downloader")
    print("=" * 50)
    
    # Get cache directory
    cache_dir = get_torch_cache_dir()
    print(f"📁 Cache directory: {cache_dir}")
    
    # Model URLs and filenames
    models = [
        {
            'name': 'VGGFace2',
            'filename': '20180402-114759-vggface2.pt',
            'url': 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt',
            'size_mb': 89.0
        },
        {
            'name': 'CASIA-WebFace',
            'filename': '20180408-102900-casia-webface.pt',
            'url': 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt',
            'size_mb': 89.0
        }
    ]
    
    downloaded_models = []
    
    for model in models:
        filepath = os.path.join(cache_dir, model['filename'])
        
        # Check if file already exists and is valid
        valid, msg = verify_file(filepath, model['size_mb'] * 0.9)  # 90% of expected size
        
        if valid:
            print(f"✅ {model['name']} already exists: {msg}")
            downloaded_models.append(model['name'])
            continue
        
        if os.path.exists(filepath):
            print(f"⚠️ {model['name']} exists but invalid: {msg}")
            print(f"🗑️ Removing corrupted file...")
            os.remove(filepath)
        
        # Download the model
        success = download_file(model['url'], filepath, model['name'])
        
        if success:
            # Verify download
            valid, msg = verify_file(filepath, model['size_mb'] * 0.9)
            if valid:
                print(f"✅ {model['name']} verified: {msg}")
                downloaded_models.append(model['name'])
            else:
                print(f"❌ {model['name']} verification failed: {msg}")
        
        print("-" * 30)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 50)
    
    if downloaded_models:
        print(f"✅ Successfully downloaded: {', '.join(downloaded_models)}")
        print("\n🧪 Testing model loading...")
        
        # Test model loading
        try:
            from facenet_pytorch import InceptionResnetV1
            
            for model_name in downloaded_models:
                try:
                    if 'VGGFace2' in model_name:
                        model = InceptionResnetV1(pretrained='vggface2')
                    else:
                        model = InceptionResnetV1(pretrained='casia-webface')
                    
                    print(f"✅ {model_name} loads successfully")
                    
                    # Test with dummy input
                    dummy_input = torch.randn(1, 3, 160, 160)
                    with torch.no_grad():
                        output = model(dummy_input)
                    
                    print(f"✅ {model_name} inference test passed")
                    break  # One working model is enough
                    
                except Exception as e:
                    print(f"❌ {model_name} loading failed: {e}")
            
        except Exception as e:
            print(f"❌ Model testing failed: {e}")
        
        print("\n🎉 You can now run:")
        print("  python quick_setup.py")
        print("  python train.py")
        
    else:
        print("❌ No models downloaded successfully")
        print("\n🛠️ Manual solutions:")
        print("1. Check internet connection")
        print("2. Download manually from browser:")
        print("   https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt")
        print(f"3. Save to: {cache_dir}")
        print("4. Run this script again to verify")

if __name__ == "__main__":
    main()