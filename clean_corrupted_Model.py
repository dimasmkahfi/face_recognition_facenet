#!/usr/bin/env python3
"""
Script untuk membersihkan model VGGFace2 yang korup
"""

import os
import torch

def clean_corrupted_vggface2():
    """Hapus model VGGFace2 yang korup"""
    print("🧹 Cleaning corrupted VGGFace2 model")
    print("=" * 40)
    
    # Get cache directory
    cache_dir = torch.hub.get_dir()
    checkpoints_dir = os.path.join(cache_dir, 'checkpoints')
    
    vggface2_file = os.path.join(checkpoints_dir, '20180402-114759-vggface2.pt')
    
    if os.path.exists(vggface2_file):
        size_mb = os.path.getsize(vggface2_file) / (1024 * 1024)
        print(f"📁 Found VGGFace2 model: {size_mb:.1f}MB")
        
        # Expected size should be around 89MB
        if size_mb < 85:  # If significantly smaller
            print(f"⚠️ File appears corrupted (too small)")
            try:
                os.remove(vggface2_file)
                print(f"🗑️ Removed corrupted VGGFace2 model")
            except Exception as e:
                print(f"❌ Failed to remove file: {e}")
        else:
            print(f"ℹ️ File size looks OK, keeping it")
    else:
        print(f"ℹ️ VGGFace2 model not found in cache")
    
    # List remaining files
    print(f"\n📂 Remaining model files:")
    if os.path.exists(checkpoints_dir):
        files = os.listdir(checkpoints_dir)
        for file in files:
            filepath = os.path.join(checkpoints_dir, file)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✅ {file} ({size_mb:.1f}MB)")
    else:
        print(f"  (no files)")

def main():
    clean_corrupted_vggface2()

if __name__ == "__main__":
    main()