#!/usr/bin/env python3
"""
Remove corrupted VGGFace2 model
"""

import os
import torch

def remove_corrupted_vggface2():
    """Remove the corrupted VGGFace2 model file"""
    print("🗑️ Removing Corrupted VGGFace2 Model")
    print("=" * 40)
    
    # Get torch cache directory
    cache_dir = torch.hub.get_dir()
    checkpoints_dir = os.path.join(cache_dir, 'checkpoints')
    
    vggface2_files = [
        '20180402-114759-vggface2.pt',
        'vggface2.pt'  # Alternative naming
    ]
    
    removed_files = []
    
    for filename in vggface2_files:
        filepath = os.path.join(checkpoints_dir, filename)
        
        if os.path.exists(filepath):
            try:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"📁 Found: {filename} ({size_mb:.1f}MB)")
                
                os.remove(filepath)
                print(f"🗑️ Removed: {filename}")
                removed_files.append(filename)
                
            except Exception as e:
                print(f"❌ Failed to remove {filename}: {e}")
        else:
            print(f"ℹ️ Not found: {filename}")
    
    if removed_files:
        print(f"\n✅ Removed {len(removed_files)} corrupted file(s)")
        print("This will force using CASIA-WebFace model only")
    else:
        print(f"\nℹ️ No VGGFace2 files found to remove")
    
    # Show remaining files
    print(f"\n📂 Remaining model files:")
    if os.path.exists(checkpoints_dir):
        files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
        if files:
            for file in files:
                filepath = os.path.join(checkpoints_dir, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  ✅ {file} ({size_mb:.1f}MB)")
        else:
            print("  (no .pt files)")
    else:
        print("  (checkpoints directory not found)")

def main():
    remove_corrupted_vggface2()
    
    print(f"\n🎯 Next steps:")
    print("1. python test_casia_model.py")
    print("2. python train.py")

if __name__ == "__main__":
    main()