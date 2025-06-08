#!/usr/bin/env python3
"""
Script untuk menampilkan lokasi cache PyTorch
"""

import torch
import os

def main():
    print("📁 PyTorch Cache Locations")
    print("=" * 40)
    
    # Torch hub directory
    hub_dir = torch.hub.get_dir()
    checkpoints_dir = os.path.join(hub_dir, 'checkpoints')
    
    print(f"Hub directory: {hub_dir}")
    print(f"Checkpoints directory: {checkpoints_dir}")
    
    # Create directories if they don't exist
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    print(f"\n📋 Files to download:")
    print(f"1. VGGFace2 model:")
    print(f"   URL: https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt")
    print(f"   Save to: {os.path.join(checkpoints_dir, '20180402-114759-vggface2.pt')}")
    
    print(f"\n2. CASIA-WebFace model (alternative):")
    print(f"   URL: https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt")
    print(f"   Save to: {os.path.join(checkpoints_dir, '20180408-102900-casia-webface.pt')}")
    
    # Check existing files
    print(f"\n📂 Current files in checkpoints directory:")
    if os.path.exists(checkpoints_dir):
        files = os.listdir(checkpoints_dir)
        if files:
            for file in files:
                filepath = os.path.join(checkpoints_dir, file)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {file} ({size_mb:.1f}MB)")
        else:
            print("  (empty)")
    else:
        print("  (directory does not exist)")

if __name__ == "__main__":
    main()