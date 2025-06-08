#!/usr/bin/env python3
"""
Script untuk membersihkan cache model yang korup dan re-download
"""

import os
import shutil
import torch

def clear_torch_cache():
    """Clear PyTorch cache"""
    print("🧹 Clearing PyTorch cache...")
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ CUDA cache cleared")
    
    # Get torch cache directory
    torch_cache = torch.hub.get_dir()
    print(f"📁 Torch cache directory: {torch_cache}")
    
    return torch_cache

def clear_facenet_cache():
    """Clear FaceNet model cache"""
    print("\n🧹 Clearing FaceNet cache...")
    
    # Common cache locations
    cache_locations = []
    
    # Torch hub cache
    torch_cache = torch.hub.get_dir()
    cache_locations.append(torch_cache)
    
    # User home cache
    home_cache = os.path.expanduser("~/.cache")
    if os.path.exists(home_cache):
        cache_locations.append(os.path.join(home_cache, "torch"))
    
    # Current directory cache
    if os.path.exists(".cache"):
        cache_locations.append(".cache")
    
    # Look for facenet-pytorch cache
    for cache_dir in cache_locations:
        if os.path.exists(cache_dir):
            for item in os.listdir(cache_dir):
                if "facenet" in item.lower() or "inception" in item.lower():
                    item_path = os.path.join(cache_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            print(f"🗑️ Removed file: {item_path}")
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            print(f"🗑️ Removed directory: {item_path}")
                    except Exception as e:
                        print(f"⚠️ Could not remove {item_path}: {e}")

def test_model_loading():
    """Test model loading dengan berbagai strategi"""
    print("\n🧪 Testing model loading strategies...")
    
    try:
        from facenet_pytorch import InceptionResnetV1
        
        strategies = [
            ("No pretrained", lambda: InceptionResnetV1(pretrained=None)),
            ("CASIA-WebFace", lambda: InceptionResnetV1(pretrained='casia-webface')),
            ("VGGFace2", lambda: InceptionResnetV1(pretrained='vggface2'))
        ]
        
        working_strategies = []
        
        for name, strategy in strategies:
            try:
                print(f"  Testing {name}...")
                model = strategy()
                
                # Test with dummy input
                dummy_input = torch.randn(1, 3, 160, 160)
                with torch.no_grad():
                    output = model(dummy_input)
                
                print(f"  ✅ {name}: SUCCESS")
                working_strategies.append(name)
                
                # Clean up
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"  ❌ {name}: FAILED - {e}")
        
        return working_strategies
        
    except Exception as e:
        print(f"❌ Cannot test model loading: {e}")
        return []

def force_redownload():
    """Force re-download of model weights"""
    print("\n🔄 Forcing model re-download...")
    
    try:
        from facenet_pytorch import InceptionResnetV1
        
        # This will force re-download
        torch.hub.set_dir("./temp_torch_cache")
        
        print("📥 Downloading fresh model weights...")
        model = InceptionResnetV1(pretrained='vggface2')
        
        print("✅ Fresh model downloaded successfully")
        
        # Clean up temp cache
        if os.path.exists("./temp_torch_cache"):
            shutil.rmtree("./temp_torch_cache")
        
        return True
        
    except Exception as e:
        print(f"❌ Force re-download failed: {e}")
        return False

def main():
    print("🔧 FaceNet Model Cache Fixer")
    print("=" * 50)
    
    # Step 1: Clear existing cache
    clear_torch_cache()
    clear_facenet_cache()
    
    # Step 2: Test model loading
    working_strategies = test_model_loading()
    
    if working_strategies:
        print(f"\n✅ Working strategies: {', '.join(working_strategies)}")
        print("🎉 Model loading issues resolved!")
        
        print("\nℹ️ You can now run:")
        print("  python quick_setup.py")
        print("  python train.py")
        
    else:
        print("\n❌ All strategies failed. Trying force re-download...")
        
        if force_redownload():
            print("✅ Force re-download successful!")
        else:
            print("❌ All methods failed.")
            print("\n🛠️ Manual solutions:")
            print("1. Check internet connection")
            print("2. Try: pip uninstall facenet-pytorch && pip install facenet-pytorch")
            print("3. Use offline mode: FaceNetModel(pretrained=False)")
            print("4. Check firewall/proxy settings")

if __name__ == "__main__":
    main()