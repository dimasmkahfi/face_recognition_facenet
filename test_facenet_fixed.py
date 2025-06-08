#!/usr/bin/env python3
"""
Test script untuk FaceNet dengan fix batch normalization
"""

import torch
import numpy as np

def test_facenet_loading():
    """Test FaceNet model loading dengan berbagai strategi"""
    print("🧪 Testing FaceNet Model Loading")
    print("=" * 40)
    
    from facenet_pytorch import InceptionResnetV1
    
    strategies = [
        ('CASIA-WebFace (eval mode)', lambda: InceptionResnetV1(pretrained='casia-webface').eval()),
        ('No pretrained (eval mode)', lambda: InceptionResnetV1(pretrained=None).eval()),
        ('VGGFace2 (eval mode)', lambda: InceptionResnetV1(pretrained='vggface2').eval())
    ]
    
    working_models = []
    
    for name, strategy in strategies:
        try:
            print(f"\n🔄 Testing: {name}")
            
            # Load model
            model = strategy()
            print(f"  ✅ Model loaded")
            
            # Test with single input (batch size 1)
            dummy_input = torch.randn(1, 3, 160, 160)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"  ✅ Single input test passed (output shape: {output.shape})")
            
            # Test with batch input (batch size 2)
            batch_input = torch.randn(2, 3, 160, 160)
            
            with torch.no_grad():
                batch_output = model(batch_input)
            
            print(f"  ✅ Batch input test passed (output shape: {batch_output.shape})")
            
            working_models.append(name)
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    return working_models

def test_facenet_embedding():
    """Test embedding generation"""
    print("\n🧪 Testing Embedding Generation")
    print("=" * 40)
    
    try:
        from facenet_pytorch import InceptionResnetV1
        
        # Use CASIA-WebFace since it's working
        model = InceptionResnetV1(pretrained='casia-webface').eval()
        
        # Create fake face image
        face_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        
        # Preprocess
        face_normalized = (face_image.astype(np.float32) - 127.5) / 128.0
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Get embedding
        with torch.no_grad():
            embedding = model(face_tensor)
            embedding = embedding.cpu().numpy().flatten()
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        print(f"✅ Embedding generated successfully")
        print(f"   Shape: {embedding.shape}")
        print(f"   Norm: {np.linalg.norm(embedding):.6f}")
        print(f"   Range: [{embedding.min():.3f}, {embedding.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

def test_our_facenet_model():
    """Test our custom FaceNet model class"""
    print("\n🧪 Testing Our FaceNet Model Class")
    print("=" * 40)
    
    try:
        from facenet_model_fixed import FaceNetModel
        
        # Initialize model
        model = FaceNetModel(pretrained=True)
        
        # Get model info
        info = model.get_model_info()
        print(f"✅ Model initialized")
        print(f"   Device: {info['device']}")
        print(f"   Parameters: {info['parameters']:,}")
        
        # Test embedding generation
        fake_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        embedding = model.get_embedding(fake_face)
        
        if embedding is not None:
            print(f"✅ Embedding test passed")
            print(f"   Shape: {embedding.shape}")
            print(f"   Norm: {np.linalg.norm(embedding):.6f}")
            return True
        else:
            print(f"❌ Embedding generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Our model test failed: {e}")
        return False

def main():
    print("🔧 FaceNet Model Tester")
    print("=" * 50)
    
    # Test 1: Basic model loading
    working_models = test_facenet_loading()
    
    # Test 2: Embedding generation
    embedding_test = test_facenet_embedding()
    
    # Test 3: Our custom model
    our_model_test = test_our_facenet_model()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    if working_models:
        print(f"✅ Working models: {', '.join(working_models)}")
    else:
        print("❌ No models working")
    
    if embedding_test:
        print("✅ Embedding generation: PASSED")
    else:
        print("❌ Embedding generation: FAILED")
    
    if our_model_test:
        print("✅ Our model class: PASSED")
        print("\n🎉 All tests passed! You can now run:")
        print("  python train.py")
    else:
        print("❌ Our model class: FAILED")
        print("\n🛠️ Need to fix model issues first")

if __name__ == "__main__":
    main()