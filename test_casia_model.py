#!/usr/bin/env python3
"""
Quick test untuk CASIA model
"""

import numpy as np

def test_casia_model():
    """Test CASIA FaceNet model"""
    print("🧪 Testing CASIA FaceNet Model")
    print("=" * 40)
    
    try:
        from facenet_model_casia import FaceNetModel
        
        # Initialize model
        print("🔄 Initializing model...")
        model = FaceNetModel()
        
        # Get model info
        info = model.get_model_info()
        print(f"✅ Model loaded successfully")
        print(f"   Type: {info['model_type']}")
        print(f"   Device: {info['device']}")
        print(f"   Parameters: {info['parameters']:,}")
        
        # Test embedding generation
        print("\n🔄 Testing embedding generation...")
        fake_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        
        embedding = model.get_embedding(fake_face)
        
        if embedding is not None:
            print(f"✅ Embedding generated successfully")
            print(f"   Shape: {embedding.shape}")
            print(f"   Norm: {np.linalg.norm(embedding):.6f}")
            print(f"   Range: [{embedding.min():.3f}, {embedding.max():.3f}]")
        else:
            print(f"❌ Failed to generate embedding")
            return False
        
        # Test face registration
        print("\n🔄 Testing face registration...")
        fake_faces = [np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8) for _ in range(3)]
        
        success = model.register_face("test_person", fake_faces)
        
        if success:
            print(f"✅ Face registration successful")
            identities = model.get_registered_identities()
            print(f"   Registered identities: {identities}")
        else:
            print(f"❌ Face registration failed")
            return False
        
        # Test face verification
        print("\n🔄 Testing face verification...")
        test_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        
        result = model.verify_face(test_face)
        
        print(f"✅ Face verification completed")
        print(f"   Success: {result['success']}")
        print(f"   Message: {result.get('message', 'N/A')}")
        if 'distance' in result:
            print(f"   Distance: {result['distance']:.4f}")
            print(f"   Similarity: {result['similarity']:.4f}")
        
        print(f"\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 CASIA FaceNet Model Tester")
    print("=" * 50)
    
    success = test_casia_model()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("  python train.py")
        print("  python api_server.py")
    else:
        print("❌ TESTS FAILED!")
        print("\nTroubleshooting:")
        print("1. Make sure CASIA-WebFace model downloaded correctly")
        print("2. Check PyTorch installation")
        print("3. Try: python download_models.py")

if __name__ == "__main__":
    main()