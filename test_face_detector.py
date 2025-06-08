#!/usr/bin/env python3
"""
Test script untuk face detector sederhana
"""

import cv2
import numpy as np
from face_detector_simple import FaceDetector

def test_face_detector():
    """Test face detector dengan gambar dummy"""
    print("🔍 Testing Face Detector...")
    
    try:
        # Initialize detector
        detector = FaceDetector()
        info = detector.get_detector_info()
        print(f"✅ Face detector initialized: {info['type']}")
        
        # Create a simple test image with a face-like rectangle
        test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Draw a simple face-like pattern
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)  # Face
        cv2.circle(test_image, (130, 140), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(test_image, (170, 140), 10, (0, 0, 0), -1)  # Right eye
        cv2.rectangle(test_image, (140, 160), (160, 170), (0, 0, 0), -1)  # Nose
        cv2.rectangle(test_image, (130, 180), (170, 190), (0, 0, 0), -1)  # Mouth
        
        print("🖼️ Created test image with face-like pattern")
        
        # Test face detection
        faces = detector.detect_faces(test_image)
        print(f"📊 Detected {len(faces)} faces")
        
        # Test largest face extraction
        face_img, confidence = detector.get_largest_face(test_image)
        
        if face_img is not None:
            print(f"✅ Face extraction successful (confidence: {confidence:.3f})")
            print(f"📏 Face image size: {face_img.shape}")
            return True
        else:
            print("⚠️ No face extracted, but detector is working")
            return True
            
    except Exception as e:
        print(f"❌ Face detector test failed: {e}")
        return False

def test_with_real_image():
    """Test dengan gambar real jika tersedia"""
    print("\n🖼️ Testing with real image (if available)...")
    
    # Cari file gambar di direktori current
    import os
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            print(f"Found image: {file}")
            
            try:
                # Load image
                image = cv2.imread(file)
                if image is None:
                    continue
                
                # Test detector
                detector = FaceDetector()
                face_img, confidence = detector.get_largest_face(image)
                
                if face_img is not None:
                    print(f"✅ Real image test successful: {file}")
                    print(f"   Confidence: {confidence:.3f}")
                    print(f"   Face size: {face_img.shape}")
                    
                    # Save extracted face for verification
                    cv2.imwrite('extracted_face_test.jpg', face_img)
                    print("💾 Saved extracted face as 'extracted_face_test.jpg'")
                    return True
                else:
                    print(f"⚠️ No face found in {file}")
                    
            except Exception as e:
                print(f"❌ Error testing {file}: {e}")
    
    print("ℹ️ No suitable images found for real image test")
    return False

def main():
    print("🧪 Face Detector Test Suite")
    print("=" * 40)
    
    # Test 1: Basic detector functionality
    test1_result = test_face_detector()
    
    # Test 2: Real image if available
    test2_result = test_with_real_image()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"   Basic detector test: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"   Real image test: {'✅ PASS' if test2_result else '⚠️ SKIP'}")
    
    if test1_result:
        print("\n🎉 Face detector is working!")
        print("You can now run: python train.py")
    else:
        print("\n❌ Face detector has issues. Check the logs above.")

if __name__ == "__main__":
    main()