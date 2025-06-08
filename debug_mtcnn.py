#!/usr/bin/env python3
"""
Debug script untuk menguji MTCNN dan mencari parameter yang kompatibel
"""

import sys

def test_mtcnn_import():
    """Test import MTCNN"""
    try:
        from mtcnn import MTCNN
        print("✅ MTCNN import successful")
        return True
    except ImportError as e:
        print(f"❌ MTCNN import failed: {e}")
        return False

def test_mtcnn_initialization():
    """Test berbagai cara inisialisasi MTCNN"""
    try:
        from mtcnn import MTCNN
        
        init_tests = [
            ("Basic initialization", lambda: MTCNN()),
            ("With min_face_size", lambda: MTCNN(min_face_size=40)),
            ("With factor", lambda: MTCNN(factor=0.709)),
            ("With min_face_size + factor", lambda: MTCNN(min_face_size=40, factor=0.709)),
            ("With thresholds", lambda: MTCNN(thresholds=[0.6, 0.7, 0.7])),
            ("With post_process", lambda: MTCNN(post_process=True)),
            ("Full parameters", lambda: MTCNN(
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True
            ))
        ]
        
        successful_methods = []
        
        for test_name, init_func in init_tests:
            try:
                detector = init_func()
                print(f"✅ {test_name}: SUCCESS")
                successful_methods.append(test_name)
            except Exception as e:
                print(f"❌ {test_name}: FAILED - {e}")
        
        return successful_methods
        
    except Exception as e:
        print(f"❌ Cannot test MTCNN initialization: {e}")
        return []

def test_mtcnn_detection():
    """Test deteksi wajah dengan MTCNN"""
    try:
        from mtcnn import MTCNN
        import numpy as np
        
        # Try basic initialization
        detector = MTCNN()
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test detection
        result = detector.detect_faces(dummy_image)
        print(f"✅ MTCNN detection test successful (found {len(result)} faces)")
        return True
        
    except Exception as e:
        print(f"❌ MTCNN detection test failed: {e}")
        return False

def test_opencv_alternative():
    """Test OpenCV sebagai alternatif"""
    try:
        import cv2
        
        # Test Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            print("❌ OpenCV Haar cascade failed to load")
            return False
        
        # Test detection
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        gray = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        print(f"✅ OpenCV Haar cascade available as fallback (found {len(faces)} faces)")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def check_mtcnn_version():
    """Check MTCNN version and parameters"""
    try:
        import mtcnn
        print(f"📦 MTCNN version: {getattr(mtcnn, '__version__', 'unknown')}")
        
        # Check MTCNN class signature
        from mtcnn import MTCNN
        import inspect
        
        sig = inspect.signature(MTCNN.__init__)
        params = list(sig.parameters.keys())
        print(f"📋 MTCNN.__init__ parameters: {params}")
        
        return params
        
    except Exception as e:
        print(f"❌ Cannot check MTCNN version: {e}")
        return []

def main():
    print("🔍 MTCNN Debug Tool")
    print("=" * 50)
    
    # Test import
    if not test_mtcnn_import():
        print("\n💡 Solution: Install MTCNN with:")
        print("   pip install mtcnn")
        return
    
    # Check version and parameters
    print("\n📦 Checking MTCNN version and parameters:")
    params = check_mtcnn_version()
    
    # Test initialization methods
    print("\n🧪 Testing MTCNN initialization methods:")
    successful_methods = test_mtcnn_initialization()
    
    if successful_methods:
        print(f"\n✅ Found {len(successful_methods)} working initialization methods:")
        for method in successful_methods:
            print(f"   - {method}")
        
        # Test detection
        print("\n🎯 Testing face detection:")
        test_mtcnn_detection()
    else:
        print("\n❌ No MTCNN initialization methods worked!")
        print("💡 Trying OpenCV alternative:")
        if test_opencv_alternative():
            print("\n✅ OpenCV Haar cascade can be used as fallback")
        else:
            print("\n❌ No face detection methods available")
    
    print("\n" + "=" * 50)
    print("🔧 Recommendations:")
    
    if successful_methods:
        print("✅ MTCNN is working. Use the robust face detector.")
        if 'Basic initialization' in successful_methods:
            print("   Recommended: Use MTCNN() with basic initialization")
        elif 'With min_face_size' in successful_methods:
            print("   Recommended: Use MTCNN(min_face_size=40)")
    else:
        print("❌ MTCNN has compatibility issues.")
        print("   Solution 1: Try different MTCNN version: pip install mtcnn==0.1.0")
        print("   Solution 2: Use OpenCV Haar cascade fallback")
        print("   Solution 3: Use face_detector_robust.py (automatically handles fallbacks)")

if __name__ == "__main__":
    main()