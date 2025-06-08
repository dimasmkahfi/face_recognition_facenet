#!/usr/bin/env python3
"""
Quick setup script untuk face verification system
"""

import os
import sys

def create_directories():
    """Create necessary directories"""
    dirs = ['dataset', 'models', 'temp', 'logs']
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created directory: {dir_name}")
        else:
            print(f"📁 Directory exists: {dir_name}")

def create_sample_dataset_structure():
    """Create sample dataset structure"""
    sample_persons = ['John_Doe', 'Jane_Smith', 'Ahmad_Budi']
    
    for person in sample_persons:
        person_dir = os.path.join('dataset', person)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            
            # Create README file with instructions
            readme_content = f"""
# {person} Dataset Folder

Place images of {person} in this folder with naming convention:
- {person.lower()}_0001.jpg
- {person.lower()}_0002.jpg
- {person.lower()}_0003.jpg
- etc.

Minimum 3-5 images recommended for good recognition accuracy.

Image requirements:
- Clear face visible
- Good lighting
- Minimal blur
- Various angles (front, slight left/right)
- Formats: JPG, JPEG, PNG, BMP
"""
            
            with open(os.path.join(person_dir, 'README.txt'), 'w') as f:
                f.write(readme_content)
            
            print(f"📁 Created sample directory: {person_dir}")

def test_imports():
    """Test required imports"""
    print("\n🔍 Testing imports...")
    
    required_modules = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('facenet_pytorch', 'FaceNet-PyTorch'),
        ('mtcnn', 'MTCNN'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('flask', 'Flask'),
        ('PIL', 'Pillow')
    ]
    
    missing_modules = []
    
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️ Missing modules: {', '.join(missing_modules)}")
        print("Install with: pip install " + " ".join(missing_modules))
        return False
    
    return True

def test_face_detector():
    """Test face detector initialization"""
    print("\n🔍 Testing face detector...")
    
    try:
        from face_detector_simple import FaceDetector
        
        detector = FaceDetector()
        info = detector.get_detector_info()
        
        print(f"✅ Face detector initialized: {info['type']}")
        return True
        
    except Exception as e:
        print(f"❌ Face detector failed: {e}")
        return False

def test_facenet_model():
    """Test FaceNet model loading"""
    print("\n🔍 Testing FaceNet model...")
    
    try:
        from facenet_model import FaceNetModel
        
        model = FaceNetModel()
        print("✅ FaceNet model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ FaceNet model failed: {e}")
        return False

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. Add images to dataset folders:")
    print("   - dataset/John_Doe/john_doe_0001.jpg")
    print("   - dataset/Jane_Smith/jane_smith_0001.jpg")
    print("   - etc. (minimum 3-5 images per person)")
    
    print("\n2. Train the model:")
    print("   python train.py")
    
    print("\n3. Test the system:")
    print("   python test_face_detector.py")
    print("   python test_verification.py path/to/test/image.jpg")
    
    print("\n4. Start API server:")
    print("   python api_server.py")
    print("   python api_server.py --ngrok  # for external access")
    
    print("\n📂 Dataset Structure:")
    print("   dataset/")
    print("   ├── John_Doe/")
    print("   │   ├── john_doe_0001.jpg")
    print("   │   ├── john_doe_0002.jpg")
    print("   │   └── john_doe_0003.jpg")
    print("   ├── Jane_Smith/")
    print("   │   ├── jane_smith_0001.jpg")
    print("   │   └── jane_smith_0002.jpg")
    print("   └── ...")

def main():
    print("🚀 Face Verification System - Quick Setup")
    print("=" * 60)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Create sample dataset structure
    print("\n📂 Creating sample dataset structure...")
    create_sample_dataset_structure()
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup incomplete - missing dependencies")
        print("Run: pip install -r requirements_minimal.txt")
        return
    
    # Test face detector
    if not test_face_detector():
        print("\n⚠️ Face detector issues detected")
        print("The system may still work with OpenCV fallback")
    
    # Test FaceNet model
    if not test_facenet_model():
        print("\n❌ FaceNet model issues detected")
        print("Check PyTorch and facenet-pytorch installation")
        return
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()