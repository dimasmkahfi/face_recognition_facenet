#!/usr/bin/env python3
"""
Test script for face verification system
Usage: python test_verification.py [image_path]
"""

import sys
import os
import argparse
import logging
import cv2
import matplotlib.pyplot as plt
from config import Config
from dataset_processor import DatasetProcessor

def setup_logging():
    """Setup logging configuration"""
    Config.create_directories()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def show_result(image_path, result):
    """Display verification result with image"""
    try:
        # Read and display image
        image = cv2.imread(image_path)
        if image is not None:
            # Convert BGR to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(image_rgb)
            
            # Create title based on result
            if result.get('face_detected', False):
                if result['success']:
                    title = f"✅ VERIFIED: {result['identity']}\n"
                    title += f"Confidence: {result['confidence']:.1f}% | "
                    title += f"Distance: {result['distance']:.3f} | "
                    title += f"Detection: {result['detection_confidence']:.3f}"
                    plt.title(title, color='green', fontsize=12, fontweight='bold')
                else:
                    title = f"❌ NOT RECOGNIZED\n"
                    if 'identity' in result:
                        title += f"Best match: {result['identity']} | "
                    title += f"Confidence: {result['confidence']:.1f}% | "
                    title += f"Distance: {result['distance']:.3f}"
                    plt.title(title, color='red', fontsize=12, fontweight='bold')
            else:
                plt.title("❌ NO FACE DETECTED", color='red', fontsize=14, fontweight='bold')
            
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            print(f"Error: Could not read image file: {image_path}")
            
    except Exception as e:
        print(f"Error displaying result: {e}")

def test_single_image(image_path):
    """Test verification on a single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    print(f"Testing image: {image_path}")
    print("-" * 50)
    
    # Initialize processor
    processor = DatasetProcessor()
    
    # Check if embeddings exist
    embeddings_file = os.path.join(Config.MODEL_DIR, 'face_embeddings.pkl')
    if not os.path.exists(embeddings_file):
        print("❌ No trained model found!")
        print("Please run training first: python train.py")
        return
    
    # Verify image
    result = processor.verify_single_image(image_path)
    
    # Print result
    print("Verification Result:")
    print("=" * 30)
    
    if result.get('face_detected', False):
        print(f"Face detected: ✅ (confidence: {result['detection_confidence']:.3f})")
        
        if result['success']:
            print(f"Identity: ✅ {result['identity']}")
            print(f"Verification: ✅ PASSED")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Distance: {result['distance']:.3f}")
        else:
            print(f"Identity: ❌ NOT RECOGNIZED")
            if 'identity' in result:
                print(f"Best match: {result['identity']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Distance: {result['distance']:.3f}")
    else:
        print("Face detected: ❌ NO FACE FOUND")
        print(f"Message: {result.get('message', 'Unknown error')}")
    
    print("=" * 30)
    
    # Show visual result
    try:
        show_result(image_path, result)
    except Exception as e:
        print(f"Could not display image: {e}")

def test_dataset(dataset_path=None):
    """Test verification on entire dataset"""
    if dataset_path is None:
        dataset_path = Config.DATASET_DIR
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found: {dataset_path}")
        return
    
    print(f"Testing dataset: {dataset_path}")
    print("-" * 50)
    
    # Initialize processor
    processor = DatasetProcessor()
    
    # Check if embeddings exist
    embeddings_file = os.path.join(Config.MODEL_DIR, 'face_embeddings.pkl')
    if not os.path.exists(embeddings_file):
        print("❌ No trained model found!")
        print("Please run training first: python train.py")
        return
    
    # Get dataset info
    dataset_info = processor.get_dataset_info(dataset_path)
    
    if not dataset_info['exists']:
        print("Dataset directory is empty or invalid")
        return
    
    print(f"Found {dataset_info['total_identities']} identities")
    print(f"Total images: {dataset_info['total_images']}")
    print()
    
    # Test each identity
    total_tests = 0
    correct_tests = 0
    false_positives = 0
    no_face_detected = 0
    
    for identity_info in dataset_info['identities']:
        identity_name = identity_info['name']
        identity_path = os.path.join(dataset_path, identity_name)
        
        print(f"Testing {identity_name}...")
        
        # Get all images for this identity
        image_files = [f for f in os.listdir(identity_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        identity_correct = 0
        identity_total = 0
        identity_no_face = 0
        
        for img_file in image_files:
            img_path = os.path.join(identity_path, img_file)
            result = processor.verify_single_image(img_path)
            
            total_tests += 1
            identity_total += 1
            
            if not result.get('face_detected', False):
                no_face_detected += 1
                identity_no_face += 1
                continue
            
            if result['success'] and result['identity'] == identity_name:
                correct_tests += 1
                identity_correct += 1
            elif result['success'] and result['identity'] != identity_name:
                false_positives += 1
        
        # Print identity results
        accuracy = (identity_correct / max(1, identity_total - identity_no_face)) * 100
        print(f"  {identity_name}: {identity_correct}/{identity_total - identity_no_face} correct ({accuracy:.1f}%)")
        if identity_no_face > 0:
            print(f"  No face detected in {identity_no_face} images")
    
    # Print overall results
    print("\n" + "=" * 50)
    print("OVERALL TEST RESULTS")
    print("=" * 50)
    
    valid_tests = total_tests - no_face_detected
    if valid_tests > 0:
        accuracy = (correct_tests / valid_tests) * 100
        false_positive_rate = (false_positives / valid_tests) * 100
        
        print(f"Total images tested: {total_tests}")
        print(f"Valid face detections: {valid_tests}")
        print(f"No face detected: {no_face_detected}")
        print(f"Correct identifications: {correct_tests}")
        print(f"False positives: {false_positives}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"False positive rate: {false_positive_rate:.1f}%")
    else:
        print("No valid tests completed")
    
    print("=" * 50)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test face verification system')
    parser.add_argument('image_path', nargs='?', help='Path to image file to test')
    parser.add_argument('--dataset', help='Test entire dataset instead of single image')
    parser.add_argument('--show-plot', action='store_true', help='Show matplotlib plots')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Create directories
    Config.create_directories()
    
    if args.dataset:
        # Test entire dataset
        test_dataset(args.dataset)
    elif args.image_path:
        # Test single image
        if not args.show_plot:
            # Disable matplotlib GUI
            import matplotlib
            matplotlib.use('Agg')
        test_single_image(args.image_path)
    else:
        # Interactive mode
        print("Face Verification Test Tool")
        print("=" * 30)
        print("1. Test single image")
        print("2. Test entire dataset")
        print("3. Exit")
        
        while True:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                img_path = input("Enter path to image file: ").strip()
                if img_path:
                    test_single_image(img_path)
            elif choice == '2':
                dataset_path = input(f"Enter dataset path (default: {Config.DATASET_DIR}): ").strip()
                if not dataset_path:
                    dataset_path = None
                test_dataset(dataset_path)
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-3.")

if __name__ == "__main__":
    main()