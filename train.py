#!/usr/bin/env python3
"""
Training script for face verification system
Usage: python train.py [dataset_path]
"""

import sys
import os
import logging
import argparse
from config import Config
from dataset_processor import DatasetProcessor

def setup_logging():
    """Setup logging configuration"""
    Config.create_directories()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Config.LOGS_DIR, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train face verification system')
    parser.add_argument('dataset_path', nargs='?', default=None,
                       help='Path to dataset directory (default: ./dataset)')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Verification threshold (default: 0.6)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting face verification training")
    logger.info(f"Dataset path: {args.dataset_path or Config.DATASET_DIR}")
    logger.info(f"Verification threshold: {args.threshold}")
    
    # Initialize dataset processor
    try:
        processor = DatasetProcessor()
        
        # Get dataset info
        dataset_info = processor.get_dataset_info(args.dataset_path)
        
        if not dataset_info['exists']:
            logger.error(f"Dataset directory not found: {args.dataset_path or Config.DATASET_DIR}")
            logger.info("Please create a dataset directory with the following structure:")
            logger.info("dataset/")
            logger.info("├── person1/")
            logger.info("│   ├── person1_0001.jpg")
            logger.info("│   ├── person1_0002.jpg")
            logger.info("│   └── ...")
            logger.info("├── person2/")
            logger.info("│   ├── person2_0001.jpg")
            logger.info("│   └── ...")
            logger.info("└── ...")
            return
        
        logger.info(f"Dataset found: {dataset_info['total_identities']} identities, {dataset_info['total_images']} images")
        
        # Display dataset information
        print("\nDataset Information:")
        print("=" * 50)
        for identity in dataset_info['identities']:
            print(f"  {identity['name']}: {identity['image_count']} images")
        print("=" * 50)
        print(f"Total: {dataset_info['total_identities']} identities, {dataset_info['total_images']} images")
        
        # Confirm before processing
        response = input("\nProceed with training? (y/n): ")
        if response.lower() != 'y':
            logger.info("Training cancelled by user")
            return
        
        # Process dataset
        logger.info("Starting dataset processing...")
        results = processor.process_dataset(args.dataset_path)
        
        if results['success']:
            print("\n" + "=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Processed identities: {results['processed_identities']}")
            print(f"Total images processed: {results['total_images']}")
            print(f"Failed images: {results['failed_images']}")
            print("\nRegistered identities:")
            for identity in results['identities']:
                print(f"  - {identity['name']}: {identity['processed_images']}/{identity['total_images']} images")
            print(f"\nEmbeddings saved to: {os.path.join(Config.MODEL_DIR, 'face_embeddings.pkl')}")
            print("You can now run the API server using: python api_server.py")
        else:
            logger.error(f"Training failed: {results.get('message', 'Unknown error')}")
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()