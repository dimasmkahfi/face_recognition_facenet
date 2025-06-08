import os
import cv2
import logging
from tqdm import tqdm
from face_detector_simple import FaceDetector
from facenet_model_casia import FaceNetModel
from config import Config

logger = logging.getLogger(__name__)

class DatasetProcessor:
    def __init__(self):
        """Initialize dataset processor"""
        self.face_detector = FaceDetector()
        self.facenet_model = FaceNetModel()
        
    def process_dataset(self, dataset_path=None):
        """
        Process dataset and register faces
        
        Args:
            dataset_path: Path to dataset directory (default: Config.DATASET_DIR)
            
        Returns:
            dict: Processing results
        """
        if dataset_path is None:
            dataset_path = Config.DATASET_DIR
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset directory not found: {dataset_path}")
            return {'success': False, 'message': 'Dataset directory not found'}
        
        results = {
            'success': True,
            'processed_identities': 0,
            'total_images': 0,
            'failed_images': 0,
            'identities': []
        }
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
        
        if not person_dirs:
            logger.error("No person directories found in dataset")
            return {'success': False, 'message': 'No person directories found'}
        
        logger.info(f"Found {len(person_dirs)} identities in dataset")
        
        # Process each person
        for person_name in tqdm(person_dirs, desc="Processing identities"):
            person_path = os.path.join(dataset_path, person_name)
            
            # Get all image files
            image_files = [f for f in os.listdir(person_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not image_files:
                logger.warning(f"No images found for {person_name}")
                continue
            
            logger.info(f"Processing {len(image_files)} images for {person_name}")
            
            face_images = []
            processed_count = 0
            failed_count = 0
            
            # Process each image
            for img_file in tqdm(image_files, desc=f"Processing {person_name}", leave=False):
                img_path = os.path.join(person_path, img_file)
                
                try:
                    # Read image
                    image = cv2.imread(img_path)
                    if image is None:
                        logger.warning(f"Failed to read image: {img_path}")
                        failed_count += 1
                        continue
                    
                    # Detect and extract face
                    face_img, confidence = self.face_detector.get_largest_face(image)
                    
                    if face_img is not None:
                        face_images.append(face_img)
                        processed_count += 1
                        logger.debug(f"Processed {img_file} (confidence: {confidence:.2f})")
                    else:
                        logger.warning(f"No face detected in {img_file}")
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    failed_count += 1
            
            # Register the person if we have face images
            if face_images:
                success = self.facenet_model.register_face(person_name, face_images)
                if success:
                    results['processed_identities'] += 1
                    results['identities'].append({
                        'name': person_name,
                        'processed_images': processed_count,
                        'failed_images': failed_count,
                        'total_images': len(image_files)
                    })
                    logger.info(f"Successfully registered {person_name} with {processed_count} faces")
                else:
                    logger.error(f"Failed to register {person_name}")
            else:
                logger.error(f"No valid faces found for {person_name}")
            
            results['total_images'] += len(image_files)
            results['failed_images'] += failed_count
        
        # Save embeddings
        if results['processed_identities'] > 0:
            self.facenet_model.save_embeddings()
            logger.info(f"Processing complete: {results['processed_identities']} identities registered")
        else:
            results['success'] = False
            results['message'] = 'No identities were successfully processed'
        
        return results
    
    def verify_single_image(self, image_path):
        """
        Verify a single image against registered faces
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Verification result
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'message': 'Failed to read image'}
            
            # Detect face
            face_img, confidence = self.face_detector.get_largest_face(image)
            
            if face_img is None:
                return {
                    'success': False, 
                    'message': 'No face detected',
                    'face_detected': False
                }
            
            # Verify face
            result = self.facenet_model.verify_face(face_img)
            result['face_detected'] = True
            result['detection_confidence'] = float(confidence)
            
            return result
            
        except Exception as e:
            logger.error(f"Error verifying image {image_path}: {e}")
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def get_dataset_info(self, dataset_path=None):
        """
        Get information about the dataset
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            dict: Dataset information
        """
        if dataset_path is None:
            dataset_path = Config.DATASET_DIR
        
        if not os.path.exists(dataset_path):
            return {'exists': False}
        
        info = {
            'exists': True,
            'path': dataset_path,
            'identities': [],
            'total_identities': 0,
            'total_images': 0
        }
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
        
        for person_name in person_dirs:
            person_path = os.path.join(dataset_path, person_name)
            
            # Count images
            image_files = [f for f in os.listdir(person_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            info['identities'].append({
                'name': person_name,
                'image_count': len(image_files)
            })
            info['total_images'] += len(image_files)
        
        info['total_identities'] = len(person_dirs)
        
        return info