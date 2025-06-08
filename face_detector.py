import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import logging
from config import Config

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        """Initialize MTCNN face detector"""
        try:
            # Try different MTCNN initialization approaches for compatibility
            try:
                # Method 1: Full parameters (newer versions)
                self.detector = MTCNN(
                    min_face_size=Config.MIN_FACE_SIZE,
                    thresholds=Config.DETECTION_THRESHOLD,
                    factor=0.709,
                    post_process=True
                )
                logger.info("MTCNN initialized with full parameters")
            except TypeError:
                # Method 2: Basic parameters (older versions)
                try:
                    self.detector = MTCNN(
                        min_face_size=Config.MIN_FACE_SIZE,
                        factor=0.709
                    )
                    logger.info("MTCNN initialized with basic parameters")
                except TypeError:
                    # Method 3: Default parameters only
                    self.detector = MTCNN()
                    logger.info("MTCNN initialized with default parameters")
            
            logger.info("MTCNN face detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {e}")
            raise
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image: numpy array (BGR format) or PIL Image
            
        Returns:
            list: List of face bounding boxes and landmarks
        """
        try:
            # Convert BGR to RGB if numpy array
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Assume BGR format from OpenCV
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
            else:
                # PIL Image
                image_rgb = np.array(image)
            
            # Detect faces
            results = self.detector.detect_faces(image_rgb)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def extract_face(self, image, bbox, target_size=Config.IMAGE_SIZE, margin=10):
        """
        Extract and align face from image
        
        Args:
            image: Input image (numpy array)
            bbox: Bounding box dict from MTCNN
            target_size: Target size for the extracted face
            margin: Margin around the face
            
        Returns:
            numpy array: Extracted and resized face
        """
        try:
            x, y, width, height = bbox['box']
            
            # Add margin
            x = max(0, x - margin)
            y = max(0, y - margin)
            width = min(image.shape[1] - x, width + 2 * margin)
            height = min(image.shape[0] - y, height + 2 * margin)
            
            # Extract face
            face = image[y:y+height, x:x+width]
            
            # Resize to target size
            face_resized = cv2.resize(face, target_size)
            
            return face_resized
            
        except Exception as e:
            logger.error(f"Error extracting face: {e}")
            return None
    
    def get_largest_face(self, image):
        """
        Get the largest face from an image
        
        Args:
            image: Input image
            
        Returns:
            tuple: (face_image, confidence) or (None, None) if no face found
        """
        try:
            faces = self.detect_faces(image)
            
            if not faces:
                return None, None
            
            # Find the largest face
            largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
            
            # Extract the face
            if isinstance(image, np.ndarray):
                face_img = self.extract_face(image, largest_face)
            else:
                # Convert PIL to numpy
                image_np = np.array(image)
                face_img = self.extract_face(image_np, largest_face)
            
            confidence = largest_face['confidence']
            
            return face_img, confidence
            
        except Exception as e:
            logger.error(f"Error getting largest face: {e}")
            return None, None