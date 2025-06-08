import cv2
import numpy as np
import logging
from config import Config

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        """Initialize face detector with fallback options"""
        self.detector = None
        self.detector_type = None
        
        # Try to initialize MTCNN first
        if self._init_mtcnn():
            return
        
        # Fallback to OpenCV Haar Cascades
        if self._init_opencv():
            return
        
        # Fallback to DNN face detector
        if self._init_dnn():
            return
        
        raise Exception("No face detector could be initialized")
    
    def _init_mtcnn(self):
        """Try to initialize MTCNN"""
        try:
            from mtcnn import MTCNN
            
            # Try different initialization methods
            init_methods = [
                # Method 1: Basic initialization
                lambda: MTCNN(),
                
                # Method 2: With min_face_size only
                lambda: MTCNN(min_face_size=Config.MIN_FACE_SIZE),
                
                # Method 3: With additional parameters if supported
                lambda: MTCNN(
                    min_face_size=Config.MIN_FACE_SIZE,
                    factor=0.709
                )
            ]
            
            for i, init_method in enumerate(init_methods):
                try:
                    self.detector = init_method()
                    self.detector_type = "MTCNN"
                    logger.info(f"MTCNN initialized successfully (method {i+1})")
                    return True
                except Exception as e:
                    logger.debug(f"MTCNN init method {i+1} failed: {e}")
                    continue
            
            return False
            
        except ImportError:
            logger.warning("MTCNN not available, trying fallback detectors")
            return False
        except Exception as e:
            logger.warning(f"MTCNN initialization failed: {e}")
            return False
    
    def _init_opencv(self):
        """Initialize OpenCV Haar Cascade detector"""
        try:
            # Try to load Haar cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            
            if self.detector.empty():
                return False
            
            self.detector_type = "OpenCV_Haar"
            logger.info("OpenCV Haar Cascade detector initialized")
            return True
            
        except Exception as e:
            logger.warning(f"OpenCV Haar initialization failed: {e}")
            return False
    
    def _init_dnn(self):
        """Initialize OpenCV DNN face detector"""
        try:
            # You can download these files and place them in the project directory
            # For now, we'll skip this and return False
            logger.warning("DNN face detector not implemented yet")
            return False
            
        except Exception as e:
            logger.warning(f"DNN detector initialization failed: {e}")
            return False
    
    def detect_faces(self, image):
        """
        Detect faces in an image using the available detector
        
        Args:
            image: numpy array (BGR format) or PIL Image
            
        Returns:
            list: List of face detection results
        """
        try:
            if self.detector_type == "MTCNN":
                return self._detect_mtcnn(image)
            elif self.detector_type == "OpenCV_Haar":
                return self._detect_opencv(image)
            else:
                logger.error("No detector available")
                return []
                
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def _detect_mtcnn(self, image):
        """Detect faces using MTCNN"""
        try:
            # Convert BGR to RGB if numpy array
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
            else:
                image_rgb = np.array(image)
            
            # Detect faces
            results = self.detector.detect_faces(image_rgb)
            return results
            
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
            return []
    
    def _detect_opencv(self, image):
        """Detect faces using OpenCV Haar Cascade"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(Config.MIN_FACE_SIZE, Config.MIN_FACE_SIZE)
            )
            
            # Convert to MTCNN-like format
            results = []
            for (x, y, w, h) in faces:
                result = {
                    'box': [x, y, w, h],
                    'confidence': 0.9,  # Default confidence for OpenCV
                    'keypoints': {}  # No keypoints for Haar cascade
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"OpenCV detection error: {e}")
            return []
    
    def extract_face(self, image, bbox, target_size=Config.IMAGE_SIZE, margin=10):
        """
        Extract and align face from image
        
        Args:
            image: Input image (numpy array)
            bbox: Bounding box dict from detector
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
                logger.debug("No faces detected")
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
            
            logger.debug(f"Extracted face with confidence: {confidence:.3f}")
            return face_img, confidence
            
        except Exception as e:
            logger.error(f"Error getting largest face: {e}")
            return None, None