import cv2
import numpy as np
import logging
from config import Config

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        """Initialize face detector with simple MTCNN"""
        self.detector = None
        self.detector_type = None
        
        # Try MTCNN with basic initialization only
        if self._init_mtcnn_basic():
            return
        
        # Fallback to OpenCV
        if self._init_opencv():
            return
        
        raise Exception("No face detector could be initialized")
    
    def _init_mtcnn_basic(self):
        """Initialize MTCNN with basic parameters only"""
        try:
            from mtcnn import MTCNN
            
            # Only use basic initialization without any parameters
            self.detector = MTCNN()
            self.detector_type = "MTCNN"
            logger.info("MTCNN initialized with basic parameters")
            return True
            
        except ImportError:
            logger.warning("MTCNN not available")
            return False
        except Exception as e:
            logger.warning(f"MTCNN initialization failed: {e}")
            return False
    
    def _init_opencv(self):
        """Initialize OpenCV Haar Cascade detector"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            
            if self.detector.empty():
                return False
            
            self.detector_type = "OpenCV_Haar"
            logger.info("OpenCV Haar Cascade detector initialized")
            return True
            
        except Exception as e:
            logger.warning(f"OpenCV initialization failed: {e}")
            return False
    
    def detect_faces(self, image):
        """Detect faces using available detector"""
        try:
            if self.detector_type == "MTCNN":
                return self._detect_mtcnn(image)
            elif self.detector_type == "OpenCV_Haar":
                return self._detect_opencv(image)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def _detect_mtcnn(self, image):
        """Detect faces using MTCNN"""
        try:
            # Convert BGR to RGB if needed
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
            logger.error(f"MTCNN detection error: {e}")
            return []
    
    def _detect_opencv(self, image):
        """Detect faces using OpenCV"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces with more conservative parameters
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),  # Smaller minimum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to MTCNN-like format
            results = []
            for (x, y, w, h) in faces:
                result = {
                    'box': [x, y, w, h],
                    'confidence': 0.85,  # Default confidence
                    'keypoints': {}
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"OpenCV detection error: {e}")
            return []
    
    def extract_face(self, image, bbox, target_size=Config.IMAGE_SIZE, margin=10):
        """Extract face from image"""
        try:
            x, y, width, height = bbox['box']
            
            # Ensure coordinates are valid
            img_h, img_w = image.shape[:2]
            
            # Add margin but keep within image bounds
            x = max(0, x - margin)
            y = max(0, y - margin)
            x2 = min(img_w, x + width + 2 * margin)
            y2 = min(img_h, y + height + 2 * margin)
            
            # Extract face
            face = image[y:y2, x:x2]
            
            # Check if face is valid
            if face.size == 0:
                logger.warning("Empty face region extracted")
                return None
            
            # Resize to target size
            face_resized = cv2.resize(face, target_size)
            
            return face_resized
            
        except Exception as e:
            logger.error(f"Error extracting face: {e}")
            return None
    
    def get_largest_face(self, image):
        """Get the largest face from image"""
        try:
            faces = self.detect_faces(image)
            
            if not faces:
                logger.debug("No faces detected")
                return None, None
            
            # Find largest face by area
            largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
            
            # Filter out very small faces
            box = largest_face['box']
            if box[2] < 30 or box[3] < 30:
                logger.debug("Face too small, skipping")
                return None, None
            
            # Extract face
            if isinstance(image, np.ndarray):
                face_img = self.extract_face(image, largest_face)
            else:
                # Convert PIL to numpy
                image_np = np.array(image)
                if len(image_np.shape) == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                face_img = self.extract_face(image_np, largest_face)
            
            if face_img is None:
                return None, None
            
            confidence = largest_face['confidence']
            logger.debug(f"Face extracted: {box[2]}x{box[3]}, confidence: {confidence:.3f}")
            
            return face_img, confidence
            
        except Exception as e:
            logger.error(f"Error getting largest face: {e}")
            return None, None
    
    def get_detector_info(self):
        """Get information about current detector"""
        return {
            'type': self.detector_type,
            'available': self.detector is not None
        }