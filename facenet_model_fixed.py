import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import numpy as np
import pickle
import os
import logging
from config import Config

# Set torch to use CPU if CUDA not available
if not torch.cuda.is_available():
    torch.set_num_threads(1)

logger = logging.getLogger(__name__)

class FaceNetModel:
    def __init__(self, pretrained=True, max_retries=3):
        """
        Initialize FaceNet model with retry logic
        
        Args:
            pretrained: Whether to use pretrained weights
            max_retries: Maximum number of download retries
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.embeddings_db = {}
        self.embeddings_file = os.path.join(Config.MODEL_DIR, 'face_embeddings.pkl')
        
        # Try to load model with retries
        for attempt in range(max_retries):
            try:
                self._load_model(pretrained, attempt)
                break
            except Exception as e:
                logger.warning(f"Model loading attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to load model after {max_retries} attempts")
                
                # Clear torch cache and try again
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        logger.info("FaceNet model initialized successfully")
    
    def _load_model(self, pretrained, attempt):
        """Load FaceNet model with different strategies"""
        
        strategies = [
            # Strategy 1: CASIA-WebFace pretrained (working from download)
            lambda: InceptionResnetV1(pretrained='casia-webface', classify=False, num_classes=None).eval(),
            
            # Strategy 2: No pretrained weights
            lambda: InceptionResnetV1(pretrained=None, classify=False, num_classes=None).eval(),
            
            # Strategy 3: VGGFace2 pretrained (might be corrupted)
            lambda: InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=None).eval()
        ]
        
        if not pretrained:
            # Skip pretrained strategies if not requested
            strategies = [strategies[1]]  # Use only no-pretrained
        
        strategy_names = ['CASIA-WebFace', 'No pretrained', 'VGGFace2']
        
        for i, strategy in enumerate(strategies):
            try:
                logger.info(f"Trying strategy {i+1}: {strategy_names[i]}")
                
                # Load model
                self.model = strategy().to(self.device)
                
                # Important: Set to eval mode to fix batch norm issue
                self.model.eval()
                
                # Test model with dummy input
                dummy_input = torch.randn(1, 3, 160, 160).to(self.device)
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                logger.info(f"Model loaded successfully with strategy: {strategy_names[i]}")
                return
                
            except Exception as e:
                logger.warning(f"Strategy {i+1} ({strategy_names[i]}) failed: {e}")
                continue
        
        raise Exception("All model loading strategies failed")
    
    def preprocess_face(self, face_image):
        """
        Preprocess face image for FaceNet
        
        Args:
            face_image: numpy array of face image
            
        Returns:
            torch.Tensor: Preprocessed face tensor
        """
        try:
            # Ensure image is the right size
            if face_image.shape[:2] != Config.IMAGE_SIZE:
                import cv2
                face_image = cv2.resize(face_image, Config.IMAGE_SIZE)
            
            # Normalize to [-1, 1] range (FaceNet requirement)
            face_normalized = (face_image.astype(np.float32) - 127.5) / 128.0
            
            # Convert to tensor and add batch dimension
            face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
            
            return face_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            return None
    
    def get_embedding(self, face_image):
        """
        Get embedding for a face image
        
        Args:
            face_image: numpy array of face image (160x160x3)
            
        Returns:
            numpy array: Face embedding
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            face_tensor = self.preprocess_face(face_image)
            if face_tensor is None:
                return None
            
            with torch.no_grad():
                embedding = self.model(face_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            # L2 normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def register_face(self, name, face_images):
        """
        Register a person's face embeddings
        
        Args:
            name: Person's name
            face_images: List of face images
            
        Returns:
            bool: Success status
        """
        try:
            embeddings = []
            
            for i, face_img in enumerate(face_images):
                embedding = self.get_embedding(face_img)
                if embedding is not None:
                    embeddings.append(embedding)
                    logger.debug(f"Generated embedding {i+1}/{len(face_images)} for {name}")
                else:
                    logger.warning(f"Failed to generate embedding {i+1}/{len(face_images)} for {name}")
            
            if not embeddings:
                logger.error(f"No valid embeddings found for {name}")
                return False
            
            # Average the embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Normalize again after averaging
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            
            # Store in database
            self.embeddings_db[name] = avg_embedding
            
            logger.info(f"Registered {name} with {len(embeddings)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error registering face for {name}: {e}")
            return False
    
    def verify_face(self, face_image, threshold=None):
        """
        Verify a face against registered faces
        
        Args:
            face_image: Face image to verify
            threshold: Distance threshold for verification
            
        Returns:
            dict: Verification result
        """
        if threshold is None:
            threshold = Config.VERIFICATION_THRESHOLD
        
        try:
            # Get embedding for input face
            embedding = self.get_embedding(face_image)
            if embedding is None:
                return {
                    'success': False,
                    'message': 'Failed to get embedding'
                }
            
            if not self.embeddings_db:
                return {
                    'success': False,
                    'message': 'No registered faces found'
                }
            
            # Compare with all registered faces
            best_match = None
            min_distance = float('inf')
            
            for name, stored_embedding in self.embeddings_db.items():
                # Calculate Euclidean distance
                distance = np.linalg.norm(embedding - stored_embedding)
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = name
            
            # Calculate similarity score (0-1, higher is better)
            similarity = 1.0 / (1.0 + min_distance)
            
            if min_distance <= threshold:
                return {
                    'success': True,
                    'identity': best_match,
                    'distance': float(min_distance),
                    'similarity': float(similarity),
                    'confidence': float(similarity * 100)
                }
            else:
                return {
                    'success': False,
                    'identity': best_match,
                    'distance': float(min_distance),
                    'similarity': float(similarity),
                    'confidence': float(similarity * 100),
                    'message': 'Face not recognized'
                }
                
        except Exception as e:
            logger.error(f"Error in face verification: {e}")
            return {
                'success': False,
                'message': f'Verification error: {str(e)}'
            }
    
    def save_embeddings(self):
        """Save embeddings database to file"""
        try:
            os.makedirs(os.path.dirname(self.embeddings_file), exist_ok=True)
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings_db, f)
            logger.info(f"Embeddings saved to {self.embeddings_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self):
        """Load embeddings database from file"""
        try:
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings_db = pickle.load(f)
                logger.info(f"Loaded embeddings for {len(self.embeddings_db)} identities")
                return True
            else:
                logger.info("No existing embeddings file found")
                return False
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False
    
    def get_registered_identities(self):
        """Get list of registered identities"""
        return list(self.embeddings_db.keys())
    
    def remove_identity(self, name):
        """Remove an identity from the database"""
        if name in self.embeddings_db:
            del self.embeddings_db[name]
            logger.info(f"Removed identity: {name}")
            return True
        return False
    
    def clear_cache(self):
        """Clear torch cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'registered_identities': len(self.embeddings_db)
        }