import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
import numpy as np
import pickle
import os
import logging
from config import Config

logger = logging.getLogger(__name__)

class FaceNetModel:
    def __init__(self, pretrained=True):
        """
        Initialize FaceNet model
        
        Args:
            pretrained: Whether to use pretrained weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load FaceNet model
        self.model = InceptionResnetV1(
            pretrained='vggface2' if pretrained else None,
            classify=False,
            num_classes=None
        ).eval().to(self.device)
        
        self.embeddings_db = {}
        self.embeddings_file = os.path.join(Config.MODEL_DIR, 'face_embeddings.pkl')
        
        logger.info("FaceNet model initialized successfully")
    
    def preprocess_face(self, face_image):
        """
        Preprocess face image for FaceNet
        
        Args:
            face_image: numpy array of face image
            
        Returns:
            torch.Tensor: Preprocessed face tensor
        """
        try:
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
            face_tensor = self.preprocess_face(face_image)
            if face_tensor is None:
                return None
            
            with torch.no_grad():
                embedding = self.model(face_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            # L2 normalize the embedding
            embedding = embedding / np.linalg.norm(embedding)
            
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
            
            for face_img in face_images:
                embedding = self.get_embedding(face_img)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if not embeddings:
                logger.error(f"No valid embeddings found for {name}")
                return False
            
            # Average the embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            
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