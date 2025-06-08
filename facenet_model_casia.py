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
    def __init__(self, force_casia=True):
        """
        Initialize FaceNet model with CASIA-WebFace (working model)
        
        Args:
            force_casia: Force use CASIA-WebFace model only
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.embeddings_db = {}
        self.embeddings_file = os.path.join(Config.MODEL_DIR, 'face_embeddings.pkl')
        
        # Load model with CASIA-WebFace only (working model)
        self._load_casia_model()
        
        logger.info("FaceNet model initialized successfully with CASIA-WebFace")
    
    def _load_casia_model(self):
        """Load CASIA-WebFace model (the working one)"""
        try:
            logger.info("Loading CASIA-WebFace model...")
            
            # Load model in eval mode to fix batch norm issues
            self.model = InceptionResnetV1(
                pretrained='casia-webface',
                classify=False,
                num_classes=None
            ).eval()
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Test with dummy input to ensure it works
            dummy_input = torch.randn(1, 3, 160, 160).to(self.device)
            with torch.no_grad():
                test_output = self.model(dummy_input)
            
            logger.info(f"CASIA-WebFace model loaded successfully")
            logger.info(f"Output shape: {test_output.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load CASIA-WebFace model: {e}")
            # Fallback to no pretrained weights
            try:
                logger.info("Falling back to model without pretrained weights...")
                self.model = InceptionResnetV1(
                    pretrained=None,
                    classify=False,
                    num_classes=None
                ).eval().to(self.device)
                
                # Test fallback model
                dummy_input = torch.randn(1, 3, 160, 160).to(self.device)
                with torch.no_grad():
                    test_output = self.model(dummy_input)
                
                logger.info("Fallback model loaded successfully (no pretrained weights)")
                
            except Exception as e2:
                logger.error(f"All model loading strategies failed: {e2}")
                raise Exception("Could not load any FaceNet model")
    
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
            
            # Ensure model is in eval mode
            self.model.eval()
            
            with torch.no_grad():
                embedding = self.model(face_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            # L2 normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                logger.warning("Zero norm embedding detected")
                return None
            
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
            successful_embeddings = 0
            
            for i, face_img in enumerate(face_images):
                try:
                    embedding = self.get_embedding(face_img)
                    if embedding is not None:
                        embeddings.append(embedding)
                        successful_embeddings += 1
                        logger.debug(f"Generated embedding {successful_embeddings}/{len(face_images)} for {name}")
                        avg_embedding = np.mean(embeddings, axis=0)
                        self.embeddings_db[name] = avg_embedding

                    else:
                        logger.warning(f"Failed to generate embedding {i+1}/{len(face_images)} for {name}")
                except Exception as e:
                    logger.error(f"Error processing image {i+1} for {name}: {e}")
            
            if not embeddings:
                logger.error(f"No valid embeddings found for {name}")
                return False
            
            if successful_embeddings < len(face_images) * 0.5:  # Less than 50% success
                logger.warning(f"Low success rate for {name}: {successful_embeddings}/{len(face_images)}")
            
            # Average the embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Normalize again after averaging
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            else:
                logger.error(f"Zero norm average embedding for {name}")
                return False
            
            # Store in database
            self.embeddings_db[name] = avg_embedding
            
            logger.info(f"Registered {name} with {len(embeddings)} valid embeddings")
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
                    'message': 'Failed to get embedding from face image'
                }
            
            if not self.embeddings_db:
                return {
                    'success': False,
                    'message': 'No registered faces found in database'
                }
            
            # Compare with all registered faces
            best_match = None
            min_distance = float('inf')
            all_results = {}
            
            for name, stored_embedding in self.embeddings_db.items():
                try:
                    # Calculate Euclidean distance
                    distance = np.linalg.norm(embedding - stored_embedding)
                    all_results[name] = distance
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = name
                        
                except Exception as e:
                    logger.error(f"Error comparing with {name}: {e}")
                    continue
            
            if best_match is None:
                return {
                    'success': False,
                    'message': 'Error during face comparison'
                }
            
            # Calculate similarity score (0-1, higher is better)
            similarity = 1.0 / (1.0 + min_distance)
            
            # Log all distances for debugging
            logger.debug(f"Face verification distances: {all_results}")
            logger.debug(f"Best match: {best_match} (distance: {min_distance:.4f}, similarity: {similarity:.4f})")
            
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
                    'message': f'Face not recognized (distance {min_distance:.4f} > threshold {threshold})'
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
            'model_type': 'CASIA-WebFace',
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'registered_identities': len(self.embeddings_db)
        }