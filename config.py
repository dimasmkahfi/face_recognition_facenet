import os

# Configuration settings
class Config:
    # Directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    TEMP_DIR = os.path.join(BASE_DIR, 'temp')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Model settings
    IMAGE_SIZE = (160, 160)  # FaceNet standard input size
    EMBEDDING_SIZE = 512
    
    # Face detection settings
    MIN_FACE_SIZE = 40
    # Note: thresholds parameter might not be available in all MTCNN versions
    DETECTION_THRESHOLD = [0.6, 0.7, 0.7]  # P-Net, R-Net, O-Net thresholds
    
    # Verification settings
    VERIFICATION_THRESHOLD = 0.6  # Distance threshold for verification
    
    # API settings
    API_HOST = '0.0.0.0'
    API_PORT = 5000
    
    # Ngrok settings (optional)
    NGROK_AUTH_TOKEN = None  # Set your ngrok auth token if you have one
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for dir_path in [cls.DATASET_DIR, cls.MODEL_DIR, cls.TEMP_DIR, cls.LOGS_DIR]:
            os.makedirs(dir_path, exist_ok=True)