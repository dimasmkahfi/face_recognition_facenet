#!/usr/bin/env python3
"""
Flask API server for face verification
Usage: python api_server.py [--port PORT] [--ngrok]
"""

import os
import sys
import logging
import argparse
import tempfile
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from config import Config
from face_detector_simple import FaceDetector
from facenet_model_casia import FaceNetModel

def setup_logging():
    """Setup logging configuration"""
    Config.create_directories()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Config.LOGS_DIR, 'api_server.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_app():
    """Create and configure Flask app"""
    app = Flask(__name__)
    CORS(app)
    
    # Initialize components
    face_detector = FaceDetector()
    facenet_model = FaceNetModel()
    
    # Load existing embeddings
    facenet_model.load_embeddings()
    
    logger = logging.getLogger(__name__)
    
    @app.route('/', methods=['GET'])
    def index():
        """API status endpoint"""
        identities = facenet_model.get_registered_identities()
        return jsonify({
            'status': 'running',
            'service': 'Face Verification API with FaceNet',
            'version': '1.0.0',
            'registered_identities': len(identities),
            'identities': identities
        })
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time()
        })
    
    @app.route('/verify-face', methods=['POST'])
    def verify_face():
        """Face verification endpoint"""
        start_time = time.time()
        
        # Check if image file is provided
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image file provided',
                'code': 'NO_IMAGE'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No image file selected',
                'code': 'NO_IMAGE'
            }), 400
        
        try:
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=os.path.splitext(file.filename)[1]
            )
            file.save(temp_file.name)
            temp_file.close()
            
            # Read image
            image = cv2.imread(temp_file.name)
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            if image is None:
                return jsonify({
                    'success': False,
                    'message': 'Failed to read image file',
                    'code': 'INVALID_IMAGE'
                }), 400
            
            # Detect face
            face_img, detection_confidence = face_detector.get_largest_face(image)
            
            if face_img is None:
                processing_time = (time.time() - start_time) * 1000
                return jsonify({
                    'success': False,
                    'message': 'No face detected in the image',
                    'code': 'NO_FACE_DETECTED',
                    'data': {
                        'face_detected': False,
                        'processing_time_ms': round(processing_time, 2)
                    }
                }), 200
            
            # Verify face
            verification_result = facenet_model.verify_face(face_img)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Prepare response
            response_data = {
                'face_detected': True,
                'detection_confidence': round(detection_confidence, 3),
                'processing_time_ms': round(processing_time, 2),
                'distance': verification_result.get('distance', 0),
                'similarity': verification_result.get('similarity', 0),
                'confidence': verification_result.get('confidence', 0)
            }
            
            if verification_result['success']:
                return jsonify({
                    'success': True,
                    'message': 'Face verification successful',
                    'user_id': verification_result['identity'],
                    'name': verification_result['identity'],
                    'similarity': round(verification_result['similarity'], 3),
                    'data': response_data
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': verification_result.get('message', 'Face not recognized'),
                    'code': 'FACE_NOT_RECOGNIZED',
                    'best_match': verification_result.get('identity'),
                    'data': response_data
                }), 200
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Error during face verification: {e}")
            return jsonify({
                'success': False,
                'message': f'Internal server error: {str(e)}',
                'code': 'PROCESSING_ERROR',
                'data': {
                    'processing_time_ms': round(processing_time, 2)
                }
            }), 500
    
    @app.route('/identities', methods=['GET'])
    def list_identities():
        """List all registered identities"""
        try:
            identities = facenet_model.get_registered_identities()
            return jsonify({
                'success': True,
                'count': len(identities),
                'identities': identities
            }), 200
        except Exception as e:
            logger.error(f"Error listing identities: {e}")
            return jsonify({
                'success': False,
                'message': f'Error listing identities: {str(e)}'
            }), 500
    
    @app.route('/register-identity', methods=['POST'])
    def register_identity():
        """Register a new identity with uploaded images"""
        try:
            # Get identity name
            if 'name' not in request.form:
                return jsonify({
                    'success': False,
                    'message': 'Identity name is required'
                }), 400
            
            name = request.form['name'].strip()
            if not name:
                return jsonify({
                    'success': False,
                    'message': 'Identity name cannot be empty'
                }), 400
            
            # Check if images are provided
            if 'images' not in request.files:
                return jsonify({
                    'success': False,
                    'message': 'No image files provided'
                }), 400
            
            files = request.files.getlist('images')
            if not files or all(f.filename == '' for f in files):
                return jsonify({
                    'success': False,
                    'message': 'No valid image files provided'
                }), 400
            
            face_images = []
            processed_count = 0
            failed_count = 0
            
            # Process each uploaded image
            for file in files:
                if file.filename == '':
                    continue
                
                try:
                    # Save to temporary file
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(file.filename)[1]
                    )
                    file.save(temp_file.name)
                    temp_file.close()
                    
                    # Read and process image
                    image = cv2.imread(temp_file.name)
                    os.unlink(temp_file.name)
                    
                    if image is not None:
                        face_img, confidence = face_detector.get_largest_face(image)
                        if face_img is not None:
                            face_images.append(face_img)
                            processed_count += 1
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing uploaded image: {e}")
                    failed_count += 1
            
            if not face_images:
                return jsonify({
                    'success': False,
                    'message': 'No valid faces found in uploaded images'
                }), 400
            
            # Register the identity
            success = facenet_model.register_face(name, face_images)
            
            if success:
                # Save embeddings
                facenet_model.save_embeddings()
                
                return jsonify({
                    'success': True,
                    'message': f'Identity {name} registered successfully',
                    'processed_images': processed_count,
                    'failed_images': failed_count,
                    'total_images': len(files)
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': 'Failed to register identity'
                }), 500
                
        except Exception as e:
            logger.error(f"Error registering identity: {e}")
            return jsonify({
                'success': False,
                'message': f'Error registering identity: {str(e)}'
            }), 500
    
    @app.route('/add-faces', methods=['POST'])
    def add_faces_to_identity():
        """Add more faces to existing identity"""
        try:
            # Get identity name
            if 'name' not in request.form:
                return jsonify({
                    'success': False,
                    'message': 'Identity name is required'
                }), 400
            
            name = request.form['name'].strip()
            if not name:
                return jsonify({
                    'success': False,
                    'message': 'Identity name cannot be empty'
                }), 400
            
            # Check if identity exists
            existing_identities = facenet_model.get_registered_identities()
            if name not in existing_identities:
                return jsonify({
                    'success': False,
                    'message': f'Identity {name} not found. Use /register-identity for new users.'
                }), 404
            
            # Check if images are provided
            if 'images' not in request.files:
                return jsonify({
                    'success': False,
                    'message': 'No image files provided'
                }), 400
            
            files = request.files.getlist('images')
            if not files or all(f.filename == '' for f in files):
                return jsonify({
                    'success': False,
                    'message': 'No valid image files provided'
                }), 400
            
            # Get existing embedding
            existing_embedding = facenet_model.embeddings_db[name]
            
            # Process new images
            new_face_images = []
            processed_count = 0
            failed_count = 0
            
            for file in files:
                if file.filename == '':
                    continue
                
                try:
                    # Save to temporary file
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(file.filename)[1]
                    )
                    file.save(temp_file.name)
                    temp_file.close()
                    
                    # Read and process image
                    image = cv2.imread(temp_file.name)
                    os.unlink(temp_file.name)
                    
                    if image is not None:
                        face_img, confidence = face_detector.get_largest_face(image)
                        if face_img is not None:
                            new_face_images.append(face_img)
                            processed_count += 1
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing uploaded image: {e}")
                    failed_count += 1
            
            if not new_face_images:
                return jsonify({
                    'success': False,
                    'message': 'No valid faces found in uploaded images'
                }), 400
            
            # Generate embeddings for new faces
            new_embeddings = []
            for face_img in new_face_images:
                embedding = facenet_model.get_embedding(face_img)
                if embedding is not None:
                    new_embeddings.append(embedding)
            
            if not new_embeddings:
                return jsonify({
                    'success': False,
                    'message': 'Failed to generate embeddings from new faces'
                }), 500
            
            # Combine with existing embedding (weighted average)
            # Give more weight to existing embedding (it represents multiple faces already)
            existing_weight = 0.7
            new_weight = 0.3
            
            avg_new_embedding = np.mean(new_embeddings, axis=0)
            combined_embedding = (existing_weight * existing_embedding + 
                                 new_weight * avg_new_embedding)
            
            # Normalize combined embedding
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            
            # Update database
            facenet_model.embeddings_db[name] = combined_embedding
            
            # Save embeddings
            facenet_model.save_embeddings()
            
            return jsonify({
                'success': True,
                'message': f'Added {len(new_embeddings)} new faces to {name}',
                'processed_images': processed_count,
                'failed_images': failed_count,
                'total_images': len(files),
                'new_faces_added': len(new_embeddings)
            }), 200
            
        except Exception as e:
            logger.error(f"Error adding faces to identity: {e}")
            return jsonify({
                'success': False,
                'message': f'Error adding faces: {str(e)}'
            }), 500
    def remove_identity():
        """Remove a registered identity"""
        try:
            data = request.get_json()
            if not data or 'name' not in data:
                return jsonify({
                    'success': False,
                    'message': 'Identity name is required'
                }), 400
            
            name = data['name'].strip()
            if not name:
                return jsonify({
                    'success': False,
                    'message': 'Identity name cannot be empty'
                }), 400
            
            success = facenet_model.remove_identity(name)
            
            if success:
                # Save updated embeddings
                facenet_model.save_embeddings()
                
                return jsonify({
                    'success': True,
                    'message': f'Identity {name} removed successfully'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': f'Identity {name} not found'
                }), 404
                
        except Exception as e:
            logger.error(f"Error removing identity: {e}")
            return jsonify({
                'success': False,
                'message': f'Error removing identity: {str(e)}'
            }), 500
    
    return app

def main():
    """Main function to run the API server"""
    parser = argparse.ArgumentParser(description='Face Verification API Server')
    parser.add_argument('--port', type=int, default=Config.API_PORT,
                       help='Port to run the server on')
    parser.add_argument('--host', default=Config.API_HOST,
                       help='Host to run the server on')
    parser.add_argument('--ngrok', action='store_true',
                       help='Start ngrok tunnel for external access')
    parser.add_argument('--ngrok-token', default=Config.NGROK_AUTH_TOKEN,
                       help='Ngrok auth token')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create directories
    Config.create_directories()
    
    # Check if embeddings exist
    embeddings_file = os.path.join(Config.MODEL_DIR, 'face_embeddings.pkl')
    if not os.path.exists(embeddings_file):
        logger.warning("No face embeddings found. Please run training first:")
        logger.warning("python train.py")
        print("\nWarning: No trained model found!")
        print("Please run the training script first:")
        print("  python train.py")
        print("\nOr you can register identities using the /register-identity endpoint")
    
    # Create Flask app
    app = create_app()
    
    # Start ngrok if requested
    public_url = None
    if args.ngrok:
        try:
            from pyngrok import ngrok
            
            if args.ngrok_token:
                ngrok.set_auth_token(args.ngrok_token)
            
            # Start ngrok tunnel
            public_url = ngrok.connect(args.port)
            logger.info(f"Ngrok tunnel started: {public_url}")
            print(f"\n🌐 Public URL: {public_url}")
        except ImportError:
            logger.error("pyngrok not installed. Install it with: pip install pyngrok")
        except Exception as e:
            logger.error(f"Failed to start ngrok: {e}")
    
    # Print API information
    print("\n" + "=" * 60)
    print("🚀 FACE VERIFICATION API SERVER")
    print("=" * 60)
    print(f"📍 Local URL: http://{args.host}:{args.port}")
    if public_url:
        print(f"🌐 Public URL: {public_url}")
    print("\n📋 Available Endpoints:")
    print("  GET  /                    - API status and info")
    print("  GET  /health              - Health check")
    print("  POST /verify-face         - Verify face (multipart/form-data, field: image)")
    print("  GET  /identities          - List registered identities")
    print("  POST /register-identity   - Register new identity (form: name, files: images)")
    print("  DELETE /remove-identity   - Remove identity (JSON: {\"name\": \"identity_name\"})")
    print("=" * 60)
    
    try:
        # Run the server
        logger.info(f"Starting server on {args.host}:{args.port}")
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        if args.ngrok:
            try:
                ngrok.kill()
                logger.info("Ngrok tunnel closed")
            except:
                pass
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    main()