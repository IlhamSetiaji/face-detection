import os
import uuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from typing import Optional, Tuple

from ...application.use_cases import FaceDetectionUseCase
from ...application.services import DetectionResultSerializer, ValidationService
from ...infrastructure.mtcnn_detector import MTCNNFaceDetector
from ...infrastructure.deepface_emotion_detector import DeepFaceEmotionDetector
from ...infrastructure.deepface_age_detector import DeepFaceAgeDetector
from ...infrastructure.deepface_combined_detector import DeepFaceCombinedDetector
from ...infrastructure.opencv_processor import OpenCVImageProcessor
from ...domain.exceptions import DetectionError, InvalidImageError, FileError


class FaceDetectionAPI:
    """Flask API for face detection"""
    
    def __init__(self, upload_folder: str = "uploads", results_folder: str = "results"):
        self.app = Flask(__name__)
        # Enable CORS for all routes and origins
        self.cors = CORS(self.app, resources={r"/*": {"origins": "*"}})
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        self.upload_folder = upload_folder
        self.results_folder = results_folder
        
        # Ensure directories exist
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)
        
        # Initialize dependencies
        self._setup_dependencies()
        
        # Register routes
        self._register_routes()
    
    def _setup_dependencies(self):
        """Setup dependency injection"""
        try:
            face_detector = MTCNNFaceDetector()
            image_processor = OpenCVImageProcessor()
            
            # Use combined detector for efficiency (handles both emotion and age)
            combined_detector = DeepFaceCombinedDetector()
            
            # Also keep individual detectors for flexibility
            emotion_detector = DeepFaceEmotionDetector()
            age_detector = DeepFaceAgeDetector()
            
            self.face_detection_use_case = FaceDetectionUseCase(
                face_detector, 
                image_processor,
                emotion_detector,
                age_detector,
                combined_detector
            )
        except Exception as e:
            print(f"Error initializing dependencies: {str(e)}")
            raise
    
    def _register_routes(self):
        """Register Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "service": "face-detection-api"
            })
        
        @self.app.route('/detect', methods=['POST'])
        def detect_faces():
            """Detect faces in uploaded image"""
            try:
                # Validate request
                if 'image' not in request.files:
                    return jsonify({"error": "No image file provided"}), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({"error": "No file selected"}), 400
                
                # Get optional parameters
                confidence_threshold = request.form.get('confidence', type=float)
                detect_emotions = request.form.get('emotions', 'false').lower() == 'true'
                detect_age = request.form.get('age', 'false').lower() == 'true'
                
                if confidence_threshold is not None:
                    if not ValidationService.validate_confidence_threshold(confidence_threshold):
                        return jsonify({"error": "Confidence threshold must be between 0.0 and 1.0"}), 400
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                if not ValidationService.validate_image_file(filename):
                    return jsonify({"error": "Unsupported image format"}), 400
                
                file_id = str(uuid.uuid4())
                file_path = os.path.join(self.upload_folder, f"{file_id}_{filename}")
                file.save(file_path)
                
                # Perform detection
                result = self.face_detection_use_case.detect_faces_in_image(
                    file_path, confidence_threshold, detect_emotions, detect_age
                )
                
                # Serialize result
                response_data = DetectionResultSerializer.to_dict(result)
                response_data["file_id"] = file_id
                
                return jsonify(response_data)
                
            except DetectionError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                return jsonify({"error": f"Internal server error: {str(e)}"}), 500
            finally:
                # Clean up uploaded file
                if 'file_path' in locals() and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
        
        @self.app.route('/detect-and-annotate', methods=['POST'])
        def detect_and_annotate():
            """Detect faces and return annotated image"""
            try:
                # Validate request
                if 'image' not in request.files:
                    return jsonify({"error": "No image file provided"}), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({"error": "No file selected"}), 400
                
                # Get optional parameters
                confidence_threshold = request.form.get('confidence', type=float)
                detect_emotions = request.form.get('emotions', 'false').lower() == 'true'
                detect_age = request.form.get('age', 'false').lower() == 'true'
                
                if confidence_threshold is not None:
                    if not ValidationService.validate_confidence_threshold(confidence_threshold):
                        return jsonify({"error": "Confidence threshold must be between 0.0 and 1.0"}), 400
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                if not ValidationService.validate_image_file(filename):
                    return jsonify({"error": "Unsupported image format"}), 400
                
                file_id = str(uuid.uuid4())
                input_path = os.path.join(self.upload_folder, f"{file_id}_{filename}")
                output_path = os.path.join(self.results_folder, f"{file_id}_annotated_{filename}")
                
                file.save(input_path)
                
                # Perform detection and annotation
                result = self.face_detection_use_case.detect_and_annotate(
                    input_path, output_path, confidence_threshold, True, detect_emotions, detect_age
                )
                
                # Return annotated image
                return send_file(
                    output_path,
                    as_attachment=True,
                    download_name=f"annotated_{filename}",
                    mimetype='image/jpeg'
                )
                
            except DetectionError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                return jsonify({"error": f"Internal server error: {str(e)}"}), 500
            finally:
                # Clean up files
                for path in [input_path, output_path]:
                    if 'path' in locals() and os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
        
        @self.app.errorhandler(RequestEntityTooLarge)
        def handle_file_too_large(e):
            return jsonify({"error": "File too large. Maximum size is 16MB"}), 413
        
        @self.app.errorhandler(404)
        def handle_not_found(e):
            return jsonify({"error": "Endpoint not found"}), 404
        
        @self.app.errorhandler(500)
        def handle_internal_error(e):
            return jsonify({"error": "Internal server error"}), 500
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Run the Flask application"""
        self.app.run(host=host, port=port, debug=debug)
