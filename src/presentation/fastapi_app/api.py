"""
FastAPI implementation demonstrating framework independence
"""

import os
import uuid
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
import uvicorn

from ...application.use_cases import FaceDetectionUseCase
from ...application.services import DetectionResultSerializer, ValidationService
from ...infrastructure.mtcnn_detector import MTCNNFaceDetector
from ...infrastructure.opencv_processor import OpenCVImageProcessor
from ...domain.exceptions import DetectionError


class FaceDetectionFastAPI:
    """FastAPI implementation of face detection service"""
    
    def __init__(self, upload_folder: str = "uploads", results_folder: str = "results"):
        self.app = FastAPI(
            title="Face Detection API",
            description="Face detection service using RetinaFace",
            version="1.0.0"
        )
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
        face_detector = MTCNNFaceDetector()
        image_processor = OpenCVImageProcessor()
        self.face_detection_use_case = FaceDetectionUseCase(face_detector, image_processor)
    
    def _register_routes(self):
        """Register FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "service": "face-detection-api-fastapi"}
        
        @self.app.post("/detect")
        async def detect_faces(
            image: UploadFile = File(...),
            confidence: Optional[float] = Form(None)
        ):
            """Detect faces in uploaded image"""
            try:
                # Validate file
                if not ValidationService.validate_image_file(image.filename):
                    raise HTTPException(
                        status_code=400, 
                        detail="Unsupported image format"
                    )
                
                # Validate confidence
                if confidence is not None:
                    if not ValidationService.validate_confidence_threshold(confidence):
                        raise HTTPException(
                            status_code=400,
                            detail="Confidence threshold must be between 0.0 and 1.0"
                        )
                
                # Save uploaded file
                file_id = str(uuid.uuid4())
                file_path = os.path.join(self.upload_folder, f"{file_id}_{image.filename}")
                
                content = await image.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                
                # Perform detection
                result = self.face_detection_use_case.detect_faces_in_image(
                    file_path, confidence
                )
                
                # Clean up
                os.remove(file_path)
                
                # Return result
                response_data = DetectionResultSerializer.to_dict(result)
                response_data["file_id"] = file_id
                
                return response_data
                
            except DetectionError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        
        @self.app.post("/detect-and-annotate")
        async def detect_and_annotate(
            image: UploadFile = File(...),
            confidence: Optional[float] = Form(None)
        ):
            """Detect faces and return annotated image"""
            try:
                # Validate file
                if not ValidationService.validate_image_file(image.filename):
                    raise HTTPException(
                        status_code=400,
                        detail="Unsupported image format"
                    )
                
                # Validate confidence
                if confidence is not None:
                    if not ValidationService.validate_confidence_threshold(confidence):
                        raise HTTPException(
                            status_code=400,
                            detail="Confidence threshold must be between 0.0 and 1.0"
                        )
                
                # Save uploaded file
                file_id = str(uuid.uuid4())
                input_path = os.path.join(self.upload_folder, f"{file_id}_{image.filename}")
                output_path = os.path.join(self.results_folder, f"{file_id}_annotated_{image.filename}")
                
                content = await image.read()
                with open(input_path, "wb") as f:
                    f.write(content)
                
                # Perform detection and annotation
                self.face_detection_use_case.detect_and_annotate(
                    input_path, output_path, confidence
                )
                
                # Clean up input file
                os.remove(input_path)
                
                # Return annotated image
                return FileResponse(
                    output_path,
                    media_type="image/jpeg",
                    filename=f"annotated_{image.filename}",
                    background=lambda: os.remove(output_path) if os.path.exists(output_path) else None
                )
                
            except DetectionError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI application"""
        uvicorn.run(self.app, host=host, port=port)


def create_app():
    """Application factory for FastAPI"""
    upload_folder = os.path.join(os.path.dirname(__file__), '../../../uploads')
    results_folder = os.path.join(os.path.dirname(__file__), '../../../results')
    
    api = FaceDetectionFastAPI(upload_folder, results_folder)
    return api.app


if __name__ == "__main__":
    upload_folder = os.path.join(os.path.dirname(__file__), '../../../uploads')
    results_folder = os.path.join(os.path.dirname(__file__), '../../../results')
    
    api = FaceDetectionFastAPI(upload_folder, results_folder)
    
    print("Starting Face Detection API (FastAPI)...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /detect - Detect faces in image (returns JSON)")
    print("  POST /detect-and-annotate - Detect faces and return annotated image")
    print("  GET  /docs - Interactive API documentation")
    print("\nServer starting on http://localhost:8000")
    
    api.run(host="0.0.0.0", port=8000)
