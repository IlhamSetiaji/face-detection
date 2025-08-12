from typing import Dict, Any, Optional
from ..domain.entities import DetectionResult, FaceDetection


class DetectionResultSerializer:
    """Service for serializing detection results"""
    
    @staticmethod
    def to_dict(result: DetectionResult) -> Dict[str, Any]:
        """Convert DetectionResult to dictionary"""
        return {
            "image_path": result.image_path,
            "face_count": result.get_face_count(),
            "has_faces": result.has_faces(),
            "processing_time": result.processing_time,
            "original_image_size": {
                "width": result.original_image_size[0],
                "height": result.original_image_size[1]
            },
            "faces": [
                DetectionResultSerializer._face_to_dict(face) 
                for face in result.faces
            ]
        }
    
    @staticmethod
    def _face_to_dict(face: FaceDetection) -> Dict[str, Any]:
        """Convert FaceDetection to dictionary"""
        face_dict = {
            "bbox": {
                "x1": float(face.bbox[0]),
                "y1": float(face.bbox[1]),
                "x2": float(face.bbox[2]),
                "y2": float(face.bbox[3])
            },
            "confidence": float(face.confidence),
            "width": face.get_width(),
            "height": face.get_height(),
            "area": face.get_area()
        }
        
        if face.landmarks:
            face_dict["landmarks"] = [
                {"x": float(point[0]), "y": float(point[1])} 
                for point in face.landmarks
            ]
        
        return face_dict


class ValidationService:
    """Service for input validation"""
    
    @staticmethod
    def validate_image_file(file_path: str) -> bool:
        """Validate if file is a supported image format"""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return any(file_path.lower().endswith(ext) for ext in supported_extensions)
    
    @staticmethod
    def validate_confidence_threshold(threshold: float) -> bool:
        """Validate confidence threshold is in valid range"""
        return 0.0 <= threshold <= 1.0
