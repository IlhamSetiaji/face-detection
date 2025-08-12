import time
from typing import Union, List, Tuple
import numpy as np
from retinaface import RetinaFace
from ..domain.interfaces import FaceDetectorInterface
from ..domain.entities import DetectionResult, FaceDetection
from ..domain.exceptions import ModelNotLoadedError, ProcessingError, InvalidImageError


class RetinaFaceDetector(FaceDetectorInterface):
    """RetinaFace implementation of face detection"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self._confidence_threshold = confidence_threshold
        self._model_loaded = False
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the RetinaFace model"""
        try:
            # RetinaFace loads the model automatically on first use
            # We'll verify it's working with a dummy detection
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            RetinaFace.detect_faces(dummy_image)
            self._model_loaded = True
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to initialize RetinaFace model: {str(e)}")
    
    def detect_faces(self, image: Union[str, np.ndarray]) -> DetectionResult:
        """Detect faces using RetinaFace"""
        if not self._model_loaded:
            raise ModelNotLoadedError("RetinaFace model is not loaded")
        
        start_time = time.time()
        
        try:
            # Handle string path
            if isinstance(image, str):
                image_path = image
            else:
                image_path = "numpy_array"
            
            # Detect faces
            detections = RetinaFace.detect_faces(image, threshold=self._confidence_threshold)
            
            # Get image dimensions
            if isinstance(image, str):
                import cv2
                img_array = cv2.imread(image)
                if img_array is None:
                    raise InvalidImageError(f"Could not load image: {image}")
                image_size = (img_array.shape[1], img_array.shape[0])  # (width, height)
            else:
                image_size = (image.shape[1], image.shape[0])
            
            # Convert detections to domain entities
            faces = []
            if isinstance(detections, dict):
                for face_key, face_data in detections.items():
                    face = self._convert_detection(face_data)
                    faces.append(face)
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                image_path=image_path,
                faces=faces,
                processing_time=processing_time,
                original_image_size=image_size
            )
            
        except Exception as e:
            if isinstance(e, (ModelNotLoadedError, InvalidImageError)):
                raise
            raise ProcessingError(f"Error during face detection: {str(e)}")
    
    def _convert_detection(self, face_data: dict) -> FaceDetection:
        """Convert RetinaFace detection to domain entity"""
        # RetinaFace returns: {'facial_area': [x, y, w, h], 'score': confidence, 'landmarks': {...}}
        facial_area = face_data['facial_area']
        
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        bbox = (
            float(facial_area[0]),  # x1
            float(facial_area[1]),  # y1
            float(facial_area[0] + facial_area[2]),  # x2
            float(facial_area[1] + facial_area[3])   # y2
        )
        
        confidence = float(face_data['score'])
        
        # Extract landmarks if available
        landmarks = None
        if 'landmarks' in face_data:
            landmarks_dict = face_data['landmarks']
            landmarks = [
                (float(landmarks_dict['left_eye'][0]), float(landmarks_dict['left_eye'][1])),
                (float(landmarks_dict['right_eye'][0]), float(landmarks_dict['right_eye'][1])),
                (float(landmarks_dict['nose'][0]), float(landmarks_dict['nose'][1])),
                (float(landmarks_dict['mouth_left'][0]), float(landmarks_dict['mouth_left'][1])),
                (float(landmarks_dict['mouth_right'][0]), float(landmarks_dict['mouth_right'][1]))
            ]
        
        return FaceDetection(
            bbox=bbox,
            confidence=confidence,
            landmarks=landmarks
        )
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for detection"""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        self._confidence_threshold = threshold
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model_loaded
