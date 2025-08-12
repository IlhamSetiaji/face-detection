import time
from typing import Union, List, Tuple
import numpy as np
import cv2
from mtcnn import MTCNN
from ..domain.interfaces import FaceDetectorInterface
from ..domain.entities import DetectionResult, FaceDetection
from ..domain.exceptions import ModelNotLoadedError, ProcessingError, InvalidImageError


class MTCNNFaceDetector(FaceDetectorInterface):
    """MTCNN implementation of face detection"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        self._confidence_threshold = confidence_threshold
        self._model_loaded = False
        self._detector = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the MTCNN model"""
        try:
            self._detector = MTCNN()
            self._model_loaded = True
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to initialize MTCNN model: {str(e)}")
    
    def detect_faces(self, image: Union[str, np.ndarray]) -> DetectionResult:
        """Detect faces using MTCNN"""
        if not self._model_loaded:
            raise ModelNotLoadedError("MTCNN model is not loaded")
        
        start_time = time.time()
        
        try:
            # Handle string path
            if isinstance(image, str):
                image_path = image
                img_array = cv2.imread(image)
                if img_array is None:
                    raise InvalidImageError(f"Could not load image: {image}")
                # Convert BGR to RGB for MTCNN
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            else:
                image_path = "numpy_array"
                img_array = image.copy()
                # Ensure RGB format
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    # Assume BGR from OpenCV, convert to RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            image_size = (img_array.shape[1], img_array.shape[0])  # (width, height)
            
            # Detect faces
            detections = self._detector.detect_faces(img_array)
            
            # Convert detections to domain entities
            faces = []
            for detection in detections:
                if detection['confidence'] >= self._confidence_threshold:
                    face = self._convert_detection(detection)
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
    
    def _convert_detection(self, detection: dict) -> FaceDetection:
        """Convert MTCNN detection to domain entity"""
        # MTCNN returns: {'box': [x, y, w, h], 'confidence': float, 'keypoints': {...}}
        box = detection['box']
        
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        bbox = (
            float(box[0]),  # x1
            float(box[1]),  # y1
            float(box[0] + box[2]),  # x2
            float(box[1] + box[3])   # y2
        )
        
        confidence = float(detection['confidence'])
        
        # Extract landmarks if available
        landmarks = None
        if 'keypoints' in detection:
            keypoints = detection['keypoints']
            landmarks = [
                (float(keypoints['left_eye'][0]), float(keypoints['left_eye'][1])),
                (float(keypoints['right_eye'][0]), float(keypoints['right_eye'][1])),
                (float(keypoints['nose'][0]), float(keypoints['nose'][1])),
                (float(keypoints['mouth_left'][0]), float(keypoints['mouth_left'][1])),
                (float(keypoints['mouth_right'][0]), float(keypoints['mouth_right'][1]))
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
