from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np
from .entities import DetectionResult


class FaceDetectorInterface(ABC):
    """Abstract interface for face detection implementations"""
    
    @abstractmethod
    def detect_faces(self, image: Union[str, np.ndarray]) -> DetectionResult:
        """
        Detect faces in an image
        
        Args:
            image: Either a file path (str) or numpy array representing the image
            
        Returns:
            DetectionResult containing detected faces and metadata
        """
        pass
    
    @abstractmethod
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the minimum confidence threshold for face detection"""
        pass
    
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Check if the detection model is properly loaded"""
        pass


class ImageProcessorInterface(ABC):
    """Abstract interface for image processing operations"""
    
    @abstractmethod
    def load_image(self, image_path: str) -> np.ndarray:
        """Load an image from file path"""
        pass
    
    @abstractmethod
    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """Save an image to file"""
        pass
    
    @abstractmethod
    def draw_detections(self, image: np.ndarray, detection_result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes and landmarks on the image"""
        pass
    
    @abstractmethod
    def resize_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize image to target dimensions"""
        pass
