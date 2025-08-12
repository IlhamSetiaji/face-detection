from typing import Union, Optional
import time
import os
from ..domain.interfaces import FaceDetectorInterface, ImageProcessorInterface
from ..domain.entities import DetectionResult
from ..domain.exceptions import InvalidImageError, FileError


class FaceDetectionUseCase:
    """Use case for face detection operations"""
    
    def __init__(
        self, 
        face_detector: FaceDetectorInterface,
        image_processor: ImageProcessorInterface
    ):
        self._face_detector = face_detector
        self._image_processor = image_processor
    
    def detect_faces_in_image(
        self, 
        image_path: str, 
        confidence_threshold: Optional[float] = None
    ) -> DetectionResult:
        """
        Detect faces in an image file
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence for face detection
            
        Returns:
            DetectionResult with detected faces
        """
        if not os.path.exists(image_path):
            raise FileError(f"Image file not found: {image_path}")
        
        if confidence_threshold is not None:
            self._face_detector.set_confidence_threshold(confidence_threshold)
        
        try:
            # Load and validate image
            image = self._image_processor.load_image(image_path)
            if image is None or image.size == 0:
                raise InvalidImageError(f"Unable to load image: {image_path}")
            
            # Perform detection
            result = self._face_detector.detect_faces(image)
            return result
            
        except Exception as e:
            if isinstance(e, (InvalidImageError, FileError)):
                raise
            raise InvalidImageError(f"Error processing image: {str(e)}")
    
    def detect_and_annotate(
        self,
        image_path: str,
        output_path: str,
        confidence_threshold: Optional[float] = None,
        draw_landmarks: bool = True
    ) -> DetectionResult:
        """
        Detect faces and save annotated image
        
        Args:
            image_path: Input image path
            output_path: Output path for annotated image
            confidence_threshold: Minimum confidence for detection
            draw_landmarks: Whether to draw facial landmarks
            
        Returns:
            DetectionResult with detected faces
        """
        # Detect faces
        result = self.detect_faces_in_image(image_path, confidence_threshold)
        
        # Load original image for annotation
        image = self._image_processor.load_image(image_path)
        
        # Draw detections
        annotated_image = self._image_processor.draw_detections(image, result)
        
        # Save annotated image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._image_processor.save_image(annotated_image, output_path)
        
        return result
