from typing import Union, Optional
import time
import os
import cv2
import numpy as np
from ..domain.interfaces import FaceDetectorInterface, ImageProcessorInterface, EmotionDetectorInterface
from ..domain.entities import DetectionResult, FaceDetection
from ..domain.exceptions import InvalidImageError, FileError


class FaceDetectionUseCase:
    """Use case for face detection operations"""
    
    def __init__(
        self, 
        face_detector: FaceDetectorInterface,
        image_processor: ImageProcessorInterface,
        emotion_detector: Optional[EmotionDetectorInterface] = None
    ):
        self._face_detector = face_detector
        self._image_processor = image_processor
        self._emotion_detector = emotion_detector
    
    def detect_faces_in_image(
        self, 
        image_path: str, 
        confidence_threshold: Optional[float] = None,
        detect_emotions: bool = False
    ) -> DetectionResult:
        """
        Detect faces in an image file
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Minimum confidence for face detection
            detect_emotions: Whether to perform emotion detection on detected faces
            
        Returns:
            DetectionResult with detected faces and emotions (if requested)
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
            
            # Perform face detection
            result = self._face_detector.detect_faces(image)
            
            # Perform emotion detection if requested and available
            if detect_emotions and self._emotion_detector and result.faces:
                result = self._add_emotion_detection(result, image)
            
            return result
            
        except Exception as e:
            if isinstance(e, (InvalidImageError, FileError)):
                raise
            raise InvalidImageError(f"Error processing image: {str(e)}")
    
    def _add_emotion_detection(self, result: DetectionResult, image: np.ndarray) -> DetectionResult:
        """Add emotion detection to face detection results"""
        if not self._emotion_detector:
            return result
        
        updated_faces = []
        
        for face in result.faces:
            try:
                # Extract face region
                x1, y1, x2, y2 = map(int, face.bbox)
                face_image = image[y1:y2, x1:x2]
                
                # Skip if face region is too small
                if face_image.shape[0] < 10 or face_image.shape[1] < 10:
                    updated_faces.append(face)
                    continue
                
                # Detect emotion
                emotion_result = self._emotion_detector.detect_emotion(face_image)
                
                # Create updated face detection with emotion
                updated_face = FaceDetection(
                    bbox=face.bbox,
                    confidence=face.confidence,
                    landmarks=face.landmarks,
                    emotion=emotion_result
                )
                updated_faces.append(updated_face)
                
            except Exception as e:
                # If emotion detection fails for this face, keep original
                print(f"Warning: Emotion detection failed for face: {e}")
                updated_faces.append(face)
        
        # Return updated result
        return DetectionResult(
            image_path=result.image_path,
            faces=updated_faces,
            processing_time=result.processing_time,
            original_image_size=result.original_image_size
        )
    
    def detect_and_annotate(
        self,
        image_path: str,
        output_path: str,
        confidence_threshold: Optional[float] = None,
        draw_landmarks: bool = True,
        detect_emotions: bool = False
    ) -> DetectionResult:
        """
        Detect faces and save annotated image
        
        Args:
            image_path: Input image path
            output_path: Output path for annotated image
            confidence_threshold: Minimum confidence for detection
            draw_landmarks: Whether to draw facial landmarks
            detect_emotions: Whether to perform emotion detection
            
        Returns:
            DetectionResult with detected faces and emotions (if requested)
        """
        # Detect faces (and emotions if requested)
        result = self.detect_faces_in_image(image_path, confidence_threshold, detect_emotions)
        
        # Load original image for annotation
        image = self._image_processor.load_image(image_path)
        
        # Draw detections
        annotated_image = self._image_processor.draw_detections(image, result)
        
        # Save annotated image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._image_processor.save_image(annotated_image, output_path)
        
        return result
