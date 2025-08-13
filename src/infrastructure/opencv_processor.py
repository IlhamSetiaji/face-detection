import cv2
import numpy as np
from typing import Tuple
from ..domain.interfaces import ImageProcessorInterface
from ..domain.entities import DetectionResult
from ..domain.exceptions import InvalidImageError, ProcessingError, FileError


class OpenCVImageProcessor(ImageProcessorInterface):
    """OpenCV implementation of image processing"""
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image using OpenCV"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise InvalidImageError(f"Could not load image: {image_path}")
            return image
        except Exception as e:
            if isinstance(e, InvalidImageError):
                raise
            raise FileError(f"Error loading image {image_path}: {str(e)}")
    
    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """Save image using OpenCV"""
        try:
            success = cv2.imwrite(output_path, image)
            if not success:
                raise ProcessingError(f"Failed to save image to {output_path}")
        except Exception as e:
            if isinstance(e, ProcessingError):
                raise
            raise FileError(f"Error saving image to {output_path}: {str(e)}")
    
    def draw_detections(self, image: np.ndarray, detection_result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes, landmarks, and emotions on image"""
        try:
            # Create a copy to avoid modifying the original
            annotated_image = image.copy()
            
            for face in detection_result.faces:
                # Draw bounding box
                x1, y1, x2, y2 = map(int, face.bbox)
                
                # Choose color based on confidence
                confidence = face.confidence
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                # Draw bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Prepare text labels
                labels = [f"Conf: {confidence:.2f}"]
                
                # Add emotion information if available
                if face.emotion:
                    emotion_text = f"{face.emotion.emotion.title()}: {face.emotion.confidence:.2f}"
                    labels.append(emotion_text)
                
                # Draw labels above the bounding box
                y_offset = y1
                for i, label in enumerate(labels):
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    label_y = y_offset - (len(labels) - i) * (text_size[1] + 10)
                    
                    # Draw background rectangle for text
                    cv2.rectangle(
                        annotated_image, 
                        (x1, label_y - text_size[1] - 5), 
                        (x1 + text_size[0] + 10, label_y + 5), 
                        color, 
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        annotated_image, 
                        label, 
                        (x1 + 5, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 255), 
                        2
                    )
                
                # Draw landmarks if available
                if face.landmarks:
                    for landmark in face.landmarks:
                        x, y = map(int, landmark)
                        cv2.circle(annotated_image, (x, y), 3, (255, 0, 0), -1)
            
            return annotated_image
            
        except Exception as e:
            raise ProcessingError(f"Error drawing detections: {str(e)}")
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target dimensions"""
        try:
            width, height = target_size
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            return resized
        except Exception as e:
            raise ProcessingError(f"Error resizing image: {str(e)}")
    
    def get_image_info(self, image: np.ndarray) -> dict:
        """Get image information"""
        try:
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1
            
            return {
                "width": width,
                "height": height,
                "channels": channels,
                "dtype": str(image.dtype)
            }
        except Exception as e:
            raise ProcessingError(f"Error getting image info: {str(e)}")
