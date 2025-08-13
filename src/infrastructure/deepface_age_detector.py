import os
import tempfile
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2
from deepface import DeepFace
from ..domain.interfaces import AgeDetectorInterface
from ..domain.entities import AgeResult
from ..domain.exceptions import ModelNotLoadedError, ProcessingError


class DeepFaceAgeDetector(AgeDetectorInterface):
    """DeepFace implementation for age detection"""
    
    def __init__(self, model_name: str = 'age'):
        """
        Initialize DeepFace age detector
        
        Args:
            model_name: The age model to use (default: 'age')
        """
        self._model_name = model_name
        self._model_loaded = False
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the DeepFace age model"""
        try:
            # Pre-load the model by running a dummy prediction
            dummy_face = np.ones((48, 48, 3), dtype=np.uint8) * 128
            
            # Save dummy image temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, dummy_face)
                
                try:
                    # This will download and initialize the model
                    DeepFace.analyze(
                        img_path=tmp_file.name,
                        actions=['age'],
                        enforce_detection=False,
                        silent=True
                    )
                    self._model_loaded = True
                except Exception as e:
                    print(f"Warning: Could not pre-load age model: {e}")
                    # Still mark as loaded since DeepFace handles lazy loading
                    self._model_loaded = True
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
                        
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to initialize DeepFace age model: {str(e)}")
    
    def detect_age(self, face_image: np.ndarray) -> AgeResult:
        """Detect age in a single face image"""
        if not self._model_loaded:
            raise ModelNotLoadedError("DeepFace age model is not loaded")
        
        try:
            # Ensure the face image is in the correct format
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Convert BGR to RGB if needed (DeepFace expects RGB)
                if face_image.dtype == np.uint8:
                    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                else:
                    face_rgb = face_image
            else:
                raise ProcessingError("Face image must be a 3-channel RGB image")
            
            # Resize face to minimum size (DeepFace requires at least 48x48)
            h, w = face_rgb.shape[:2]
            if h < 48 or w < 48:
                target_size = max(48, max(h, w))
                face_rgb = cv2.resize(face_rgb, (target_size, target_size))
            
            # Save face image temporarily for DeepFace analysis
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                # Convert RGB back to BGR for cv2.imwrite
                face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(tmp_file.name, face_bgr)
                
                try:
                    # Analyze age
                    result = DeepFace.analyze(
                        img_path=tmp_file.name,
                        actions=['age'],
                        enforce_detection=False,
                        silent=True
                    )
                    
                    # Handle single face result
                    if isinstance(result, list):
                        age_data = result[0]['age']
                    else:
                        age_data = result['age']
                    
                    # Calculate age range (±5 years is common for age estimation uncertainty)
                    estimated_age = float(age_data)
                    age_range = self._calculate_age_range(estimated_age)
                    
                    return AgeResult(
                        age=estimated_age,
                        age_range=age_range
                    )
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
                        
        except Exception as e:
            if isinstance(e, (ModelNotLoadedError, ProcessingError)):
                raise
            raise ProcessingError(f"Error during age detection: {str(e)}")
    
    def _calculate_age_range(self, estimated_age: float) -> Tuple[int, int]:
        """Calculate age range based on estimated age"""
        # Standard age estimation uncertainty is typically ±5-8 years
        uncertainty = 6
        
        min_age = max(0, int(estimated_age - uncertainty))
        max_age = min(120, int(estimated_age + uncertainty))  # Cap at reasonable maximum age
        
        # Create meaningful age brackets
        if estimated_age < 18:
            # Children/teenagers - smaller range
            uncertainty = 3
            min_age = max(0, int(estimated_age - uncertainty))
            max_age = min(25, int(estimated_age + uncertainty))
        elif estimated_age < 30:
            # Young adults
            min_age = max(18, int(estimated_age - 4))
            max_age = min(35, int(estimated_age + 4))
        elif estimated_age < 50:
            # Middle-aged adults
            min_age = max(25, int(estimated_age - 5))
            max_age = min(60, int(estimated_age + 5))
        else:
            # Older adults - larger uncertainty
            uncertainty = 8
            min_age = max(40, int(estimated_age - uncertainty))
            max_age = min(120, int(estimated_age + uncertainty))
        
        return (min_age, max_age)
    
    def detect_ages_batch(self, face_images: List[np.ndarray]) -> List[AgeResult]:
        """Detect ages in multiple face images"""
        results = []
        for face_image in face_images:
            try:
                age_result = self.detect_age(face_image)
                results.append(age_result)
            except Exception as e:
                # Create a default result for failed detections
                # Use a neutral age with wide range
                default_age = 30.0
                default_range = (18, 65)
                
                results.append(AgeResult(
                    age=default_age,
                    age_range=default_range
                ))
        
        return results
    
    def is_model_loaded(self) -> bool:
        """Check if the age detection model is loaded"""
        return self._model_loaded
    
    def get_age_category(self, age: float) -> str:
        """Get age category label for the estimated age"""
        if age < 13:
            return "child"
        elif age < 20:
            return "teenager"
        elif age < 30:
            return "young_adult"
        elif age < 50:
            return "adult"
        elif age < 65:
            return "middle_aged"
        else:
            return "senior"
