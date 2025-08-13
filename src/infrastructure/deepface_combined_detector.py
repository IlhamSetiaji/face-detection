import os
import tempfile
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from deepface import DeepFace
from ..domain.interfaces import EmotionDetectorInterface, AgeDetectorInterface
from ..domain.entities import EmotionResult, AgeResult
from ..domain.exceptions import ModelNotLoadedError, ProcessingError


class DeepFaceCombinedDetector:
    """
    Combined DeepFace detector for both emotion and age detection
    This is more efficient than running separate detectors as it analyzes the face once
    """
    
    def __init__(self):
        """Initialize the combined DeepFace detector"""
        self._model_loaded = False
        self._emotion_labels = [
            'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
        ]
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the DeepFace models"""
        try:
            # Pre-load the models by running a dummy prediction
            dummy_face = np.ones((48, 48, 3), dtype=np.uint8) * 128
            
            # Save dummy image temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, dummy_face)
                
                try:
                    # This will download and initialize both emotion and age models
                    DeepFace.analyze(
                        img_path=tmp_file.name,
                        actions=['emotion', 'age'],
                        enforce_detection=False,
                        silent=True
                    )
                    self._model_loaded = True
                except Exception as e:
                    print(f"Warning: Could not pre-load combined models: {e}")
                    # Still mark as loaded since DeepFace handles lazy loading
                    self._model_loaded = True
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
                        
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to initialize DeepFace combined models: {str(e)}")
    
    def detect_emotion_and_age(
        self, 
        face_image: np.ndarray
    ) -> Tuple[Optional[EmotionResult], Optional[AgeResult]]:
        """
        Detect both emotion and age in a single face image
        
        Args:
            face_image: Numpy array of the cropped face image
            
        Returns:
            Tuple of (EmotionResult, AgeResult) - either can be None if detection fails
        """
        if not self._model_loaded:
            raise ModelNotLoadedError("DeepFace combined models are not loaded")
        
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
                    # Analyze both emotion and age in a single call
                    result = DeepFace.analyze(
                        img_path=tmp_file.name,
                        actions=['emotion', 'age'],
                        enforce_detection=False,
                        silent=True
                    )
                    
                    # Handle single face result
                    if isinstance(result, list):
                        analysis_data = result[0]
                    else:
                        analysis_data = result
                    
                    # Extract emotion data
                    emotion_result = None
                    if 'emotion' in analysis_data:
                        emotion_data = analysis_data['emotion']
                        dominant_emotion = max(emotion_data, key=emotion_data.get)
                        dominant_confidence = emotion_data[dominant_emotion] / 100.0
                        
                        normalized_emotions = {
                            emotion: score / 100.0 for emotion, score in emotion_data.items()
                        }
                        
                        emotion_result = EmotionResult(
                            emotion=dominant_emotion,
                            confidence=dominant_confidence,
                            emotions=normalized_emotions
                        )
                    
                    # Extract age data
                    age_result = None
                    if 'age' in analysis_data:
                        estimated_age = float(analysis_data['age'])
                        age_range = self._calculate_age_range(estimated_age)
                        
                        age_result = AgeResult(
                            age=estimated_age,
                            age_range=age_range
                        )
                    
                    return emotion_result, age_result
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
                        
        except Exception as e:
            if isinstance(e, (ModelNotLoadedError, ProcessingError)):
                raise
            raise ProcessingError(f"Error during combined detection: {str(e)}")
    
    def _calculate_age_range(self, estimated_age: float) -> Tuple[int, int]:
        """Calculate age range based on estimated age"""
        # Standard age estimation uncertainty is typically ±5-8 years
        uncertainty = 6
        
        min_age = max(0, int(estimated_age - uncertainty))
        max_age = min(120, int(estimated_age + uncertainty))
        
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
    
    def detect_emotion_only(self, face_image: np.ndarray) -> Optional[EmotionResult]:
        """Detect only emotion (for backward compatibility)"""
        emotion_result, _ = self.detect_emotion_and_age(face_image)
        return emotion_result
    
    def detect_age_only(self, face_image: np.ndarray) -> Optional[AgeResult]:
        """Detect only age"""
        _, age_result = self.detect_emotion_and_age(face_image)
        return age_result
    
    def is_model_loaded(self) -> bool:
        """Check if the models are loaded"""
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
