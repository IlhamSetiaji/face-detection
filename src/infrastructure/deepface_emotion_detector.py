import os
import tempfile
import time
from typing import List, Dict, Any
import numpy as np
import cv2
from deepface import DeepFace
from ..domain.interfaces import EmotionDetectorInterface
from ..domain.entities import EmotionResult
from ..domain.exceptions import ModelNotLoadedError, ProcessingError


class DeepFaceEmotionDetector(EmotionDetectorInterface):
    """DeepFace implementation for emotion detection"""
    
    def __init__(self, model_name: str = 'emotion'):
        """
        Initialize DeepFace emotion detector
        
        Args:
            model_name: The emotion model to use (default: 'emotion')
        """
        self._model_name = model_name
        self._model_loaded = False
        self._emotion_labels = [
            'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
        ]
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the DeepFace emotion model"""
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
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    self._model_loaded = True
                except Exception as e:
                    print(f"Warning: Could not pre-load emotion model: {e}")
                    # Still mark as loaded since DeepFace handles lazy loading
                    self._model_loaded = True
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass
                        
        except Exception as e:
            raise ModelNotLoadedError(f"Failed to initialize DeepFace emotion model: {str(e)}")
    
    def detect_emotion(self, face_image: np.ndarray) -> EmotionResult:
        """Detect emotion in a single face image"""
        if not self._model_loaded:
            raise ModelNotLoadedError("DeepFace emotion model is not loaded")
        
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
                    # Analyze emotion
                    result = DeepFace.analyze(
                        img_path=tmp_file.name,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    
                    # Handle single face result
                    if isinstance(result, list):
                        emotion_data = result[0]['emotion']
                    else:
                        emotion_data = result['emotion']
                    
                    # Find dominant emotion
                    dominant_emotion = max(emotion_data, key=emotion_data.get)
                    dominant_confidence = emotion_data[dominant_emotion] / 100.0  # Convert percentage to decimal
                    
                    # Normalize all emotion scores to 0-1 range
                    normalized_emotions = {
                        emotion: score / 100.0 for emotion, score in emotion_data.items()
                    }
                    
                    return EmotionResult(
                        emotion=dominant_emotion,
                        confidence=dominant_confidence,
                        emotions=normalized_emotions
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
            raise ProcessingError(f"Error during emotion detection: {str(e)}")
    
    def detect_emotions_batch(self, face_images: List[np.ndarray]) -> List[EmotionResult]:
        """Detect emotions in multiple face images"""
        results = []
        for face_image in face_images:
            try:
                emotion_result = self.detect_emotion(face_image)
                results.append(emotion_result)
            except Exception as e:
                # Create a default result for failed detections
                default_emotions = {emotion: 0.0 for emotion in self._emotion_labels}
                default_emotions['neutral'] = 1.0
                
                results.append(EmotionResult(
                    emotion='neutral',
                    confidence=0.0,
                    emotions=default_emotions
                ))
        
        return results
    
    def is_model_loaded(self) -> bool:
        """Check if the emotion detection model is loaded"""
        return self._model_loaded
    
    def get_supported_emotions(self) -> List[str]:
        """Get list of supported emotion labels"""
        return self._emotion_labels.copy()
