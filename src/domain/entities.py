from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class EmotionResult:
    """Domain entity representing emotion analysis result"""
    emotion: str  # dominant emotion
    confidence: float  # confidence for dominant emotion
    emotions: Dict[str, float]  # all emotion probabilities


@dataclass
class AgeResult:
    """Domain entity representing age analysis result"""
    age: float  # estimated age
    age_range: Tuple[int, int]  # estimated age range (min, max)
    

@dataclass
class FaceDetection:
    """Domain entity representing a detected face"""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[List[Tuple[float, float]]] = None  # facial landmarks
    emotion: Optional[EmotionResult] = None  # emotion analysis result
    age: Optional[AgeResult] = None  # age analysis result
    
    def get_width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    def get_height(self) -> float:
        return self.bbox[3] - self.bbox[1]
    
    def get_area(self) -> float:
        return self.get_width() * self.get_height()


@dataclass
class DetectionResult:
    """Domain entity representing the result of face detection"""
    image_path: str
    faces: List[FaceDetection]
    processing_time: float
    original_image_size: Tuple[int, int]  # (width, height)
    
    def get_face_count(self) -> int:
        return len(self.faces)
    
    def has_faces(self) -> bool:
        return len(self.faces) > 0
