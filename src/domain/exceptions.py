from enum import Enum


class DetectionError(Exception):
    """Base exception for face detection errors"""
    pass


class ModelNotLoadedError(DetectionError):
    """Raised when trying to use a detector with no loaded model"""
    pass


class InvalidImageError(DetectionError):
    """Raised when the provided image is invalid or cannot be processed"""
    pass


class ProcessingError(DetectionError):
    """Raised when an error occurs during image processing"""
    pass


class FileError(DetectionError):
    """Raised when there are file-related errors"""
    pass
