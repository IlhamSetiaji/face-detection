"""
Configuration settings for the face detection application
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AppConfig:
    """Application configuration"""
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = True
    
    # File settings
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER: str = "uploads"
    RESULTS_FOLDER: str = "results"
    
    # Detection settings
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    MIN_CONFIDENCE_THRESHOLD: float = 0.0
    MAX_CONFIDENCE_THRESHOLD: float = 1.0
    
    # Supported image formats
    SUPPORTED_EXTENSIONS: set = frozenset({
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'
    })
    
    # Model settings
    ENABLE_LANDMARKS: bool = True
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables"""
        return cls(
            HOST=os.getenv('HOST', cls.HOST),
            PORT=int(os.getenv('PORT', cls.PORT)),
            DEBUG=os.getenv('DEBUG', 'True').lower() == 'true',
            MAX_CONTENT_LENGTH=int(os.getenv('MAX_CONTENT_LENGTH', cls.MAX_CONTENT_LENGTH)),
            UPLOAD_FOLDER=os.getenv('UPLOAD_FOLDER', cls.UPLOAD_FOLDER),
            RESULTS_FOLDER=os.getenv('RESULTS_FOLDER', cls.RESULTS_FOLDER),
            DEFAULT_CONFIDENCE_THRESHOLD=float(os.getenv('DEFAULT_CONFIDENCE_THRESHOLD', cls.DEFAULT_CONFIDENCE_THRESHOLD)),
        )


# Global configuration instance
config = AppConfig.from_env()
