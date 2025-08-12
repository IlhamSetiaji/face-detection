#!/usr/bin/env python3
"""
Example script demonstrating face detection usage
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.application.use_cases import FaceDetectionUseCase
from src.infrastructure.mtcnn_detector import MTCNNFaceDetector
from src.infrastructure.opencv_processor import OpenCVImageProcessor
from src.application.services import DetectionResultSerializer


def main():
    """Example usage of the face detection system"""
    
    # Initialize dependencies
    print("Initializing face detection system...")
    face_detector = MTCNNFaceDetector(confidence_threshold=0.7)
    image_processor = OpenCVImageProcessor()
    use_case = FaceDetectionUseCase(face_detector, image_processor)
    
    # Example image path (you'll need to provide your own image)
    image_path = "example_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Please place an image file named '{image_path}' in the current directory")
        print("Supported formats: jpg, jpeg, png, bmp, tiff, webp")
        return
    
    try:
        print(f"Processing image: {image_path}")
        
        # Detect faces
        result = use_case.detect_faces_in_image(image_path, confidence_threshold=0.6)
        
        # Print results
        print(f"\nDetection Results:")
        print(f"Number of faces detected: {result.get_face_count()}")
        print(f"Processing time: {result.processing_time:.3f} seconds")
        print(f"Image size: {result.original_image_size[0]}x{result.original_image_size[1]}")
        
        for i, face in enumerate(result.faces, 1):
            print(f"\nFace {i}:")
            print(f"  Confidence: {face.confidence:.3f}")
            print(f"  Bounding box: ({face.bbox[0]:.1f}, {face.bbox[1]:.1f}, {face.bbox[2]:.1f}, {face.bbox[3]:.1f})")
            print(f"  Width: {face.get_width():.1f}")
            print(f"  Height: {face.get_height():.1f}")
            print(f"  Area: {face.get_area():.1f}")
            
            if face.landmarks:
                print(f"  Landmarks: {len(face.landmarks)} points")
        
        # Create annotated image
        output_path = "results/annotated_example.jpg"
        annotated_result = use_case.detect_and_annotate(
            image_path, 
            output_path, 
            confidence_threshold=0.6
        )
        
        print(f"\nAnnotated image saved to: {output_path}")
        
        # Serialize to JSON
        json_result = DetectionResultSerializer.to_dict(result)
        print(f"\nJSON output available")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    main()
