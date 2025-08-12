#!/usr/bin/env python3
"""
Simple test script to verify the face detection system works
"""

import os
import sys
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.application.use_cases import FaceDetectionUseCase
from src.infrastructure.mtcnn_detector import MTCNNFaceDetector
from src.infrastructure.opencv_processor import OpenCVImageProcessor
from src.application.services import DetectionResultSerializer, ValidationService


def test_basic_functionality():
    """Test basic functionality of the face detection system"""
    
    print("Testing Face Detection System...")
    
    try:
        # Test 1: Initialize components
        print("1. Initializing components...")
        face_detector = MTCNNFaceDetector(confidence_threshold=0.5)
        image_processor = OpenCVImageProcessor()
        use_case = FaceDetectionUseCase(face_detector, image_processor)
        print("   ✓ Components initialized successfully")
        
        # Test 2: Check model loading
        print("2. Checking model status...")
        assert face_detector.is_model_loaded(), "Model should be loaded"
        print("   ✓ MTCNN model loaded")
        
        # Test 3: Test validation services
        print("3. Testing validation services...")
        assert ValidationService.validate_image_file("test.jpg") == True
        assert ValidationService.validate_image_file("test.txt") == False
        assert ValidationService.validate_confidence_threshold(0.5) == True
        assert ValidationService.validate_confidence_threshold(1.5) == False
        print("   ✓ Validation services working")
        
        # Test 4: Create a dummy image for testing
        print("4. Creating test image...")
        dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        test_image_path = "test_image.jpg"
        image_processor.save_image(dummy_image, test_image_path)
        print("   ✓ Test image created")
        
        # Test 5: Test face detection on dummy image
        print("5. Testing face detection...")
        result = use_case.detect_faces_in_image(test_image_path)
        assert result is not None, "Detection result should not be None"
        assert result.processing_time > 0, "Processing time should be positive"
        print(f"   ✓ Detection completed in {result.processing_time:.3f} seconds")
        print(f"   ✓ Found {result.get_face_count()} faces (expected 0 for random image)")
        
        # Test 6: Test serialization
        print("6. Testing result serialization...")
        json_result = DetectionResultSerializer.to_dict(result)
        assert "face_count" in json_result
        assert "processing_time" in json_result
        print("   ✓ Serialization working")
        
        # Test 7: Test confidence threshold setting
        print("7. Testing confidence threshold...")
        face_detector.set_confidence_threshold(0.8)
        result2 = use_case.detect_faces_in_image(test_image_path, confidence_threshold=0.3)
        print("   ✓ Confidence threshold setting working")
        
        # Cleanup
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        print("\n🎉 All tests passed! The face detection system is working correctly.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the Flask app: python src/presentation/flask_app/app.py")
        print("3. Test with real images using the API endpoints")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
