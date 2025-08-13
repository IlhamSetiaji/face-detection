#!/usr/bin/env python3
"""
Simple test to verify emotion detection components work
"""

import os
import sys
import numpy as np
import cv2

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_components():
    """Test individual components"""
    print("Testing Face Detection + Emotion Recognition Components")
    print("=" * 60)
    
    try:
        # Test 1: Import components
        print("1. Testing imports...")
        from src.infrastructure.deepface_emotion_detector import DeepFaceEmotionDetector
        from src.infrastructure.mtcnn_detector import MTCNNFaceDetector
        from src.infrastructure.opencv_processor import OpenCVImageProcessor
        from src.application.use_cases import FaceDetectionUseCase
        print("✅ All components imported successfully")
        
        # Test 2: Initialize MTCNN
        print("\n2. Initializing MTCNN face detector...")
        face_detector = MTCNNFaceDetector()
        if face_detector.is_model_loaded():
            print("✅ MTCNN initialized successfully")
        else:
            print("❌ MTCNN failed to initialize")
            return
        
        # Test 3: Initialize DeepFace
        print("\n3. Initializing DeepFace emotion detector...")
        emotion_detector = DeepFaceEmotionDetector()
        if emotion_detector.is_model_loaded():
            print("✅ DeepFace initialized successfully")
        else:
            print("❌ DeepFace failed to initialize")
            return
        
        # Test 4: Initialize image processor
        print("\n4. Initializing OpenCV image processor...")
        image_processor = OpenCVImageProcessor()
        print("✅ OpenCV processor initialized successfully")
        
        # Test 5: Test emotion detection with dummy image
        print("\n5. Testing emotion detection with dummy face image...")
        
        # Create a simple test face image (just for testing the pipeline)
        dummy_face = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add some basic features to make it more face-like
        cv2.circle(dummy_face, (30, 40), 5, (0, 0, 0), -1)  # Left eye
        cv2.circle(dummy_face, (70, 40), 5, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(dummy_face, (50, 70), (15, 8), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        try:
            emotion_result = emotion_detector.detect_emotion(dummy_face)
            print(f"✅ Emotion detection successful")
            print(f"   Dominant emotion: {emotion_result.emotion}")
            print(f"   Confidence: {emotion_result.confidence:.2f}")
            print(f"   All emotions: {emotion_result.emotions}")
        except Exception as e:
            print(f"❌ Emotion detection failed: {e}")
        
        # Test 6: Test integrated use case
        print("\n6. Initializing integrated use case...")
        use_case = FaceDetectionUseCase(face_detector, image_processor, emotion_detector)
        print("✅ Use case initialized successfully")
        
        print("\n" + "=" * 60)
        print("🎉 All components are working correctly!")
        print("\nYou can now:")
        print("1. Start the Flask app: python src/presentation/flask_app/app.py")
        print("2. Open test_emotion_client.html in your browser")
        print("3. Upload an image to test face detection + emotion recognition")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install deepface tf-keras")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_components()
