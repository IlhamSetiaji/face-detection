#!/usr/bin/env python3
"""
Test script for emotion detection functionality
"""

import os
import sys
import requests
import json

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_emotion_detection_api():
    """Test emotion detection via API"""
    
    # API endpoint
    base_url = "http://localhost:5000"
    
    # Test health check
    print("Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print("❌ Health check failed")
            return
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        print("Make sure the Flask app is running on localhost:5000")
        return
    
    # Test with a sample image (you'll need to provide a test image)
    test_image_path = "test_image.jpg"  # Replace with actual test image path
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        print("Please provide a test image file")
        return
    
    print(f"\nTesting emotion detection with image: {test_image_path}")
    
    # Test face detection without emotions
    print("\n1. Testing face detection only...")
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            data = {'confidence': '0.5', 'emotions': 'false'}
            response = requests.post(f"{base_url}/detect", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Face detection successful")
            print(f"   Found {result['face_count']} face(s)")
            for i, face in enumerate(result['faces']):
                print(f"   Face {i+1}: confidence={face['confidence']:.2f}")
        else:
            print(f"❌ Face detection failed: {response.text}")
    except Exception as e:
        print(f"❌ Error during face detection: {e}")
    
    # Test face detection with emotions
    print("\n2. Testing face detection with emotions...")
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            data = {'confidence': '0.5', 'emotions': 'true'}
            response = requests.post(f"{base_url}/detect", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Face detection with emotions successful")
            print(f"   Found {result['face_count']} face(s)")
            for i, face in enumerate(result['faces']):
                print(f"   Face {i+1}: confidence={face['confidence']:.2f}")
                if 'emotion' in face:
                    emotion_data = face['emotion']
                    print(f"     Emotion: {emotion_data['dominant_emotion']} ({emotion_data['confidence']:.2f})")
                    print(f"     All emotions: {emotion_data['all_emotions']}")
                else:
                    print("     No emotion data")
        else:
            print(f"❌ Face detection with emotions failed: {response.text}")
    except Exception as e:
        print(f"❌ Error during emotion detection: {e}")
    
    # Test annotated image with emotions
    print("\n3. Testing annotated image with emotions...")
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            data = {'confidence': '0.5', 'emotions': 'true'}
            response = requests.post(f"{base_url}/detect-and-annotate", files=files, data=data)
        
        if response.status_code == 200:
            output_path = "test_emotion_annotated.jpg"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Annotated image with emotions saved to: {output_path}")
        else:
            print(f"❌ Annotated image generation failed: {response.text}")
    except Exception as e:
        print(f"❌ Error during annotated image generation: {e}")


def test_emotion_detection_direct():
    """Test emotion detection directly (without API)"""
    
    print("\n" + "="*50)
    print("Testing emotion detection directly...")
    
    try:
        from src.infrastructure.deepface_emotion_detector import DeepFaceEmotionDetector
        from src.infrastructure.mtcnn_detector import MTCNNFaceDetector
        from src.infrastructure.opencv_processor import OpenCVImageProcessor
        from src.application.use_cases import FaceDetectionUseCase
        
        # Initialize components
        print("Initializing components...")
        face_detector = MTCNNFaceDetector()
        emotion_detector = DeepFaceEmotionDetector()
        image_processor = OpenCVImageProcessor()
        use_case = FaceDetectionUseCase(face_detector, image_processor, emotion_detector)
        
        print("✅ Components initialized successfully")
        
        # Test with sample image
        test_image_path = "test_image.jpg"  # Replace with actual test image path
        
        if not os.path.exists(test_image_path):
            print(f"❌ Test image not found: {test_image_path}")
            return
        
        print(f"Processing image: {test_image_path}")
        
        # Test with emotion detection
        result = use_case.detect_faces_in_image(test_image_path, detect_emotions=True)
        
        print(f"✅ Detection completed in {result.processing_time:.2f}s")
        print(f"   Found {result.get_face_count()} face(s)")
        
        for i, face in enumerate(result.faces):
            print(f"   Face {i+1}:")
            print(f"     Position: {face.bbox}")
            print(f"     Confidence: {face.confidence:.2f}")
            if face.emotion:
                print(f"     Emotion: {face.emotion.emotion} ({face.emotion.confidence:.2f})")
                print(f"     All emotions: {face.emotion.emotions}")
            else:
                print("     No emotion data")
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Face Detection + Emotion Recognition Test")
    print("="*50)
    
    # Choose test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        test_emotion_detection_direct()
    else:
        print("Testing via API (make sure Flask app is running)")
        print("Use --direct flag to test without API")
        test_emotion_detection_api()
