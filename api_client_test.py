#!/usr/bin/env python3
"""
Example client for testing the Face Detection API
"""

import requests
import json
import sys
import os


def test_health_endpoint(base_url: str = "http://localhost:5000"):
    """Test the health endpoint"""
    print("Testing health endpoint...")
    
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print(f"✓ Health check passed: {response.json()}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to server. Make sure the Flask app is running.")
        return False
    except Exception as e:
        print(f"✗ Health check error: {str(e)}")
        return False


def test_face_detection(image_path: str, base_url: str = "http://localhost:5000"):
    """Test face detection endpoint"""
    print(f"Testing face detection with image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"✗ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'confidence': 0.6}
            
            response = requests.post(f"{base_url}/detect", files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Detection successful!")
            print(f"  Faces found: {result['face_count']}")
            print(f"  Processing time: {result['processing_time']:.3f}s")
            print(f"  Image size: {result['original_image_size']['width']}x{result['original_image_size']['height']}")
            
            for i, face in enumerate(result['faces'], 1):
                print(f"  Face {i}: confidence={face['confidence']:.3f}, bbox=({face['bbox']['x1']:.1f},{face['bbox']['y1']:.1f},{face['bbox']['x2']:.1f},{face['bbox']['y2']:.1f})")
            
            return True
        else:
            print(f"✗ Detection failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Detection error: {str(e)}")
        return False


def test_annotated_detection(image_path: str, output_path: str = "annotated_result.jpg", base_url: str = "http://localhost:5000"):
    """Test face detection with annotation"""
    print(f"Testing annotated detection with image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"✗ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'confidence': 0.6}
            
            response = requests.post(f"{base_url}/detect-and-annotate", files=files, data=data)
            
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"✓ Annotated image saved to: {output_path}")
            return True
        else:
            print(f"✗ Annotated detection failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Annotated detection error: {str(e)}")
        return False


def main():
    """Main test function"""
    base_url = "http://localhost:5000"
    
    print("Face Detection API Client Test")
    print("=" * 40)
    
    # Test 1: Health check
    if not test_health_endpoint(base_url):
        print("\nPlease start the Flask application first:")
        print("python src/presentation/flask_app/app.py")
        return
    
    print()
    
    # Test 2: Face detection (requires test image)
    test_image = "test_image.jpg"
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    if os.path.exists(test_image):
        test_face_detection(test_image, base_url)
        print()
        test_annotated_detection(test_image, "annotated_test.jpg", base_url)
    else:
        print(f"No test image found at '{test_image}'")
        print("Usage: python api_client_test.py <image_path>")
        print("Or place a test image named 'test_image.jpg' in the current directory")


if __name__ == '__main__':
    main()
