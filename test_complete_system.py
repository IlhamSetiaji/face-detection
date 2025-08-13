#!/usr/bin/env python3
"""
Complete test for Face Detection + Emotion + Age Recognition API
"""

import os
import sys
import requests
import json
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

API_BASE = "http://localhost:5000"

def test_api_health():
    """Test API health endpoint"""
    print("🏥 Testing API health...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy: {data}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        return False

def test_face_detection_only():
    """Test basic face detection without emotion or age"""
    print("\n👤 Testing face detection only...")
    
    # Use existing test image if available
    test_images = [
        "results/oor.jpg",  # Check if any result images exist
        "uploads/test.jpg",
        # We'll use a simple test if no images exist
    ]
    
    image_path = None
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("⚠️ No test image found, skipping this test")
        return
    
    try:
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {
                'confidence': '0.5',
                'emotions': 'false',
                'age': 'false'
            }
            
            response = requests.post(f"{API_BASE}/detect", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Face detection successful!")
                print(f"   Found {result['face_count']} faces")
                print(f"   Processing time: {result['processing_time']:.2f}s")
                return True
            else:
                print(f"❌ Face detection failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing face detection: {e}")
        return False

def test_complete_analysis():
    """Test face detection + emotion + age analysis"""
    print("\n🎭🔢 Testing complete analysis (face + emotion + age)...")
    
    # Use existing test image if available
    test_images = [
        "results/oor.jpg",
        "uploads/test.jpg",
    ]
    
    image_path = None
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("⚠️ No test image found, skipping this test")
        return
    
    try:
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {
                'confidence': '0.5',
                'emotions': 'true',
                'age': 'true'
            }
            
            start_time = time.time()
            response = requests.post(f"{API_BASE}/detect", files=files, data=data)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Complete analysis successful!")
                print(f"   Found {result['face_count']} faces")
                print(f"   Total processing time: {result['processing_time']:.2f}s")
                print(f"   API response time: {end_time - start_time:.2f}s")
                
                # Show detailed results for each face
                for i, face in enumerate(result['faces']):
                    print(f"\n   Face {i+1}:")
                    print(f"     Confidence: {face['confidence']:.2f}")
                    
                    if 'emotion' in face:
                        emotion = face['emotion']
                        print(f"     Emotion: {emotion['dominant_emotion']} ({emotion['confidence']:.2f})")
                        print(f"     All emotions: {emotion['all_emotions']}")
                    
                    if 'age' in face:
                        age = face['age']
                        print(f"     Age: {age['estimated_age']:.0f} years")
                        print(f"     Age range: {age['age_range']['min']}-{age['age_range']['max']} years")
                
                return True
            else:
                print(f"❌ Complete analysis failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"   Raw response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing complete analysis: {e}")
        return False

def test_annotated_image():
    """Test generating annotated images"""
    print("\n🖼️ Testing annotated image generation...")
    
    # Use existing test image if available
    test_images = [
        "results/oor.jpg",
        "uploads/test.jpg",
    ]
    
    image_path = None
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("⚠️ No test image found, skipping this test")
        return
    
    try:
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {
                'confidence': '0.5',
                'emotions': 'true',
                'age': 'true'
            }
            
            response = requests.post(f"{API_BASE}/detect-and-annotate", files=files, data=data)
            
            if response.status_code == 200:
                # Save the annotated image
                output_path = "test_annotated_complete.jpg"
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Annotated image generated successfully!")
                print(f"   Saved to: {output_path}")
                return True
            else:
                print(f"❌ Annotated image generation failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Error testing annotated image: {e}")
        return False

def test_performance():
    """Test performance with different configurations"""
    print("\n⚡ Testing performance...")
    
    test_images = [
        "results/oor.jpg",
        "uploads/test.jpg",
    ]
    
    image_path = None
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break
    
    if not image_path:
        print("⚠️ No test image found, skipping this test")
        return
    
    configurations = [
        ("Face only", {'emotions': 'false', 'age': 'false'}),
        ("Face + Emotion", {'emotions': 'true', 'age': 'false'}),
        ("Face + Age", {'emotions': 'false', 'age': 'true'}),
        ("Face + Emotion + Age", {'emotions': 'true', 'age': 'true'}),
    ]
    
    print("   Configuration comparisons:")
    
    for config_name, params in configurations:
        try:
            with open(image_path, 'rb') as img_file:
                files = {'image': img_file}
                data = {'confidence': '0.5', **params}
                
                start_time = time.time()
                response = requests.post(f"{API_BASE}/detect", files=files, data=data)
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    api_time = end_time - start_time
                    processing_time = result['processing_time']
                    
                    print(f"     {config_name:20} | API: {api_time:.2f}s | Processing: {processing_time:.2f}s")
                else:
                    print(f"     {config_name:20} | Failed: {response.status_code}")
                    
        except Exception as e:
            print(f"     {config_name:20} | Error: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Complete Face Analysis API Tests")
    print("=" * 60)
    
    # Test 1: API Health
    if not test_api_health():
        print("\n❌ API is not running. Please start the Flask app first:")
        print("python src/presentation/flask_app/app.py")
        return
    
    # Test 2: Basic face detection
    test_face_detection_only()
    
    # Test 3: Complete analysis
    test_complete_analysis()
    
    # Test 4: Annotated images
    test_annotated_image()
    
    # Test 5: Performance
    test_performance()
    
    print("\n" + "=" * 60)
    print("🎉 Test suite completed!")
    print("\nUsage examples:")
    print("- Basic detection: POST /detect with image file")
    print("- With emotions: Add 'emotions=true' parameter")
    print("- With age: Add 'age=true' parameter")
    print("- Complete analysis: Add both 'emotions=true' and 'age=true'")
    print("- Annotated images: Use /detect-and-annotate endpoint")
    print("\nTest the web interface: open test_complete_client.html in your browser")

if __name__ == "__main__":
    main()
