#!/usr/bin/env python3
"""
Flask application entry point for face detection API
"""

import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from src.presentation.flask_app.api import FaceDetectionAPI


def create_app():
    """Application factory"""
    upload_folder = os.path.join(os.path.dirname(__file__), '../../../uploads')
    results_folder = os.path.join(os.path.dirname(__file__), '../../../results')
    
    api = FaceDetectionAPI(upload_folder, results_folder)
    return api.app


def main():
    """Main entry point"""
    try:
        upload_folder = os.path.join(os.path.dirname(__file__), '../../../uploads')
        results_folder = os.path.join(os.path.dirname(__file__), '../../../results')
        
        api = FaceDetectionAPI(upload_folder, results_folder)
        
        print("Starting Face Detection API...")
        print("Available endpoints:")
        print("  GET  /health - Health check")
        print("  POST /detect - Detect faces in image (returns JSON)")
        print("  POST /detect-and-annotate - Detect faces and return annotated image")
        print("\nServer starting on http://localhost:5000")
        
        api.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
