# Face Detection with Clean Architecture

A face detection application built with Flask and MTCNN using clean architecture principles for framework independence and reusability.

## Features

- **Face Detection**: Powered by MTCNN for accurate face detection
- **Clean Architecture**: Separated into domain, application, infrastructure, and presentation layers
- **Framework Independent**: Easy to switch from Flask to other frameworks (FastAPI, Django, etc.)
- **REST API**: Simple HTTP endpoints for face detection
- **Image Annotation**: Automatic drawing of bounding boxes and confidence scores
- **Facial Landmarks**: Detection and visualization of facial key points

## Project Structure

```
face-detection/
├── src/
│   ├── domain/                 # Business entities and interfaces
│   │   ├── entities.py         # Core business entities
│   │   ├── interfaces.py       # Abstract interfaces
│   │   └── exceptions.py       # Domain exceptions
│   ├── application/            # Use cases and services
│   │   ├── use_cases.py        # Business logic
│   │   └── services.py         # Application services
│   ├── infrastructure/         # External dependencies implementation
│   │   ├── mtcnn_detector.py       # MTCNN implementation
│   │   └── opencv_processor.py     # OpenCV image processing
│   └── presentation/           # Web framework layer
│       └── flask_app/
│           ├── api.py          # Flask API implementation
│           └── app.py          # Application entry point
├── uploads/                    # Temporary uploaded files
├── results/                    # Processed images output
├── requirements.txt
├── example_usage.py           # Example usage script
└── README.md
```

## Installation

1. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Flask API

```bash
python src/presentation/flask_app/app.py
```

The server will start on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```
GET /health
```

#### 2. Face Detection (JSON Response)
```
POST /detect
Content-Type: multipart/form-data

Parameters:
- image: Image file (jpg, png, etc.)
- confidence: Optional confidence threshold (0.0-1.0)
```

#### 3. Face Detection with Annotation
```
POST /detect-and-annotate
Content-Type: multipart/form-data

Parameters:
- image: Image file
- confidence: Optional confidence threshold (0.0-1.0)

Returns: Annotated image file
```

### Example Usage

```python
import requests

# Detect faces
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/detect',
        files={'image': f},
        data={'confidence': 0.7}
    )
    
result = response.json()
print(f"Found {result['face_count']} faces")
```

### Using as Library

```python
from src.application.use_cases import FaceDetectionUseCase
from src.infrastructure.mtcnn_detector import MTCNNFaceDetector
from src.infrastructure.opencv_processor import OpenCVImageProcessor

# Initialize
detector = MTCNNFaceDetector(confidence_threshold=0.7)
processor = OpenCVImageProcessor()
use_case = FaceDetectionUseCase(detector, processor)

# Detect faces
result = use_case.detect_faces_in_image('image.jpg')
print(f"Found {result.get_face_count()} faces")
```

## Switching Frameworks

The clean architecture makes it easy to switch frameworks:

### FastAPI Example
```python
from fastapi import FastAPI, File, UploadFile
from src.application.use_cases import FaceDetectionUseCase
# ... use the same use cases and infrastructure

app = FastAPI()

@app.post("/detect")
async def detect_faces(image: UploadFile = File(...)):
    # Same business logic, different framework
    pass
```

### Django Example
```python
from django.http import JsonResponse
from src.application.use_cases import FaceDetectionUseCase
# ... use the same use cases and infrastructure

def detect_faces(request):
    # Same business logic, different framework
    pass
```

## Configuration

The application uses default configurations, but you can customize:

- **Confidence Threshold**: Minimum confidence for face detection (0.0-1.0)
- **Upload Directory**: Where temporary files are stored
- **Results Directory**: Where annotated images are saved
- **Max File Size**: Maximum upload file size (default: 16MB)

## Error Handling

The application includes comprehensive error handling:

- **Invalid Image Format**: Unsupported file types
- **File Too Large**: Files exceeding size limit
- **Processing Errors**: Issues during face detection
- **Model Loading**: RetinaFace initialization problems

## Performance Considerations

- **Model Loading**: MTCNN model loads on first use
- **Memory Usage**: Images are processed in memory
- **File Cleanup**: Temporary files are automatically cleaned up
- **Concurrent Requests**: Flask handles multiple requests

## Development

### Adding New Detectors

1. Implement the `FaceDetectorInterface`
2. Add to infrastructure layer
3. Update dependency injection

### Adding New Frameworks

1. Create new presentation layer
2. Use existing application and infrastructure layers
3. Implement framework-specific routing

## Dependencies

- **Flask**: Web framework
- **MTCNN**: Face detection model
- **OpenCV**: Image processing
- **NumPy**: Numerical operations
- **Pillow**: Image handling

## License

This project is provided as-is for educational and development purposes.
