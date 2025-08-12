#!/usr/bin/env python3
"""
Project structure verification script
"""

import os
import sys


def check_project_structure():
    """Verify that all necessary files and directories exist"""
    
    print("Face Detection Project Structure Verification")
    print("=" * 50)
    
    base_path = os.path.dirname(__file__)
    
    # Define expected structure
    expected_structure = {
        "files": [
            "requirements.txt",
            "README.md",
            "setup.bat",
            "example_usage.py",
            "api_client_test.py",
            "test_system.py",
            "src/__init__.py",
            "src/config.py",
            "src/domain/__init__.py",
            "src/domain/entities.py",
            "src/domain/interfaces.py",
            "src/domain/exceptions.py",
            "src/application/__init__.py",
            "src/application/use_cases.py",
            "src/application/services.py",
            "src/infrastructure/__init__.py",
            "src/infrastructure/retinaface_detector.py",
            "src/infrastructure/opencv_processor.py",
            "src/presentation/__init__.py",
            "src/presentation/flask_app/__init__.py",
            "src/presentation/flask_app/api.py",
            "src/presentation/flask_app/app.py",
            "src/presentation/fastapi_app/__init__.py",
            "src/presentation/fastapi_app/api.py",
        ],
        "directories": [
            "src",
            "src/domain",
            "src/application", 
            "src/infrastructure",
            "src/presentation",
            "src/presentation/flask_app",
            "src/presentation/fastapi_app",
            "uploads",
            "results",
        ]
    }
    
    # Check directories
    print("Checking directories...")
    missing_dirs = []
    for directory in expected_structure["directories"]:
        dir_path = os.path.join(base_path, directory)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"  ✓ {directory}")
        else:
            print(f"  ✗ {directory}")
            missing_dirs.append(directory)
    
    print()
    
    # Check files
    print("Checking files...")
    missing_files = []
    for file_path in expected_structure["files"]:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path}")
            missing_files.append(file_path)
    
    print()
    
    # Summary
    if not missing_dirs and not missing_files:
        print("🎉 Project structure is complete!")
        print("\nNext steps:")
        print("1. Install Python 3.13.6 if not already installed")
        print("2. Run setup.bat (Windows) or create virtual environment manually")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Test the system: python test_system.py")
        print("5. Start Flask app: python src/presentation/flask_app/app.py")
        return True
    else:
        print("❌ Project structure is incomplete!")
        if missing_dirs:
            print(f"Missing directories: {', '.join(missing_dirs)}")
        if missing_files:
            print(f"Missing files: {', '.join(missing_files)}")
        return False


def check_python_version():
    """Check Python version"""
    print(f"Python version: {sys.version}")
    
    version_info = sys.version_info
    if version_info.major == 3 and version_info.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8+ required")
        return False


def main():
    """Main verification function"""
    print("Starting project verification...\n")
    
    python_ok = check_python_version()
    print()
    
    structure_ok = check_project_structure()
    print()
    
    if python_ok and structure_ok:
        print("✅ Project is ready for setup!")
        print("\nArchitecture Overview:")
        print("- Domain: Core business logic and entities")
        print("- Application: Use cases and business rules")
        print("- Infrastructure: External dependencies (RetinaFace, OpenCV)")
        print("- Presentation: Web frameworks (Flask, FastAPI)")
        print("\nThis clean architecture allows easy switching between frameworks!")
    else:
        print("❌ Please fix the issues above before proceeding")
    
    return python_ok and structure_ok


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
