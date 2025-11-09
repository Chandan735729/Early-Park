#!/usr/bin/env python3
"""
EarlyPark Setup Script
Automates the setup process for the Parkinson's Detection System
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None, silent=False):
    """Run a shell command and return success status"""
    try:
        if not silent:
            print(f"🔧 Running: {cmd}")
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        if not silent:
            print(f"❌ Command failed: {e}")
        return False

def check_tool_exists(tool_name):
    """Check if a tool exists using platform-appropriate command"""
    import platform
    
    if platform.system() == "Windows":
        # Use 'where' command on Windows
        return run_command(f"where {tool_name}", silent=True)
    else:
        # Use 'which' command on Unix-like systems
        return run_command(f"which {tool_name}", silent=True)

def check_prerequisites():
    """Check if required tools are installed"""
    print("🔍 Checking prerequisites...")
    
    requirements = {
        'docker': 'Docker is required for containerization',
        'docker-compose': 'Docker Compose is required for multi-container setup',
        'flutter': 'Flutter is required for mobile app development',
        'python': 'Python 3.8+ is required'
    }
    
    missing = []
    for tool, description in requirements.items():
        print(f"  Checking {tool}...")
        if not check_tool_exists(tool):
            missing.append(f"{tool}: {description}")
        else:
            print(f"  ✅ {tool} found")
    
    if missing:
        print("\n❌ Missing prerequisites:")
        for item in missing:
            print(f"  - {item}")
        print("\n💡 Installation tips:")
        print("  - Docker: Download from https://docker.com/get-started")
        print("  - Flutter: Download from https://flutter.dev/docs/get-started/install")
        print("  - Python: Already installed (you're running this script!)")
        return False
    
    print("\n✅ All prerequisites found!")
    return True

def setup_flutter():
    """Setup Flutter dependencies"""
    print("\n📱 Setting up Flutter...")
    
    flutter_dir = Path("early_park")
    if not flutter_dir.exists():
        print("❌ Flutter project directory not found")
        return False
    
    # Get Flutter dependencies
    if not run_command("flutter pub get", cwd=flutter_dir):
        return False
    
    # Check Flutter setup
    if not run_command("flutter doctor", cwd=flutter_dir):
        print("⚠️ Flutter doctor found issues, but continuing...")
    
    return True

def setup_python_env():
    """Setup Python environments for API and training"""
    print("\n🐍 Setting up Python environments...")
    
    # Setup API environment
    api_dir = Path("early_park/api")
    if api_dir.exists():
        print("Setting up API environment...")
        if not run_command("pip install -r requirements.txt", cwd=api_dir):
            return False
    
    # Setup training environment
    training_dir = Path("early_park/training")
    if training_dir.exists():
        print("Setting up training environment...")
        if not run_command("pip install -r requirements.txt", cwd=training_dir):
            return False
    
    return True

def test_api():
    """Test the API with existing model"""
    print("\n🧪 Testing API...")
    
    api_dir = Path("early_park/api")
    if not api_dir.exists():
        print("❌ API directory not found")
        return False
    
    # Check if model files exist
    model_files = [
        "final_parkinsons_regressor.pkl",
        "parkinsons_scaler.pkl"
    ]
    
    project_root = Path(".")
    missing_models = []
    for model_file in model_files:
        if not (project_root / model_file).exists():
            missing_models.append(model_file)
    
    if missing_models:
        print(f"⚠️ Missing model files: {missing_models}")
        print("You may need to train the model first or provide the files.")
        return False
    
    print("✅ Model files found!")
    return True

def build_docker():
    """Build Docker containers"""
    print("\n🐳 Building Docker containers...")
    
    if not run_command("docker-compose build"):
        return False
    
    print("✅ Docker containers built successfully!")
    return True

def main():
    """Main setup function"""
    print("🚀 EarlyPark Setup Script")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Setup failed due to missing prerequisites")
        sys.exit(1)
    
    # Setup Flutter
    if not setup_flutter():
        print("\n❌ Flutter setup failed")
        sys.exit(1)
    
    # Setup Python environments
    if not setup_python_env():
        print("\n❌ Python environment setup failed")
        sys.exit(1)
    
    # Test API
    if not test_api():
        print("\n⚠️ API test had issues, but continuing...")
    
    # Build Docker containers
    if not build_docker():
        print("\n❌ Docker build failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'docker-compose up' to start the system")
    print("2. Open http://localhost:8080 for the web app")
    print("3. API will be available at http://localhost:5000")
    print("4. Use 'python early_park/training/train_model.py' to retrain models")

if __name__ == "__main__":
    main()