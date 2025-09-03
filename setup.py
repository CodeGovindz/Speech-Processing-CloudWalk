#!/usr/bin/env python3
"""
Setup Script for Audio Digit Classification System
This script helps users set up the project quickly and correctly.
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

def print_banner():
    """Print setup banner."""
    print("🎤" * 50)
    print("🎤  AUDIO DIGIT CLASSIFIER - SETUP SCRIPT  🎤")
    print("🎤" * 50)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print("✅ Python version:", sys.version.split()[0])
    return True

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_dataset():
    """Check if dataset files are present."""
    print("\n📊 Checking dataset files...")
    train_file = Path("train-00000-of-00001.parquet")
    test_file = Path("test-00000-of-00001.parquet")
    
    if train_file.exists() and test_file.exists():
        print("✅ Dataset files found!")
        return True
    else:
        print("⚠️ Dataset files not found.")
        print("\nPlease download the Free Spoken Digit Dataset:")
        print("1. Visit: https://huggingface.co/datasets/mteb/free-spoken-digit-dataset")
        print("2. Download: train-00000-of-00001.parquet")
        print("3. Download: test-00000-of-00001.parquet")
        print("4. Place both files in the project root directory")
        print("5. Run this setup script again")
        return False

def train_model():
    """Train the ML model."""
    print("\n🤖 Training the machine learning model...")
    print("This may take a few minutes...")
    
    try:
        result = subprocess.run([sys.executable, "train_model.py"], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            return True
        else:
            print(f"❌ Error during training: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training timed out (>10 minutes). Please check for issues.")
        return False
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def create_demo_files():
    """Create demo audio files."""
    print("\n🎵 Creating demo audio files...")
    try:
        subprocess.check_call([sys.executable, "create_demos.py"])
        print("✅ Demo files created!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Could not create demo files: {e}")
        print("You can still use the system by uploading your own audio files.")
        return False

def run_quick_test():
    """Run a quick system test."""
    print("\n🧪 Running quick system test...")
    try:
        result = subprocess.run([sys.executable, "quick_demo.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ System test passed!")
            print("Sample output:", result.stdout.split('\n')[-3:-1])
            return True
        else:
            print(f"⚠️ System test had issues: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ System test timed out.")
        return False
    except Exception as e:
        print(f"⚠️ Could not run system test: {e}")
        return False

def show_next_steps():
    """Show what to do next."""
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print()
    print("🚀 Next Steps:")
    print("1. Launch Web Interface:")
    print("   python launch_streamlit.py")
    print()
    print("2. Or try Command Line:")
    print("   python quick_demo.py")
    print("   python demo_system.py")
    print("   python real_time_recognizer.py")
    print()
    print("3. Read Documentation:")
    print("   README.md - Main documentation")
    print("   STREAMLIT_README.md - Web interface guide")
    print()
    print("📖 For help and troubleshooting:")
    print("   https://github.com/yourusername/audio-digit-classifier")
    print()

def main():
    """Main setup process."""
    print_banner()
    
    # Step 1: Check Python version
    if not check_python_version():
        return
    
    # Step 2: Install requirements
    if not install_requirements():
        print("\n❌ Setup failed at package installation.")
        return
    
    # Step 3: Check dataset
    if not check_dataset():
        print("\n❌ Setup paused - please download dataset files first.")
        return
    
    # Step 4: Train model
    if not train_model():
        print("\n❌ Setup failed at model training.")
        return
    
    # Step 5: Create demo files
    create_demo_files()  # Non-critical, continue even if it fails
    
    # Step 6: Quick test
    run_quick_test()  # Non-critical, continue even if it fails
    
    # Step 7: Show next steps
    show_next_steps()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        print("Please check the README.md for manual setup instructions.")
