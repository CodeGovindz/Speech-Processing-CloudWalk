"""
Streamlit App Launcher for Audio Digit Classification System
Run this script to start the web interface
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required files are present."""
    required_files = [
        "digit_classifier_model.pkl",
        "feature_scaler.pkl", 
        "feature_names.pkl",
        "streamlit_app.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run 'python train_model.py' first to generate the model files.")
        return False
    
    print("✅ All required files are present!")
    return True

def check_demo_files():
    """Check if demo files exist, create them if not."""
    demo_dir = Path("demo_audio")
    if not demo_dir.exists() or len(list(demo_dir.glob("*.wav"))) == 0:
        print("⚠️ Demo audio files not found. Creating them...")
        try:
            import test_demo
            test_demo.create_demo_audio_files()
            print("✅ Demo files created successfully!")
        except Exception as e:
            print(f"❌ Error creating demo files: {e}")
            print("You can still use the app by uploading your own audio files.")

def main():
    print("🎤 Audio Digit Classification - Streamlit App Launcher")
    print("=" * 55)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check/create demo files
    check_demo_files()
    
    print("\n🚀 Starting Streamlit app...")
    print("The app will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.")
    print("-" * 55)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.headless", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Streamlit app stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting Streamlit: {e}")
        print("\nTry running manually:")
        print("streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
