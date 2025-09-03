"""
Comprehensive demonstration of the Audio Digit Classification System
Run this to see all capabilities and performance metrics
"""

import os
import sys
from pathlib import Path

def print_banner():
    """Print a nice banner for the demo."""
    print("🎤" * 25)
    print("🎤  AUDIO DIGIT CLASSIFICATION SYSTEM  🎤")
    print("🎤" * 25)
    print()
    print("A lightweight, fast, and accurate system for")
    print("classifying spoken digits (0-9) from audio input")
    print()
    print("Key Features:")
    print("• 97% test accuracy")
    print("• ~60ms prediction time")
    print("• Real-time microphone support")
    print("• 8x faster than real-time processing")
    print("• 42 advanced audio features")
    print("• Random Forest classifier")
    print()

def check_files():
    """Check if all necessary files exist."""
    required_files = [
        "digit_classifier_model.pkl",
        "feature_scaler.pkl", 
        "feature_names.pkl",
        "train_model.py",
        "real_time_recognizer.py",
        "test_demo.py",
        "quick_demo.py"
    ]
    
    print("🔍 Checking system files...")
    all_present = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING!")
            all_present = False
    
    if Path("demo_audio").exists():
        demo_files = list(Path("demo_audio").glob("*.wav"))
        print(f"✅ demo_audio/ directory with {len(demo_files)} sample files")
    else:
        print("❌ demo_audio/ directory - MISSING!")
        all_present = False
    
    print()
    return all_present

def show_options():
    """Show available demo options."""
    print("🚀 Available Demonstrations:")
    print()
    print("1. Quick Demo - Test with 3 sample audio files")
    print("2. Full Test Suite - Test all 10 digits + benchmarking")
    print("3. Real-time Recognition - Use your microphone!")
    print("4. Technical Details - Show model architecture & performance")
    print("5. Exit")
    print()

def run_quick_demo():
    """Run the quick demo."""
    print("Running Quick Demo...")
    print("=" * 50)
    os.system("python quick_demo.py")

def run_full_test():
    """Run the full test suite."""
    print("Running Full Test Suite...")
    print("=" * 50)
    os.system("python test_demo.py")

def run_realtime():
    """Run real-time recognition."""
    print("Starting Real-time Recognition System...")
    print("=" * 50)
    print("This will start the microphone input system.")
    print("You can speak digits (0-9) and see live predictions!")
    print()
    input("Press Enter to continue or Ctrl+C to cancel...")
    os.system("python real_time_recognizer.py")

def show_technical_details():
    """Show technical information about the system."""
    print("🔬 Technical Details")
    print("=" * 50)
    print()
    print("MODEL ARCHITECTURE:")
    print("• Random Forest Classifier (100 estimators)")
    print("• 42 audio features per sample")
    print("• StandardScaler for feature normalization")
    print("• Trained on Free Spoken Digit Dataset (FSDD)")
    print()
    
    print("AUDIO FEATURES:")
    print("• Time domain: duration, amplitude stats, RMS, zero-crossing rate")
    print("• MFCCs: 13 coefficients (mean + std) = 26 features")
    print("• Spectral: centroid, bandwidth, rolloff (mean + std) = 6 features")
    print("• Chroma: harmonic content (mean + std) = 2 features")
    print("• Total: 8 + 26 + 6 + 2 = 42 features")
    print()
    
    print("PERFORMANCE METRICS:")
    print("• Test Accuracy: 97.0%")
    print("• Training Accuracy: 100.0%")
    print("• Average Prediction Time: ~62ms")
    print("• Real-time Factor: 8.0x")
    print("• Predictions per Second: 15.9")
    print()
    
    print("DATASET:")
    print("• Source: Free Spoken Digit Dataset (FSDD)")
    print("• Training: 2,700 samples (270 per digit)")
    print("• Testing: 300 samples (30 per digit)")
    print("• Format: 8kHz WAV files, ~0.5s duration")
    print("• Multiple English speakers")
    print()
    
    print("SYSTEM REQUIREMENTS:")
    print("• Python 3.8+")
    print("• Libraries: librosa, scikit-learn, sounddevice, numpy, pandas")
    print("• Memory: ~50MB for model")
    print("• CPU: Any modern processor (optimized for real-time)")
    print()

def main():
    """Main demo controller."""
    print_banner()
    
    if not check_files():
        print("❌ Some required files are missing!")
        print("Please ensure you've run 'python train_model.py' first.")
        return
    
    print("✅ All system files present and ready!")
    print()
    
    while True:
        show_options()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            print()
            
            if choice == "1":
                run_quick_demo()
            elif choice == "2":
                run_full_test()
            elif choice == "3":
                run_realtime()
            elif choice == "4":
                show_technical_details()
            elif choice == "5":
                print("👋 Thanks for trying the Audio Digit Classification System!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
            
            print()
            input("Press Enter to continue...")
            print()
            
        except KeyboardInterrupt:
            print("\\n\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
