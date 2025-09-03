"""
Quick demo script to showcase the digit classification system
"""
from real_time_recognizer import DigitClassifier
import librosa
import time

def demo_classification():
    """Demonstrate the digit classification system with sample files."""
    print("🎤 Audio Digit Classification Demo")
    print("=" * 50)
    
    # Initialize classifier
    print("Loading trained model...")
    classifier = DigitClassifier()
    print("✅ Model loaded successfully!")
    
    # Test with demo files
    demo_files = [
        "demo_audio/digit_0_demo.wav",
        "demo_audio/digit_5_demo.wav", 
        "demo_audio/digit_9_demo.wav"
    ]
    
    print(f"\\n🧪 Testing with {len(demo_files)} demo files...")
    print("-" * 50)
    
    total_correct = 0
    total_time = 0
    
    for file_path in demo_files:
        # Extract digit from filename like "demo_audio/digit_0_demo.wav"
        filename = file_path.split('/')[-1]  # Get just the filename
        expected_digit = int(filename.split('_')[1])
        
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(file_path, sr=8000)
            
            # Make prediction
            start_time = time.time()
            prediction, confidence, proc_time = classifier.predict_digit(audio_data, sample_rate)
            end_time = time.time()
            
            # Calculate results
            is_correct = prediction == expected_digit
            total_time += proc_time
            if is_correct:
                total_correct += 1
            
            # Display results
            status_emoji = "✅" if is_correct else "❌"
            print(f"{status_emoji} File: {file_path}")
            print(f"   Expected: {expected_digit} | Predicted: {prediction} | Confidence: {confidence:.3f}")
            print(f"   Processing time: {proc_time*1000:.1f}ms | Audio duration: {len(audio_data)/sample_rate:.2f}s")
            print()
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    # Summary
    print("📊 Demo Results Summary")
    print("-" * 30)
    print(f"Accuracy: {total_correct}/{len(demo_files)} ({total_correct/len(demo_files)*100:.1f}%)")
    print(f"Average processing time: {total_time/len(demo_files)*1000:.1f}ms")
    print(f"Real-time performance: {0.5/(total_time/len(demo_files)):.1f}x faster than real-time")
    
    print(f"\\n🚀 System is ready for real-time audio input!")
    print("To try live microphone input, run: python real_time_recognizer.py")

if __name__ == "__main__":
    demo_classification()
