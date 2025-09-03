import pandas as pd
import numpy as np
import librosa
import io
import soundfile as sf
import pickle
from pathlib import Path

def create_demo_audio_files():
    """Create some demo audio files from the dataset for testing."""
    print("Creating demo audio files from dataset...")
    
    # Load test data
    test_df = pd.read_parquet('test-00000-of-00001.parquet')
    
    demo_dir = Path('demo_audio')
    demo_dir.mkdir(exist_ok=True)
    
    # Create one demo file for each digit
    for digit in range(10):
        # Find first sample of this digit
        digit_samples = test_df[test_df['label'] == digit]
        if len(digit_samples) > 0:
            sample = digit_samples.iloc[0]
            audio_dict = sample['audio']
            
            if 'bytes' in audio_dict:
                try:
                    # Extract audio from bytes
                    audio_buffer = io.BytesIO(audio_dict['bytes'])
                    audio_data, sample_rate = sf.read(audio_buffer)
                    
                    # Save as WAV file
                    output_path = demo_dir / f'digit_{digit}_demo.wav'
                    sf.write(output_path, audio_data, sample_rate)
                    print(f"Created: {output_path}")
                    
                except Exception as e:
                    print(f"Error creating demo file for digit {digit}: {e}")

def test_classifier_on_demos():
    """Test the classifier on demo audio files."""
    from real_time_recognizer import DigitClassifier
    
    print("\\nTesting classifier on demo files...")
    classifier = DigitClassifier()
    
    demo_dir = Path('demo_audio')
    if not demo_dir.exists():
        print("Demo directory not found. Creating demo files first...")
        create_demo_audio_files()
    
    correct_predictions = 0
    total_predictions = 0
    
    for audio_file in sorted(demo_dir.glob('*.wav')):
        expected_digit = int(audio_file.stem.split('_')[1])
        
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(audio_file, sr=8000)
            
            # Make prediction
            prediction, confidence, proc_time = classifier.predict_digit(audio_data, sample_rate)
            
            is_correct = prediction == expected_digit
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} {audio_file.name}: Expected={expected_digit}, Predicted={prediction}, Confidence={confidence:.3f}, Time={proc_time*1000:.1f}ms")
            
        except Exception as e:
            print(f"Error testing {audio_file}: {e}")
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\\nDemo Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

def benchmark_prediction_speed():
    """Benchmark prediction speed."""
    from real_time_recognizer import DigitClassifier
    import time
    
    print("\\nBenchmarking prediction speed...")
    classifier = DigitClassifier()
    
    # Create a sample audio signal
    sample_rate = 8000
    duration = 0.5
    audio_data = np.random.randn(int(sample_rate * duration))
    
    # Warm up
    for _ in range(5):
        classifier.predict_digit(audio_data, sample_rate)
    
    # Benchmark
    num_tests = 100
    start_time = time.time()
    
    for _ in range(num_tests):
        prediction, confidence, proc_time = classifier.predict_digit(audio_data, sample_rate)
    
    total_time = time.time() - start_time
    avg_time = total_time / num_tests
    
    print(f"Average prediction time: {avg_time*1000:.2f}ms")
    print(f"Predictions per second: {1/avg_time:.1f}")
    print(f"Real-time factor: {duration/avg_time:.1f}x (1x = real-time)")

if __name__ == "__main__":
    create_demo_audio_files()
    test_classifier_on_demos()
    benchmark_prediction_speed()
