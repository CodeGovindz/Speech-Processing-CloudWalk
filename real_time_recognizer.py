import numpy as np
import sounddevice as sd
import librosa
import pickle
import time
import queue
import threading
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DigitClassifier:
    def __init__(self, model_path='digit_classifier_model.pkl', 
                 scaler_path='feature_scaler.pkl', 
                 feature_names_path='feature_names.pkl'):
        """Load trained model and preprocessing components."""
        print("Loading trained model...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        with open(feature_names_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print(f"Model loaded successfully with {len(self.feature_names)} features")
        
    def extract_audio_features(self, audio_data, sample_rate):
        """Extract features from audio data (same as training)."""
        features = {}
        
        # Basic audio statistics
        features['duration'] = len(audio_data) / sample_rate
        features['sample_rate'] = sample_rate
        features['mean_amplitude'] = np.mean(audio_data)
        features['std_amplitude'] = np.std(audio_data)
        features['max_amplitude'] = np.max(audio_data)
        features['min_amplitude'] = np.min(audio_data)
        features['rms'] = np.sqrt(np.mean(audio_data**2))
        
        # Zero crossing rate
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # Spectral features
        try:
            # MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13, n_fft=min(512, len(audio_data)))
            for i in range(mfccs.shape[0]):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate, n_fft=min(512, len(audio_data)))
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate, n_fft=min(512, len(audio_data)))
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate, n_fft=min(512, len(audio_data)))
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, n_fft=min(512, len(audio_data)))
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
            
        except Exception as e:
            print(f"Error extracting spectral features: {e}")
            # Fill with zeros if extraction fails
            for i in range(13):
                features[f'mfcc_{i}_mean'] = 0
                features[f'mfcc_{i}_std'] = 0
            features['spectral_centroid_mean'] = 0
            features['spectral_centroid_std'] = 0
            features['spectral_bandwidth_mean'] = 0
            features['spectral_bandwidth_std'] = 0
            features['spectral_rolloff_mean'] = 0
            features['spectral_rolloff_std'] = 0
            features['chroma_mean'] = 0
            features['chroma_std'] = 0
        
        return features
    
    def predict_digit(self, audio_data, sample_rate):
        """Predict digit from audio data."""
        start_time = time.time()
        
        # Extract features
        features = self.extract_audio_features(audio_data, sample_rate)
        
        # Convert to array in correct order
        feature_array = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
        
        # Scale features
        feature_array_scaled = self.scaler.transform(feature_array)
        
        # Make prediction
        prediction = self.model.predict(feature_array_scaled)[0]
        confidence = np.max(self.model.predict_proba(feature_array_scaled)[0])
        
        processing_time = time.time() - start_time
        
        return prediction, confidence, processing_time

class RealTimeDigitRecognizer:
    def __init__(self):
        self.classifier = DigitClassifier()
        self.audio_queue = queue.Queue()
        self.prediction_queue = queue.Queue()
        self.is_recording = False
        self.sample_rate = 8000  # Match training data
        self.recording_duration = 1.0  # Record for 1 second
        self.silence_threshold = 0.01  # Amplitude threshold for detecting speech
        self.min_speech_duration = 0.3  # Minimum duration for speech detection
        
        # Results tracking
        self.recent_predictions = deque(maxlen=10)
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input."""
        if status:
            print(f"Audio status: {status}")
        
        if self.is_recording:
            self.audio_queue.put(indata.copy())
    
    def detect_speech_activity(self, audio_data):
        """Simple voice activity detection."""
        rms = np.sqrt(np.mean(audio_data**2))
        return rms > self.silence_threshold
    
    def process_audio(self):
        """Process audio from queue and make predictions."""
        buffer = []
        speech_detected = False
        speech_start = 0
        
        while True:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=1.0)
                buffer.extend(chunk.flatten())
                
                # Check for speech activity
                if self.detect_speech_activity(chunk):
                    if not speech_detected:
                        speech_detected = True
                        speech_start = len(buffer) - len(chunk.flatten())
                
                # If we have enough audio and detected speech, process it
                if len(buffer) >= int(self.sample_rate * self.recording_duration):
                    if speech_detected:
                        # Extract the speech portion
                        speech_end = len(buffer)
                        speech_audio = np.array(buffer[speech_start:speech_end])
                        
                        if len(speech_audio) >= int(self.sample_rate * self.min_speech_duration):
                            # Make prediction
                            try:
                                prediction, confidence, proc_time = self.classifier.predict_digit(
                                    speech_audio, self.sample_rate
                                )
                                
                                result = {
                                    'prediction': prediction,
                                    'confidence': confidence,
                                    'processing_time': proc_time,
                                    'audio_duration': len(speech_audio) / self.sample_rate,
                                    'timestamp': time.time()
                                }
                                
                                self.prediction_queue.put(result)
                                self.recent_predictions.append(result)
                                
                            except Exception as e:
                                print(f"Error in prediction: {e}")
                    
                    # Reset buffer
                    buffer = []
                    speech_detected = False
                    
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                break
    
    def start_recording(self):
        """Start real-time audio recording and processing."""
        print(f"Starting real-time digit recognition...")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Recording duration: {self.recording_duration}s")
        print(f"Silence threshold: {self.silence_threshold}")
        print("Speak digits (0-9) clearly. Press Ctrl+C to stop.")
        
        self.is_recording = True
        
        # Start audio processing thread
        process_thread = threading.Thread(target=self.process_audio, daemon=True)
        process_thread.start()
        
        try:
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
            ):
                print("\\nListening... (speak digits clearly)")
                
                while True:
                    try:
                        # Check for new predictions
                        result = self.prediction_queue.get(timeout=0.1)
                        
                        print(f"\\nPredicted digit: {result['prediction']}")
                        print(f"Confidence: {result['confidence']:.3f}")
                        print(f"Processing time: {result['processing_time']*1000:.1f}ms")
                        print(f"Audio duration: {result['audio_duration']:.2f}s")
                        
                        # Show recent predictions
                        if len(self.recent_predictions) > 1:
                            recent_digits = [str(p['prediction']) for p in list(self.recent_predictions)[-5:]]
                            print(f"Recent predictions: {' '.join(recent_digits)}")
                        
                    except queue.Empty:
                        continue
                    except KeyboardInterrupt:
                        break
                        
        except KeyboardInterrupt:
            print("\\nStopping...")
        finally:
            self.is_recording = False
    
    def test_with_file(self, audio_file_path):
        """Test the classifier with an audio file."""
        print(f"Testing with audio file: {audio_file_path}")
        
        try:
            audio_data, sample_rate = librosa.load(audio_file_path, sr=self.sample_rate)
            print(f"Loaded audio: {len(audio_data)} samples, {sample_rate} Hz, {len(audio_data)/sample_rate:.2f}s")
            
            prediction, confidence, proc_time = self.classifier.predict_digit(audio_data, sample_rate)
            
            print(f"\\nPrediction Results:")
            print(f"Predicted digit: {prediction}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Processing time: {proc_time*1000:.1f}ms")
            
        except Exception as e:
            print(f"Error testing with file: {e}")

def main():
    recognizer = RealTimeDigitRecognizer()
    
    print("=== Real-Time Digit Recognition System ===")
    print("1. Start real-time microphone recognition")
    print("2. Test with audio file")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        recognizer.start_recording()
    elif choice == "2":
        file_path = input("Enter audio file path: ").strip()
        recognizer.test_with_file(file_path)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
