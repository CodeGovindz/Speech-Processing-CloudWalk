# 🎤 Audio Digit Classification System

A lightweight, fast, and accurate system for classifying spoken digits (0-9) from audio input using machine learning. Features both a command-line interface and a beautiful web UI built with Streamlit.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This project implements a lightweight prototype that listens to spoken digits and predicts the correct number with high accuracy and minimal latency. The system achieves **97% test accuracy** with **~60ms prediction time** per sample.

**🌟 Key Features:**
- 🎯 **97% Accuracy** on test data
- ⚡ **~60ms Processing Time** per audio sample  
- 🖥️ **Interactive Web Interface** built with Streamlit
- 🎤 **Real-time Audio Processing** capabilities
- 📊 **Comprehensive Analytics** and visualizations
- 🔧 **Easy Setup** with automated model training

## 🚀 Quick Start

### Option 1: Web Interface (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/audio-digit-classifier.git
cd audio-digit-classifier

# Install dependencies
pip install -r requirements.txt

# Download the dataset (see Dataset Setup section below)

# Train the model
python train_model.py

# Launch the web interface
python launch_streamlit.py
```

### Option 2: Command Line Interface
```bash
# Train the model (if not already done)
python train_model.py

# Run quick demo
python quick_demo.py

# Test real-time recognition
python real_time_recognizer.py
```

## 📊 Dataset Setup

This project uses the [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset). 

**Download the dataset:**
1. Visit the [FSDD Hugging Face page](https://huggingface.co/datasets/mteb/free-spoken-digit-dataset)
2. Download the `train-00000-of-00001.parquet` and `test-00000-of-00001.parquet` files
3. Place them in the project root directory

**Dataset Information:**
- 3,000 total recordings (2,700 train + 300 test)
- 10 digits (0-9) with equal distribution
- Multiple English speakers (male and female)
- 8kHz sampling rate, mono channel
- Average duration: ~0.5 seconds

## 🖥️ Web Interface Features

The Streamlit web interface provides an intuitive way to interact with the classifier:

### 🏠 Home Page
- **File Upload**: Drag & drop audio files (WAV, MP3, FLAC)
- **Demo Files**: Pre-loaded samples for each digit (0-9)
- **Real-time Results**: Instant predictions with confidence scores
- **Audio Visualization**: Interactive waveform plots

### 🧪 Testing Suite
- **Batch Testing**: Process multiple files simultaneously
- **Performance Metrics**: Accuracy, processing time analysis
- **Interactive Charts**: Confidence scores and result visualization

### 📊 Analytics Dashboard
- **Feature Importance**: See which audio features matter most
- **Model Performance**: Comprehensive evaluation metrics
- **System Benchmarks**: Processing speed and efficiency analysis

## 📈 Performance Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 97.0% |
| **Training Accuracy** | 100.0% |
| **Average Prediction Time** | 62.86ms |
| **Real-time Factor** | 8.0x |
| **Predictions per Second** | 15.9 |

## 🏗️ Technical Architecture

### Feature Engineering (42 Features)
- **Time Domain (8)**: Duration, amplitude statistics, zero-crossing rate
- **MFCCs (26)**: 13 coefficients with mean & standard deviation
- **Spectral (6)**: Centroid, bandwidth, rolloff features  
- **Chroma (2)**: Harmonic content analysis

### Machine Learning Pipeline
- **Model**: Random Forest Classifier (100 estimators)
- **Preprocessing**: StandardScaler for feature normalization
- **Training**: 2,700 samples with stratified validation
- **Evaluation**: 300 test samples with comprehensive metrics

## 📁 Project Structure

```
audio-digit-classifier/
├── 📄 Core Scripts
│   ├── train_model.py              # Model training pipeline
│   ├── real_time_recognizer.py     # Real-time audio processing
│   └── create_demos.py            # Generate demo audio files
│
├── 🌐 Web Interface
│   ├── streamlit_app.py           # Main Streamlit application
│   ├── launch_streamlit.py        # Web app launcher
│   └── .streamlit/config.toml     # UI configuration
│
├── 🧪 Testing & Demos
│   ├── quick_demo.py              # Quick testing script
│   ├── test_demo.py               # Comprehensive testing
│   └── demo_system.py             # Interactive demo menu
│
├── 📚 Documentation
│   ├── README.md                  # Main documentation
│   ├── STREAMLIT_README.md        # Web interface guide
│   └── requirements.txt           # Python dependencies
│
└── 📊 Generated Files (after training)
    ├── digit_classifier_model.pkl # Trained ML model
    ├── feature_scaler.pkl         # Feature normalization
    ├── feature_names.pkl          # Feature mappings
    ├── demo_audio/                # Sample audio files
    ├── confusion_matrix.png       # Performance visualization
    └── feature_importance.png     # Feature analysis plot
```

## � Installation & Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 2GB+ RAM recommended
- **Storage**: ~100MB for models and dependencies
- **Audio**: Microphone (optional, for real-time testing)

### Dependencies
```bash
# Core ML and Audio Processing
pandas>=1.5.0
numpy>=1.24.0
librosa>=0.10.0
scikit-learn>=1.3.0
soundfile>=0.12.0
sounddevice>=0.4.0

# Web Interface
streamlit>=1.28.0
plotly>=5.15.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Installation Steps
1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/audio-digit-classifier.git
   cd audio-digit-classifier
   ```

2. **Create Virtual Environment** (recommended)
   ```bash
   python -m venv audio_env
   source audio_env/bin/activate  # On Windows: audio_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset**
   - Get `train-00000-of-00001.parquet` and `test-00000-of-00001.parquet`
   - Place in project root directory

5. **Train Model**
   ```bash
   python train_model.py
   ```

6. **Launch Application**
   ```bash
   python launch_streamlit.py
   ```

## 🎮 Usage Examples

### Web Interface (Recommended)
```bash
# Launch the web application
python launch_streamlit.py

# Opens automatically in browser at http://localhost:8501
# Navigate through different pages:
# - Home: Upload files and get instant results
# - Test: Batch processing and analysis
# - Analytics: Model performance insights
# - Documentation: Detailed technical information
```

### Command Line Interface
```bash
# Quick demo with sample files
python quick_demo.py

# Interactive demo system
python demo_system.py

# Real-time microphone input
python real_time_recognizer.py
```

### Programmatic Usage
```python
from real_time_recognizer import DigitClassifier
import librosa

# Load the classifier
classifier = DigitClassifier()

# Load an audio file
audio_data, sample_rate = librosa.load('path/to/audio.wav', sr=8000)

# Make prediction
prediction, confidence, processing_time = classifier.predict_digit(audio_data, sample_rate)

print(f"Predicted digit: {prediction}")
print(f"Confidence: {confidence:.3f}")
print(f"Processing time: {processing_time*1000:.1f}ms")
```

## � Technical Details

### Audio Processing Pipeline
1. **Preprocessing**: Convert to 8kHz mono audio
2. **Feature Extraction**: Extract 42 audio features using librosa
3. **Normalization**: StandardScaler for consistent feature ranges
4. **Classification**: Random Forest prediction with confidence scores

### Key Features Extracted
- **MFCCs**: Capture spectral characteristics crucial for speech
- **Zero Crossing Rate**: Indicates voicing patterns
- **Spectral Features**: Frequency distribution properties
- **Temporal Features**: Duration and amplitude statistics

### Model Performance
- **Cross-validation**: Stratified K-fold during development
- **Metrics**: Precision, recall, F1-score all ~97%
- **Inference Speed**: 8x faster than real-time
- **Memory Usage**: ~50MB for loaded model

## �️ Development & Contribution

### Running Tests
```bash
# Test with demo files
python test_demo.py

# Run comprehensive benchmark
python quick_demo.py

# System verification
python demo_system.py
```

### Adding New Features
1. **Audio Features**: Modify `extract_audio_features()` in `real_time_recognizer.py`
2. **Models**: Experiment with different classifiers in `train_model.py`
3. **UI Components**: Add new pages/features in `streamlit_app.py`

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Free Spoken Digit Dataset (FSDD)**: Training data
- **librosa**: Exceptional audio processing library
- **scikit-learn**: Robust machine learning framework
- **Streamlit**: Beautiful web interface framework

## � Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/audio-digit-classifier/issues)
- **Documentation**: See `STREAMLIT_README.md` for web interface details
- **Dataset**: [FSDD on Hugging Face](https://huggingface.co/datasets/mteb/free-spoken-digit-dataset)

---

*🎤 Built with passion for audio ML and clean code architecture*
