# 🎤 Streamlit Audio Digit Classifier

An interactive web interface for the Audio Digit Classification System built with Streamlit.

## 🚀 Quick Start

### Option 1: Use the Launcher Script
```bash
python launch_streamlit.py
```

### Option 2: Run Streamlit Directly
```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## 📱 Features

### 🏠 Home Page
- **Upload Audio Files**: Test your own WAV/MP3/FLAC files
- **Demo Files**: Try pre-loaded sample digits
- **Real-time Results**: Instant classification with confidence scores
- **Audio Visualization**: Interactive waveform plots

### 🧪 Test Audio Files
- **Batch Testing**: Test all demo files at once
- **Multiple File Upload**: Process several files simultaneously  
- **Detailed Results**: Accuracy metrics and processing times
- **Interactive Charts**: Visualize confidence scores and performance

### 📊 Model Analytics
- **Feature Importance**: See which audio features matter most
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Speed Analysis**: Processing time and throughput metrics
- **Model Architecture**: Technical details and design decisions

### 📖 Documentation
- **Quick Start Guide**: How to use the interface
- **Technical Details**: Feature engineering and ML explanations
- **Troubleshooting**: Common issues and solutions
- **Performance Tips**: Getting the best results

### ⚙️ System Information
- **Requirements**: Software and hardware needs
- **File Status**: Check if all components are present
- **Performance Benchmarks**: System performance metrics
- **About**: Project details and technology stack

## 🎯 How to Use

1. **Start the App**: Run `python launch_streamlit.py`
2. **Upload Audio**: Use the file uploader on the Home page
3. **Get Results**: See prediction, confidence, and processing time
4. **Explore**: Try different pages for detailed analysis
5. **Learn**: Read documentation for technical insights

## 📊 Expected Results

- **Accuracy**: 97% on test data
- **Processing Speed**: ~60ms per audio file
- **Supported Formats**: WAV, MP3, FLAC
- **Audio Length**: 0.3-2.0 seconds recommended

## 🔧 Audio Requirements

For best results, your audio should:
- Contain a single spoken digit (0-9)
- Be clear with minimal background noise
- Have good audio quality
- Be spoken in English
- Last between 0.3-2.0 seconds

## 🐛 Troubleshooting

### Common Issues

**"Model files not found"**
- Run `python train_model.py` first to create the model files

**"Demo files missing"**
- Run `python create_demos.py` to generate sample files

**"Low accuracy on my audio"**
- Ensure clear pronunciation
- Minimize background noise
- Check audio quality and volume

**"App won't start"**
- Check that all requirements are installed: `pip install -r requirements.txt`
- Try running manually: `streamlit run streamlit_app.py`

## 🛠️ Development

### Project Structure
```
├── streamlit_app.py          # Main Streamlit application
├── launch_streamlit.py       # App launcher script
├── create_demos.py          # Demo file generator
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── demo_audio/              # Sample audio files
    ├── digit_0_demo.wav
    ├── digit_1_demo.wav
    └── ...
```

### Key Dependencies
- **streamlit**: Web interface framework
- **plotly**: Interactive visualizations
- **librosa**: Audio processing
- **scikit-learn**: Machine learning model
- **pandas**: Data manipulation

### Customization

You can customize the app by modifying:
- `streamlit_app.py`: Main application logic
- `.streamlit/config.toml`: UI theme and settings
- CSS styles in the app for custom appearance

## 📈 Performance

The Streamlit app is optimized for:
- **Fast Loading**: Model caching for quick startup
- **Responsive UI**: Real-time feedback and progress bars
- **Memory Efficiency**: Optimized data handling
- **Scalability**: Can handle multiple users (with proper deployment)

## 🚀 Deployment

For production deployment, consider:

### Local Network
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

### Cloud Deployment
- **Streamlit Cloud**: Free hosting for public repos
- **Heroku**: Easy deployment with git integration
- **AWS/GCP/Azure**: Full cloud solutions
- **Docker**: Containerized deployment

## 💡 Tips for Best Experience

1. **Use Chrome/Firefox**: Best browser compatibility
2. **Upload WAV files**: Fastest processing and best quality
3. **Keep audio short**: 0.5-1.0 seconds is optimal
4. **Clear pronunciation**: Speak digits clearly and distinctly
5. **Minimize noise**: Use quiet environment for recording

## 🔮 Future Enhancements

Planned features:
- **Real-time microphone input**: Live audio capture in browser
- **Batch processing**: Upload multiple files at once
- **Audio recording**: Record directly in the app
- **Model comparison**: Try different ML algorithms
- **Export results**: Download prediction reports

## 📞 Support

If you encounter issues:
1. Check the **System Info** page for file status
2. Review the **Documentation** page for help
3. Try the **troubleshooting** steps above
4. Ensure all requirements are installed correctly

---

*Built with ❤️ using Streamlit, librosa, and scikit-learn*
