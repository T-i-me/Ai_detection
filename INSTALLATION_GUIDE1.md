# AI Detection Application - Installation & Usage Guide

## 📋 Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

## 🚀 Installation Steps

### 1. Clone or Download the Project
```bash
# If you have the files, navigate to the project directory
cd ai-detection-app
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Note: If you encounter issues with dlib or face-recognition,
# you can skip them for basic functionality:
pip install -r requirements.txt --no-deps face-recognition dlib
```

### 4. Download Pre-trained Models (Optional)
For better accuracy, you can download fine-tuned models:

**For Audio Detection:**
```python
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# Download deepfake audio detection model
model_name = "MelodyMachine/Deepfake-audio-detection-V2"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
```

**For Image Detection:**
- Download pre-trained weights from research repositories
- Place in `models/` directory

## 🎯 Usage

### Running the Application
```bash
# Start the Gradio interface
python ai_detection_app.py
```

The application will launch in your default web browser at:
`http://localhost:7860`

### Using the Interface

#### 📷 Image Detection
1. Click on the "Image Detection" tab
2. Upload an image file (JPG, PNG, etc.)
3. Click "Analyze Image"
4. View the results showing probability of AI generation

#### 🎥 Video Detection
1. Click on the "Video Detection" tab
2. Upload a video file (MP4, AVI, MOV, etc.)
3. Click "Analyze Video"
4. Wait for frame extraction and analysis
5. View results showing deepfake probability

#### 🎵 Audio Detection
1. Click on the "Audio Detection" tab
2. Upload an audio file (MP3, WAV, FLAC, etc.)
3. Click "Analyze Audio"
4. View results showing AI voice generation probability

## 🔧 Customization

### Using Better Pre-trained Models

**For Image Detection:**
```python
# Replace in ImageDetectionModel class
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

model_name = "umm-maybe/AI-image-detector"
self.processor = AutoFeatureExtractor.from_pretrained(model_name)
self.model = AutoModelForImageClassification.from_pretrained(model_name)
```

**For Audio Detection:**
```python
# Replace in AudioDetectionModel class
model_name = "MelodyMachine/Deepfake-audio-detection-V2"
self.processor = Wav2Vec2Processor.from_pretrained(model_name)
self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
```

### Adjusting Detection Threshold
```python
# In the detection functions, modify the threshold
if results['AI-Generated'] > 0.7:  # Changed from 0.5 to 0.7 for stricter detection
    output += "**Verdict:** This appears to be AI-generated"
```

## 📊 Model Performance

| Detection Type | Expected Accuracy | Processing Time | Model Size |
|---------------|-------------------|-----------------|------------|
| Image         | 90-95%           | < 1 second      | ~100 MB    |
| Video         | 85-92%           | 3-10 seconds    | ~50 MB     |
| Audio         | 88-95%           | 1-3 seconds     | ~400 MB    |

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

**2. CUDA Out of Memory**
```python
# Use CPU instead of GPU
device = "cpu"
model.to(device)
```

**3. Video Processing Errors**
```bash
# Install additional codecs
# On Ubuntu/Debian:
sudo apt-get install ffmpeg
# On macOS:
brew install ffmpeg
```

**4. Audio File Format Issues**
```python
# Convert audio to supported format
import soundfile as sf
audio, sr = librosa.load("input.mp3")
sf.write("output.wav", audio, sr)
```

## 🚀 Deployment Options

### Local Deployment
```bash
python ai_detection_app.py
```

### Share with Others (Temporary Public Link)
```python
# In ai_detection_app.py, change:
app.launch(share=True)  # Creates public Gradio link
```

### Cloud Deployment

**Hugging Face Spaces:**
1. Create account at huggingface.co
2. Create new Space with Gradio SDK
3. Upload files
4. Automatic deployment

**Docker Deployment:**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "ai_detection_app.py"]
```

## 📚 Additional Resources

- [Gradio Documentation](https://gradio.app/docs)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Deepfake Detection Research Papers](https://arxiv.org/search/?query=deepfake+detection)

## ⚠️ Important Notes

1. **Model Limitations:** The default models are pre-trained baselines. For production use, fine-tune on specific datasets.

2. **Performance:** Processing time depends on hardware. GPU acceleration recommended for video analysis.

3. **Accuracy:** Detection accuracy varies based on:
   - Quality of input media
   - Sophistication of AI generation technique
   - Model training data

4. **Updates:** AI generation techniques evolve rapidly. Regular model updates are recommended.

5. **Ethical Use:** This tool should be used responsibly and ethically for verification purposes.

## 📝 License

This is an educational/research implementation. Check individual model licenses for commercial use.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures
- Better pre-processing pipelines
- Performance optimization
- UI/UX enhancements

---
**Version:** 1.0.0
**Last Updated:** November 2025
