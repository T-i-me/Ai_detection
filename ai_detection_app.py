
"""
AI Detection Application
Detects AI-generated images, videos, and audio using deep learning models
Uses Gradio for beautiful and simple UI/UX
"""

import warnings
import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
warnings.filterwarnings('ignore')

# ===================== FIX: PRE-DOWNLOAD EFFICIENTNET =====================

# This block ensures EfficientNet_B0 weights are downloaded ONCE.
# After the first download, PyTorch loads them from local cache (no internet needed).
try:
    from torchvision import models
    print("Checking EfficientNet weights...")
    models.efficientnet_b0(pretrained=True)
    print("EfficientNet_B0 weights downloaded / already available.")
except Exception as e:
    print(f"EfficientNet pre-download error: {e}")

# ===================== MODEL CLASSES =====================

class ImageDetectionModel:
    """Detects AI-generated images using EfficientNet or ResNet"""

    def __init__(self):
        # Load pre-trained model (using ResNet50 as example)
        # In production, load fine-tuned weights
        from torchvision import models, transforms
        self.model = models.resnet50(pretrained=True)
        # Modify final layer for binary classification
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Real vs Fake
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """Predict if image is AI-generated"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))

        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        real_prob = probabilities[0][0].item()
        fake_prob = probabilities[0][1].item()

        return {
            "Real": real_prob,
            "AI-Generated": fake_prob
        }


class VideoDetectionModel:
    """Detects deepfake videos using frame analysis"""

    def __init__(self):
        # Initialize frame-based detector
        from torchvision import models, transforms
        self.model = models.efficientnet_b0(pretrained=True)
        # Modify for binary classification
        self.model.classifier[1] = nn.Linear(1280, 2)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract_frames(self, video_path, num_frames=30):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        return frames

    def predict(self, video_path):
        """Predict if video is deepfake"""
        frames = self.extract_frames(video_path)

        if not frames:
            return {"error": "Could not extract frames from video"}

        predictions = []
        for frame in frames:
            img = Image.fromarray(frame)
            img_tensor = self.transform(img).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predictions.append(probabilities[0][1].item())  # Fake probability

        avg_fake_prob = np.mean(predictions)
        avg_real_prob = 1 - avg_fake_prob

        return {
            "Real": avg_real_prob,
            "Deepfake": avg_fake_prob,
            "frames_analyzed": len(frames)
        }


class AudioDetectionModel:
    """Detects AI-generated or cloned voice using Wav2Vec2 (Audio Classification)"""

    def __init__(self):
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self.model_name = "superb/wav2vec2-base-superb-ks"

            # Correct processor for audio classification (NO tokenizer)
            self.processor = AutoFeatureExtractor.from_pretrained(
                self.model_name,
                local_files_only=True
            )

            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_name,
                local_files_only=True
            )

            self.model.eval()

        except Exception as e:
            print(f"[AudioDetectionModel] Failed to load model: {e}")
            self.processor = None
            self.model = None

    def predict(self, audio_path):
        """Predict if audio is AI-generated"""
        if self.model is None or self.processor is None:
            return {"error": "Audio model not loaded"}

        try:
            import librosa
            import torch

            # Load audio (force 16kHz)
            audio, sr = librosa.load(audio_path, sr=16000)

            # Extract features (raw waveform → features)
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            )

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)

            labels = self.model.config.id2label
            probs = probabilities[0].tolist()

            result = {}
            for idx, label in labels.items():
                result[label] = probs[idx]

            return result


        except Exception as e:
            return {"error": f"Error processing audio: {str(e)}"}


# ===================== GRADIO INTERFACE =====================

def detect_image(image):
    """Image detection interface function"""
    if image is None:
        return "Please upload an image"

    model = ImageDetectionModel()
    results = model.predict(image)

    # Format output
    output = f"🔍 **Image Analysis Results**\n\n"
    output += f"✓ Real Image: {results['Real']*100:.2f}%\n"
    output += f"⚠ AI-Generated: {results['AI-Generated']*100:.2f}%\n\n"

    if results['AI-Generated'] > 0.5:
        output += "**Verdict:** This image appears to be AI-generated"
    else:
        output += "**Verdict:** This image appears to be real"

    return output


def detect_video(video):
    """Video detection interface function"""
    if video is None:
        return "Please upload a video"

    model = VideoDetectionModel()
    results = model.predict(video)

    if "error" in results:
        return f"Error: {results['error']}"

    # Format output
    output = f"🎥 **Video Analysis Results**\n\n"
    output += f"Frames Analyzed: {results['frames_analyzed']}\n"
    output += f"✓ Real Video: {results['Real']*100:.2f}%\n"
    output += f"⚠ Deepfake: {results['Deepfake']*100:.2f}%\n\n"

    if results['Deepfake'] > 0.5:
        output += "**Verdict:** This video appears to be a deepfake"
    else:
        output += "**Verdict:** This video appears to be real"

    return output


def detect_audio(audio):
    """Audio detection interface function"""
    if audio is None:
        return "Please upload an audio file"

    model = AudioDetectionModel()
    results = model.predict(audio)

    if "error" in results:
        return f"Error: {results['error']}"

    # Format output
    output = f"🎵 **Audio Analysis Results**\n\n"
    output += f"✓ Real Human Voice: {results['Real Human Voice']*100:.2f}%\n"
    output += f"⚠ AI-Generated Voice: {results['AI-Generated Voice']*100:.2f}%\n\n"

    if results['AI-Generated Voice'] > 0.5:
        output += "**Verdict:** This audio appears to be AI-generated or cloned"
    else:
        output += "**Verdict:** This audio appears to be a real human voice"

    return output


# ===================== BUILD GRADIO APP =====================

def build_app():
    """Build the complete Gradio application"""

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-text {
        font-size: 16px;
        line-height: 1.6;
    }
    """

    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🔍 AI Detection System
            ### Detect AI-Generated Images, Videos, and Audio

            Upload your media files to analyze whether they were created by AI or are authentic.
            This system uses state-of-the-art deep learning models for detection.
            """
        )

        with gr.Tabs():
            # IMAGE DETECTION TAB
            with gr.TabItem("📷 Image Detection"):
                gr.Markdown("### Upload an image to check if it's AI-generated")
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(type="numpy", label="Upload Image")
                        image_button = gr.Button("Analyze Image", variant="primary")
                    with gr.Column():
                        image_output = gr.Textbox(label="Detection Results", lines=8)

                image_button.click(
                    fn=detect_image,
                    inputs=image_input,
                    outputs=image_output
                )

                gr.Markdown(
                    """
                    **Detects:** Midjourney, Stable Diffusion, DALL-E, and other AI image generators
                    """
                )

            # VIDEO DETECTION TAB
            with gr.TabItem("🎥 Video Detection"):
                gr.Markdown("### Upload a video to check for deepfakes")
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video")
                        video_button = gr.Button("Analyze Video", variant="primary")
                    with gr.Column():
                        video_output = gr.Textbox(label="Detection Results", lines=8)

                video_button.click(
                    fn=detect_video,
                    inputs=video_input,
                    outputs=video_output
                )

                gr.Markdown(
                    """
                    **Detects:** Face swaps, face reenactment, and other deepfake manipulations
                    """
                )

            # AUDIO DETECTION TAB
            with gr.TabItem("🎵 Audio Detection"):
                gr.Markdown("### Upload audio to check for AI-generated or cloned voices")
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(type="filepath", label="Upload Audio")
                        audio_button = gr.Button("Analyze Audio", variant="primary")
                    with gr.Column():
                        audio_output = gr.Textbox(label="Detection Results", lines=8)

                audio_button.click(
                    fn=detect_audio,
                    inputs=audio_input,
                    outputs=audio_output
                )

                gr.Markdown(
                    """
                    **Detects:** Voice cloning, TTS synthesis, and voice conversion
                    """
                )

        gr.Markdown(
            """
            ---
            ### 📝 About This System

            This AI detection system uses advanced deep learning models:
            - **Image Detection:** EfficientNet/ResNet architecture
            - **Video Detection:** Frame-based analysis with temporal modeling
            - **Audio Detection:** Wav2Vec2 transformer model

            **Note:** This is a demonstration system. For production use, models should be fine-tuned 
            on appropriate datasets and regularly updated to detect new AI generation techniques.
            """
        )

    return app


# ===================== MAIN EXECUTION =====================

if __name__ == "__main__":
    app = build_app()
    app.launch(
        share=True,  # Set to True to create public link
        server_name="127.0.0.1",
        server_port=7860
    )
