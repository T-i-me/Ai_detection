from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

model_name = "superb/wav2vec2-base-superb-ks"

print("Downloading processor...")
AutoFeatureExtractor.from_pretrained(model_name)

print("Downloading model...")
AutoModelForAudioClassification.from_pretrained(model_name)

print("✅ Audio model downloaded and cached successfully")
