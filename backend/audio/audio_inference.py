import torch
import torchaudio
import os
import tempfile
from backend.audio.audio_model import AudioDeepfakeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioDeepfakeModel().to(device)

# Load model if it exists, otherwise skip
model_path = "models/audio_deepfake.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
else:
    print(f"Warning: Model file {model_path} not found. Audio inference may not work.")

def convert_audio_to_wav(file_path):
    """Convert audio file (MP3, etc.) to WAV format if needed."""
    if file_path.lower().endswith('.mp3'):
        try:
            import librosa
            import soundfile as sf
            
            # Load MP3 with librosa
            y, sr = librosa.load(file_path, sr=None)
            
            # Create temporary WAV file
            temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()
            
            # Save as WAV
            sf.write(temp_wav_path, y, sr)
            return temp_wav_path, True
        except Exception as e:
            print(f"Error converting MP3: {e}")
            return file_path, False
    return file_path, False

def predict_audio(path):
    # Convert MP3 to WAV if needed
    audio_path, is_temp = convert_audio_to_wav(path)
    
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    wav = wav.mean(dim=0)
    wav = wav[:64000]
    wav = torch.nn.functional.pad(wav, (0, 64000 - len(wav)))

    with torch.no_grad():
        out = model(wav.unsqueeze(0).unsqueeze(0).to(device))
        prob = torch.softmax(out, dim=1)[0][1].item()

    # Clean up temporary file if created
    if is_temp and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except:
            pass

    return prob
