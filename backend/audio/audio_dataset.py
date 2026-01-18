import os
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F
import tempfile

class AudioDeepfakeDataset(Dataset):
    def __init__(self, root, sample_rate=16000, max_seconds=4):
        self.samples = []
        self.sr = sample_rate
        self.max_len = sample_rate * max_seconds

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = os.path.join(root, cls)
            for f in os.listdir(cls_dir):
                if f.endswith(".wav") or f.endswith(".mp3"):
                    self.samples.append((os.path.join(cls_dir, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # Convert MP3 to WAV if needed
        if path.lower().endswith('.mp3'):
            try:
                import librosa
                import soundfile as sf
                
                # Load MP3 with librosa
                y, sr = librosa.load(path, sr=None)
                
                # Create temporary WAV file
                temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_wav_path = temp_wav.name
                temp_wav.close()
                
                # Save as WAV
                sf.write(temp_wav_path, y, sr)
                path = temp_wav_path
                is_temp = True
            except Exception as e:
                print(f"Error converting MP3: {e}")
                is_temp = False
        else:
            is_temp = False
        
        wav, sr = torchaudio.load(path)

        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        wav = wav.mean(dim=0)  # mono

        if len(wav) > self.max_len:
            wav = wav[:self.max_len]
        else:
            wav = F.pad(wav, (0, self.max_len - len(wav)))
        
        # Clean up temporary file if created
        if is_temp and os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

        return wav.unsqueeze(0), torch.tensor(label)
