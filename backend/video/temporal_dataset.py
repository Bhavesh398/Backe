import os
import cv2
import torch
from torch.utils.data import Dataset
from backend.preprocessing.image import preprocess_frame


class VideoTemporalDataset(Dataset):
    """
    Loads video clips and returns:
    - clip tensor: (T, 3, 224, 224)
    - label: 0 (REAL) or 1 (FAKE)
    """

    def __init__(self, root_dir, frames_per_clip=16):
        self.root_dir = os.path.abspath(root_dir)
        self.frames_per_clip = frames_per_clip
        self.samples = []

        # Collect video paths
        for label, cls in enumerate(["real", "fake"]):
            cls_dir = os.path.join(self.root_dir, cls)

            if not os.path.exists(cls_dir):
                raise FileNotFoundError(f"Missing directory: {cls_dir}")

            for file in os.listdir(cls_dir):
                if file.lower().endswith(".mp4"):
                    self.samples.append(
                        (os.path.join(cls_dir, file), label)
                    )

        if len(self.samples) == 0:
            raise RuntimeError("No video files found in dataset")

    def __len__(self):
        return len(self.samples)

    def _read_clip(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        frames = []

        while len(frames) < self.frames_per_clip:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        if len(frames) < self.frames_per_clip:
            return None

        # Preprocess frames (NO disk I/O)
        frames = [
            preprocess_frame(frame)
            for frame in frames
        ]

        # Shape: (T, 3, 224, 224)
        return torch.stack(frames)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        clip = self._read_clip(video_path)

        # Skip short/broken videos safely
        if clip is None:
            return self.__getitem__((idx + 1) % len(self.samples))

        return clip, torch.tensor(label, dtype=torch.long)
