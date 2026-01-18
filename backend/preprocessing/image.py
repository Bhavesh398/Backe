import cv2
import torch
from torchvision import transforms

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Standard transform for inference
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Augmented transform for training (lightweight for speed)
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

def preprocess_frame(frame):
    """
    Preprocess a single video frame (NumPy array)
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return transform(frame)

def load_image(path: str):
    """
    Load image from disk and preprocess
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Invalid image path: {path}")
    return preprocess_frame(img)
