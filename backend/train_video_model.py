"""
Training script for deepfake detection model using video datasets
This will extract frames from videos and train the image model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
import sys
import cv2
from pathlib import Path
from PIL import Image
import random
import torchvision.transforms as transforms
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.image_model import DeepfakeImageModel
from backend.preprocessing.image import preprocess_frame

# Data augmentation to handle low quality images
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=5),
    # Simulate compression artifacts and low quality
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    ], p=0.3),
    transforms.ToTensor(),
    transforms.RandomApply([
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02)  # Add noise
    ], p=0.2),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])


class VideoFrameDataset(Dataset):
    """
    Dataset that extracts frames from videos for training
    Expected folder structure:
    datasets/videos/
        real/
            video1.mp4
            video2.mp4
        fake/
            video1.mp4
            video2.mp4
    """
    def __init__(self, root_dir, frames_per_video=5, use_augmentation=True):
        self.root_dir = Path(root_dir)
        self.frames_per_video = frames_per_video
        self.use_augmentation = use_augmentation
        self.video_paths = []
        
        # Load real videos (label 0)
        real_dir = self.root_dir / "real"
        if real_dir.exists():
            for video_path in real_dir.glob("*"):
                if video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    self.video_paths.append((str(video_path), 0))
        
        # Load fake videos (label 1)
        fake_dir = self.root_dir / "fake"
        if fake_dir.exists():
            for video_path in fake_dir.glob("*"):
                if video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    self.video_paths.append((str(video_path), 1))
        
        # Shuffle to mix real and fake
        random.shuffle(self.video_paths)
        
        print(f"Loaded {len(self.video_paths)} videos")
        real_count = sum(1 for _, label in self.video_paths if label == 0)
        fake_count = sum(1 for _, label in self.video_paths if label == 1)
        print(f"Real: {real_count}, Fake: {fake_count}")
        
    def __len__(self):
        return len(self.video_paths)
    
    def extract_frames(self, video_path, num_frames=5):
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Get frame indices evenly spaced
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def __getitem__(self, idx):
        video_path, label = self.video_paths[idx]
        
        try:
            # Extract frames from video
            frames = self.extract_frames(video_path, self.frames_per_video)
            
            if len(frames) == 0:
                # Return a black frame if extraction fails
                return torch.zeros(3, 224, 224), label
            
            # Use a random frame from the video for training
            frame = random.choice(frames)
            
            # Apply augmentation for training
            if self.use_augmentation:
                frame_tensor = train_transform(frame)
            else:
                frame_tensor = preprocess_frame(frame)
            
            return frame_tensor, label
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return torch.zeros(3, 224, 224), label


def train_model(
    data_dir="datasets/videos",
    epochs=20,
    lr=0.0001,
    batch_size=16,
    train_split=0.8
):
    """
    Train the deepfake detection model on video data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    print("\n" + "="*50)
    print("Loading video dataset with augmentation...")
    print("="*50)
    full_dataset = VideoFrameDataset(data_dir, frames_per_video=5, use_augmentation=False)
    
    if len(full_dataset) == 0:
        print(f"❌ No data found in {data_dir}")
        print("Please organize your data as:")
        print("  datasets/videos/real/  <- put real videos here")
        print("  datasets/videos/fake/  <- put fake videos here")
        return
    
    # Split into train/val (70% training, 30% testing)
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create separate datasets with and without augmentation
    train_dataset = VideoFrameDataset(data_dir, frames_per_video=5, use_augmentation=True)
    train_dataset.video_paths = [full_dataset.video_paths[i] for i in train_indices.indices]
    
    val_dataset = VideoFrameDataset(data_dir, frames_per_video=5, use_augmentation=False)
    val_dataset.video_paths = [full_dataset.video_paths[i] for i in val_indices.indices]
    
    print(f"\nTrain samples: {len(train_dataset)} (with augmentation)")
    print(f"Validation samples: {len(val_dataset)} (no augmentation)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues on Windows
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    model = DeepfakeImageModel()
    
    # Try to load existing weights
    MODEL_PATH = "models/image_deepfake.pth"
    if os.path.exists(MODEL_PATH):
        print(f"\n⚠️  Found existing model at {MODEL_PATH}")
        response = input("Do you want to (1) start fresh or (2) continue training? [1/2]: ")
        if response == "2":
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("✓ Loaded existing weights")
                else:
                    model.load_state_dict(checkpoint)
                    print("✓ Loaded existing weights")
            except Exception as e:
                print(f"⚠️  Could not load existing weights: {e}, starting fresh")
    
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Only train the classifier, freeze the feature extractor
    optimizer = optim.Adam(model.model.classifier.parameters(), lr=lr)
    
    best_val_acc = 0.0
    patience = 5  # Stop if no improvement for 5 epochs
    patience_counter = 0
    
    print(f"\n{'='*50}")
    print(f"Starting training for {epochs} epochs")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Early stopping patience: {patience} epochs")
    print(f"{'='*50}\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                acc = 100. * train_correct / train_total
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} - Acc: {acc:.2f}%")
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"{'='*50}\n")
        
        # Save best model and check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # Reset patience counter
            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, MODEL_PATH)
            print(f"✓ Saved best model with val_acc: {val_acc:.2f}%\n")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}\n")
            
            if patience_counter >= patience:
                print(f"\n{'='*50}")
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"No improvement for {patience} consecutive epochs")
                print(f"{'='*50}\n")
                break
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"{'='*50}")
    
    if best_val_acc < 70:
        print("\n⚠️  WARNING: Validation accuracy is below 70%")
        print("   Consider:")
        print("   1. Training for more epochs")
        print("   2. Adjusting learning rate")
        print("   3. Using more frames per video")
    else:
        print("\n✓ Model training successful!")
        print("  You can now use this model for predictions.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train deepfake detection model on video data")
    parser.add_argument("--data_dir", type=str, default="datasets/videos",
                       help="Path to video data directory")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--train_split", type=float, default=0.7,
                       help="Training data split ratio (default: 0.7 = 70% train, 30% test)")
    
    args = parser.parse_args()
    
    print("="*50)
    print("Video-Based Deepfake Detection Training")
    print("="*50)
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Train/Val split: {args.train_split}/{1-args.train_split}")
    print("="*50)
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        train_split=args.train_split
    )
