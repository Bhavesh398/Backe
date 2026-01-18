"""
Script to diagnose and fix model issues
Run this to check if you need to retrain
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models.image_model import DeepfakeImageModel
from backend.preprocessing.image import load_image

# Improved data augmentation to handle various image qualities
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=10),
    # Simulate compression and low quality
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    ], p=0.3),
    transforms.ToTensor(),
    # Add random noise
    transforms.RandomApply([
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02)
    ], p=0.2),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])


class SimpleDeepfakeDataset(Dataset):
    """
    Expected folder structure:
    data/
        train/
            real/
                img1.jpg
                img2.jpg
            fake/
                img1.jpg
                img2.jpg
    """
    def __init__(self, root_dir, transform=None, use_pil=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_pil = use_pil
        self.samples = []
        
        # Load real images (label 0)
        real_dir = self.root_dir / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 0))
        
        # Load fake images (label 1)
        fake_dir = self.root_dir / "fake"
        if fake_dir.exists():
            for img_path in fake_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 1))
        
        print(f"Loaded {len(self.samples)} images")
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        print(f"Real: {real_count}, Fake: {fake_count}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            if self.transform and self.use_pil:
                # Use PIL for transforms
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            else:
                # Use the standard preprocessing
                image = load_image(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if loading fails
            image = torch.zeros(3, 224, 224)
        
        return image, label


def train_model(data_dir="data/train", epochs=10, lr=0.001, batch_size=16, train_split=0.7):
    """
    Train or retrain the deepfake detection model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets with augmentation
    print("\nCreating training dataset with augmentation...")
    train_full_dataset = SimpleDeepfakeDataset(data_dir, transform=None, use_pil=False)
    
    if len(train_full_dataset) == 0:
        print(f"❌ No data found in {data_dir}")
        print("Please organize your data as:")
        print("  data/train/real/  <- put real images here")
        print("  data/train/fake/  <- put fake images here")
        return
    
    # Split into train/val (80% training, 20% testing)
    train_size = int(0.8 * len(train_full_dataset.samples))
    val_size = len(train_full_dataset.samples) - train_size
    
    # Create separate datasets for train and val
    train_dataset = SimpleDeepfakeDataset(data_dir, transform=train_transform, use_pil=True)
    train_dataset.samples = train_full_dataset.samples[:train_size]
    
    val_dataset = SimpleDeepfakeDataset(data_dir, transform=val_transform, use_pil=True)
    val_dataset.samples = train_full_dataset.samples[train_size:]
    
    print(f"Training samples: {len(train_dataset)} (with augmentation)")
    print(f"Validation samples: {len(val_dataset)} (no augmentation)")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = DeepfakeImageModel()
    
    # Try to load existing weights
    MODEL_PATH = "models/image_deepfake.pth"
    if os.path.exists(MODEL_PATH):
        print(f"\n⚠️  Found existing model at {MODEL_PATH}")
        response = input("Do you want to (1) start fresh or (2) continue training? [1/2]: ")
        if response == "2":
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
                print("✓ Loaded existing weights")
            except:
                print("⚠️  Could not load existing weights, starting fresh")
    
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    patience = 5  # Stop if no improvement for 5 epochs
    patience_counter = 0
    
    print(f"\n{'='*50}")
    print(f"Starting training for {epochs} epochs")
    print(f"Early stopping patience: {patience} epochs")
    print(f"{'='*50}\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
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
                print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} - Acc: {100.*train_correct/train_total:.2f}%")
        
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
        print("   2. Getting more training data")
        print("   3. Using a different model architecture")
        print("   4. Adjusting learning rate or other hyperparameters")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--data_dir", type=str, default="data/train",
                       help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )
