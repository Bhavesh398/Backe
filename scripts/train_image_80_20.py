import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.models.image_model import DeepfakeImageModel
from backend.preprocessing.image import transform, transform_train

# ===================== CONFIG =====================
DATASET_DIR = "datasets/images"  # Flat structure: real/ and fake/
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
PATIENCE = 3  # Early stopping patience
TRAIN_SPLIT = 0.8
MODEL_SAVE_PATH = "models/image_deepfake.pth"
# ==================================================

class FlatImageDataset(Dataset):
    """Dataset that reads from datasets/images/real and datasets/images/fake"""
    def __init__(self, root_dir, is_train=True):
        self.samples = []
        self.is_train = is_train
        self.transform = transform_train if is_train else transform
        
        # Load real images (label 0)
        real_dir = os.path.join(root_dir, "real")
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(real_dir, file), 0))
        
        # Load fake images (label 1)
        fake_dir = os.path.join(root_dir, "fake")
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(fake_dir, file), 1))
        
        print(f"{'Train' if is_train else 'Val'} dataset: {len(self.samples)} samples")
        
        # Count classes
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        print(f"  Real: {real_count}, Fake: {fake_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Invalid image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.transform(img)
        return image, torch.tensor(label, dtype=torch.long)


def get_class_weights(dataset):
    """Calculate class weights for balanced training"""
    labels = [label for _, label in dataset.samples]
    class_counts = torch.bincount(torch.tensor(labels))
    total = len(labels)
    weights = total / (len(class_counts) * class_counts.float())
    return weights


def get_weighted_sampler(dataset):
    """Create weighted sampler for balanced batches"""
    labels = [label for _, label in dataset.samples]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", unit="batch")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            real_mask = labels == 0
            fake_mask = labels == 1
            
            real_total += real_mask.sum().item()
            fake_total += fake_mask.sum().item()
            
            real_correct += (predicted[real_mask] == labels[real_mask]).sum().item()
            fake_correct += (predicted[fake_mask] == labels[fake_mask]).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    real_acc = 100. * real_correct / real_total if real_total > 0 else 0
    fake_acc = 100. * fake_correct / fake_total if fake_total > 0 else 0
    
    return val_loss, val_acc, real_acc, fake_acc


def main():
    print("=" * 60)
    print("🚀 Training Image Deepfake Detection Model")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load full dataset
    print(f"\n📂 Loading dataset from: {DATASET_DIR}")
    full_dataset = FlatImageDataset(DATASET_DIR, is_train=True)
    
    # Split into train/val (80/20)
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset to use validation transforms
    val_dataset.dataset.is_train = False
    val_dataset.dataset.transform = transform
    
    print(f"\n📊 Split: Train={train_size}, Val={val_size}")
    
    # Calculate class weights from full dataset
    class_weights = get_class_weights(full_dataset).to(device)
    print(f"Class weights: Real={class_weights[0]:.4f}, Fake={class_weights[1]:.4f}")
    
    # Create weighted sampler ONLY for training subset
    train_labels = [full_dataset.samples[idx][1] for idx in train_dataset.indices]
    class_counts = torch.bincount(torch.tensor(train_labels))
    class_weights_train = 1.0 / class_counts.float()
    sample_weights = [class_weights_train[train_labels[i]] for i in range(len(train_labels))]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model
    print("\n🔧 Initializing model...")
    model = DeepfakeImageModel().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop with early stopping
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n🎯 Training for {EPOCHS} epochs (Early stopping patience: {PATIENCE})")
    print("=" * 60)
    
    for epoch in range(EPOCHS):
        print(f"\n📍 Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, real_acc, fake_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Print summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"   Per-class: Real={real_acc:.2f}%, Fake={fake_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'real_accuracy': real_acc,
                'fake_accuracy': fake_acc,
            }
            torch.save(checkpoint, MODEL_SAVE_PATH)
            print(f"   ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{PATIENCE})")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
            print(f"   Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print(f"💾 Best model saved to: {MODEL_SAVE_PATH}")
    print(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
