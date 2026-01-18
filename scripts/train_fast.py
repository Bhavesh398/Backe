import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from backend.models.image_model import DeepfakeImageModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import random
from tqdm import tqdm
import cv2
from backend.preprocessing.image import transform, transform_train

class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, use_augmentation=True):
        self.samples = []
        self.transform = transform_train if use_augmentation else transform
        
        for label, cls in enumerate(["real", "fake"]):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.exists(cls_path):
                raise FileNotFoundError(f"Missing folder: {cls_path}")
            
            for file in os.listdir(cls_path):
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(cls_path, file), label))
    
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

def main():
    # ---- CONFIG ----
    EPOCHS = 20
    BATCH_SIZE = 32
    LR = 1e-4
    DATASET_PATH = "datasets/images"
    MODEL_SAVE_PATH = "models/image_deepfake.pth"
    TRAIN_SPLIT = 0.8  # 80% train, 20% val
    # ----------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load full dataset
    print("Loading dataset...")
    full_dataset = SimpleImageDataset(DATASET_PATH, use_augmentation=False)
    
    # Create 80/20 split
    total_size = len(full_dataset)
    train_size = int(total_size * TRAIN_SPLIT)
    val_size = total_size - train_size
    
    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets with appropriate augmentation
    train_dataset = SimpleImageDataset(DATASET_PATH, use_augmentation=True)
    train_dataset = Subset(train_dataset, train_indices)
    
    val_dataset = SimpleImageDataset(DATASET_PATH, use_augmentation=False)
    val_dataset = Subset(val_dataset, val_indices)

    print(f"Dataset -> Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    print("Initializing model...")
    model = DeepfakeImageModel()
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_acc = 0.0
    patience_counter = 0

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]", unit="batch")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_acc = (train_correct / train_total) * 100
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_correct_real = 0
        val_correct_fake = 0
        val_total_real = 0
        val_total_fake = 0
        
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VAL] ", unit="batch")
        with torch.no_grad():
            for images, labels in pbar_val:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                real_mask = labels == 0
                fake_mask = labels == 1
                val_correct_real += (preds[real_mask] == labels[real_mask]).sum().item()
                val_correct_fake += (preds[fake_mask] == labels[fake_mask]).sum().item()
                val_total_real += real_mask.sum().item()
                val_total_fake += fake_mask.sum().item()
        
        val_acc = (val_correct / val_total) * 100
        val_acc_real = (val_correct_real / val_total_real * 100) if val_total_real > 0 else 0
        val_acc_fake = (val_correct_fake / val_total_fake * 100) if val_total_fake > 0 else 0
        
        print(f"\n📊 Epoch [{epoch+1}/{EPOCHS}]")
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}% | Real: {val_acc_real:.2f}%, Fake: {val_acc_fake:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'val_acc_real': val_acc_real,
                'val_acc_fake': val_acc_fake,
            }, MODEL_SAVE_PATH)
            print(f"   ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   ⏱️  No improvement ({patience_counter}/3)")
            if patience_counter >= 3:
                print(f"\n⛔ Early stopping at epoch {epoch+1}")
                break

    print("\n" + "="*80)
    print(f"✅ TRAINING COMPLETE! Best Val Acc: {best_val_acc:.2f}%")
    print(f"📁 Model saved to: {MODEL_SAVE_PATH}")
    print("="*80)

if __name__ == '__main__':
    main()
