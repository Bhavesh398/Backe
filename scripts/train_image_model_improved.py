import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backend.models.image_model import DeepfakeImageModel
from backend.datasets.image_dataset import ImageDeepfakeDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

def main():
    # ---------------- CONFIG ----------------
    EPOCHS = 10
    BATCH_SIZE = 16
    LR = 1e-4
    DATASET_PATH = "datasets/images"
    MODEL_SAVE_PATH = "models/image_deepfake.pth"
    # ----------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Fixed splits: Train / Validation folders
    train_dataset = ImageDeepfakeDataset(DATASET_PATH, split="train")
    val_dataset = ImageDeepfakeDataset(DATASET_PATH, split="validation")

    # Compute class weights from TRAIN split only
    labels_all = [lbl for _, lbl in train_dataset.samples]
    num_real = sum(1 for l in labels_all if l == 0)
    num_fake = sum(1 for l in labels_all if l == 1)
    class_weights = torch.tensor([1.0 / num_real, 1.0 / num_fake], device=device)
    print(f"Train class counts -> Real: {num_real}, Fake: {num_fake}")

    # Weighted sampler to balance classes in training
    sample_weights = [class_weights[label].item() for _, label in train_dataset.samples]
    train_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    print(f"Dataset split -> Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # DataLoaders (train uses weighted sampler for balance, no multiprocessing on Windows)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


    # Model (from scratch)
    model = DeepfakeImageModel()
    model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Training Loop
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 5

    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
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

        train_acc = (train_correct / train_total) * 100
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Track per-class accuracy
        val_correct_real = 0
        val_correct_fake = 0
        val_total_real = 0
        val_total_fake = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                # Per-class accuracy
                real_mask = labels == 0
                fake_mask = labels == 1
                val_correct_real += (preds[real_mask] == labels[real_mask]).sum().item()
                val_correct_fake += (preds[fake_mask] == labels[fake_mask]).sum().item()
                val_total_real += real_mask.sum().item()
                val_total_fake += fake_mask.sum().item()
        
        val_acc = (val_correct / val_total) * 100
        val_acc_real = (val_correct_real / val_total_real * 100) if val_total_real > 0 else 0
        val_acc_fake = (val_correct_fake / val_total_fake * 100) if val_total_fake > 0 else 0
        
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}% (Real: {val_acc_real:.2f}%, Fake: {val_acc_fake:.2f}%)")
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'val_acc_real': val_acc_real,
                'val_acc_fake': val_acc_fake,
                'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_SAVE_PATH)
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    print(f"\n✅ Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
