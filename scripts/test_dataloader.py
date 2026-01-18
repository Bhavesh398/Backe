import torch
from torch.utils.data import DataLoader
from backend.datasets.image_dataset import ImageDeepfakeDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = ImageDeepfakeDataset(
    root_dir="datasets/images",
    device=device
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for images, labels in loader:
    print("Batch shape:", images.shape)
    print("Labels:", labels)
    break
