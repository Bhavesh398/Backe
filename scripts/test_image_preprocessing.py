import torch
from backend.preprocessing.image import load_image

device = "cuda" if torch.cuda.is_available() else "cpu"

img_tensor = load_image("test.jpg", device=device)

print("Tensor shape:", img_tensor.shape)
print("Device:", img_tensor.device)
