import torch.nn as nn
from torchvision import models

class DeepfakeImageModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )

        # 🔒 Freeze early layers but unfreeze later layers for fine-tuning
        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last few blocks of features for fine-tuning
        # EfficientNet has 9 blocks (0-8), unfreeze blocks 6, 7, 8
        for i in range(6, 9):
            for param in self.model.features[i].parameters():
                param.requires_grad = True

        # Replace classifier with dropout for regularization
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

        # 🔓 Classifier is always trainable
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
