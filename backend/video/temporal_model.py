import torch
import torch.nn as nn
from torchvision import models

class CNNLSTMDeepfake(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()

        backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=1792,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x: (B, T, 3, 224, 224)
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x)
        feats = self.pool(feats).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, -1)

        lstm_out, _ = self.lstm(feats)
        video_feat = lstm_out[:, -1, :]  # last timestep

        return self.classifier(video_feat)
