"""EfficientNet-B4 with gender fusion for bone age regression."""

import torch
import torch.nn as nn
import timm
import config


class BoneAgeModel(nn.Module):
    """
    Architecture:
        Image → EfficientNet-B4 (pretrained) → GlobalAvgPool → 1792-d
        Gender (1-d) ──────────────────────────────────────────┘
                                    ↓  concat → 1793-d
                        FC(1793→512) → BN → ReLU → Dropout
                        FC(512→1) → bone age prediction (months)
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        pretrained: bool = config.PRETRAINED,
        hidden_dim: int = config.HIDDEN_DIM,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()

        # Backbone: EfficientNet-B4 from timm, remove classifier head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features  # 1792 for efficientnet_b4

        # Regression head with gender fusion
        self.head = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),  # +1 for gender
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image:  (B, 3, H, W) tensor
            gender: (B, 1) tensor (1=male, 0=female)
        Returns:
            (B, 1) predicted bone age in months
        """
        features = self.backbone(image)          # (B, 1792)
        fused = torch.cat([features, gender], dim=1)  # (B, 1793)
        return self.head(fused)                  # (B, 1)

    def freeze_backbone(self):
        """Freeze all backbone parameters (for warmup phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters (for fine-tuning phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
