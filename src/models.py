"""Model definitions for MAGE-Vein."""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet161_Weights, DenseNet201_Weights


class MAGEVein(nn.Module):
    """MAGE-Vein multi-task model for grouped finger-vein input.

    A DenseNet-161 backbone extracts one feature vector per finger image.
    The group average feature and the three per-image features are concatenated
    (four vectors in total) and fed into separate age-regression and gender-
    classification heads.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """Initialize the model.

        Args:
            pretrained: If ``True``, load ImageNet-pretrained DenseNet-161 weights.
        """
        super().__init__()
        weights = DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.densenet161(weights=weights)
        self.backbone = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = base_model.classifier.in_features
        # avg_feat + feat1 + feat2 + feat3 -> 4 * feat_dim (e.g. 2208 * 4 = 8832)
        head_in_features = feat_dim * 4

        self.age_head = nn.Linear(head_in_features, 1)
        self.gender_head = nn.Linear(head_in_features, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, num_images, channels, height, width)``.

        Returns:
            Tuple of age predictions ``(batch, 1)`` and gender logits
            ``(batch, 2)``.
        """
        batch_size, num_images, channels, height, width = x.shape
        x = x.view(batch_size * num_images, channels, height, width)
        features = self.backbone(x)
        features = self.pool(features)
        features = torch.flatten(features, 1)
        features = features.view(batch_size, num_images, -1)

        feat1 = features[:, 0, :]
        feat2 = features[:, 1, :]
        feat3 = features[:, 2, :]
        avg_feat = features.mean(dim=1)
        combined = torch.cat([avg_feat, feat1, feat2, feat3], dim=1)

        age_out = self.age_head(combined)
        gender_out = self.gender_head(combined)
        return age_out, gender_out


def build_model(cfgs: dict) -> nn.Module:
    """Build a model from the configuration dictionary.

    Args:
        cfgs: Configuration dictionary containing a ``model`` section with
            ``name`` (``'multi'`` or ``'densenet'``) and optional ``pretrained``.

    Returns:
        Initialized ``nn.Module`` instance.

    Raises:
        ValueError: If ``model.name`` is not recognized.
    """
    model_cfg = cfgs["model"]
    if isinstance(model_cfg, dict):
        model_name = str(model_cfg["name"]).lower()
        pretrained = model_cfg.get("pretrained", True)
    else:
        model_name = str(model_cfg).lower()
        pretrained = True

    if model_name == "multi":
        return MAGEVein(pretrained=pretrained)

    if model_name == "densenet":
        weights = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet201(weights=weights)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 1)
        return model

    raise ValueError(f"Unknown model: {model_name}")


def select_model(cfgs: dict) -> nn.Module:
    """Backward-compatible alias for :func:`build_model`."""
    return build_model(cfgs)
