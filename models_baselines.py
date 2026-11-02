import torch
import torch.nn as nn
import timm

# ==============================================================================
# 1. SINGLE BACKBONE MODELS (Table 3 Baselines)
# ==============================================================================

class SingleBackbone(nn.Module):
    """Base class for single-backbone models."""
    def __init__(self, model_name, feature_dim, num_classes=4):
        super(SingleBackbone, self).__init__()
        # num_classes=0 strips the ImageNet head, giving us the raw feature vector
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class ConvNeXtOnly(SingleBackbone):
    def __init__(self, num_classes=4):
        super().__init__('convnext_base', feature_dim=1024, num_classes=num_classes)


class SwinOnly(SingleBackbone):
    def __init__(self, num_classes=4):
        super().__init__('swin_base_patch4_window7_224', feature_dim=1024, num_classes=num_classes)


class ResNet50Only(SingleBackbone):
    def __init__(self, num_classes=4):
        super().__init__('resnet50', feature_dim=2048, num_classes=num_classes)


class EfficientNetV2Only(SingleBackbone):
    def __init__(self, num_classes=4):
        super().__init__('efficientnetv2_b3', feature_dim=1536, num_classes=num_classes)


# ==============================================================================
# 2. CONCATENATION FUSION BASELINES (Table 3 Baselines)
# ==============================================================================

class ConcatenationFusion(nn.Module):
    """
    Baseline fusion: Extracts features from two backbones, concatenates them, 
    and passes them through an MLP. (Used for BiFNN and TwoStream baselines).
    """
    def __init__(self, model_name_1, dim_1, model_name_2, dim_2, num_classes=4):
        super(ConcatenationFusion, self).__init__()
        self.backbone1 = timm.create_model(model_name_1, pretrained=True, num_classes=0)
        self.backbone2 = timm.create_model(model_name_2, pretrained=True, num_classes=0)
        
        combined_dim = dim_1 + dim_2
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3), # Standard regularization for fusion heads
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.backbone1(x)
        f2 = self.backbone2(x)
        combined = torch.cat((f1, f2), dim=1)
        return self.mlp(combined)


class BiFNN_ResNet50(ConcatenationFusion):
    """ResNet50 + EfficientNetV2_B3 Fusion"""
    def __init__(self, num_classes=4):
        super().__init__('resnet50', 2048, 'efficientnetv2_b3', 1536, num_classes)


class TwoStream_Spatial_ResNet50(ConcatenationFusion):
    """ResNet50 + ConvNeXt-Base Fusion"""
    def __init__(self, num_classes=4):
        super().__init__('resnet50', 2048, 'convnext_base', 1024, num_classes)
