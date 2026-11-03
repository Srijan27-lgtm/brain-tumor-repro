import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ==========================================
# 1. SCRATCH CNN (Custom Baseline)
# ==========================================
class ScratchCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. RESNET50
# ==========================================
class ResNet50Only(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.classifier = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)

# ==========================================
# 3. CONVNEXT
# ==========================================
class ConvNeXtOnly(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)

# ==========================================
# 4. TWO-STREAM (Spatial ResNet50)
# ==========================================
class TwoStreamSpatialResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.stream1 = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.stream2 = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        feat1 = self.stream1(x)
        feat2 = self.stream2(x)
        combined = torch.cat([feat1, feat2], dim=1)
        return self.classifier(combined)

# ==========================================
# 5. BiFNN (Bimodal Feature Neural Network)
# ==========================================
class BiFNN_ResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        feat = self.backbone(x)
        feat = F.relu(self.fc1(feat))
        feat = F.relu(self.fc2(feat))
        return self.classifier(feat)

# ==========================================
# 6. SWIN-TRANSFORMER
# ==========================================
class SwinTransformerOnly(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        feat = self.backbone(x)
        # Handle potential 3D output from some timm versions
        if feat.dim() == 3:
            feat = feat.mean(dim=1)
        return self.classifier(feat)

# ==========================================
# 7. EFFICIENTNET V2
# ==========================================
class EfficientNetV2Only(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnetv2_b3', pretrained=True, num_classes=0)
        self.classifier = nn.Linear(1536, num_classes)
        
    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() > 2:
            feat = feat.mean(dim=1) if feat.dim() == 3 else feat.flatten(1)
        return self.classifier(feat)

# ==========================================
# 8. NEXTSWIN-X (PROPOSED MODEL - FIXED)
# ==========================================
class NextSwinX(nn.Module):
    """
    Cross-Attention Fusion of ConvNeXt and Swin-Transformer.
    
    FIXED: Uses forward_features() and robust flattening to handle
    tensor shape mismatches across different timm versions.
    """
    def __init__(self, num_classes=4):
        super().__init__()
        # Backbones (pretrained=False because we load custom fusion weights)
        self.cnn_backbone = timm.create_model('convnext_base', pretrained=False, num_classes=0)
        self.swin_backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=0)
        
        # LayerNorm for ConvNeXt features (matches checkpoint structure)
        self.cnn_backbone.head.norm = nn.LayerNorm(1024)
        
        # Projections
        self.proj_cnn = nn.Linear(1024, 512)
        self.proj_swin = nn.Linear(1024, 512)
        
        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        
        # Classifier
        self.layer_norm = nn.LayerNorm(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
        
    def _robust_flatten(self, x):
        """
        Converts any shape [B, ..., 1024, ...] to [B, 1024]
        Handles NCHW, NHWC, NCL, NLC formats automatically.
        """
        if x.dim() == 2:
            return x
            
        # Check if channels are at dim 1 (NCHW/NCL) or dim -1 (NHWC/NLC)
        if x.shape[1] == 1024:
            # Format: [B, 1024, ...]
            return x.mean(dim=tuple(range(2, x.dim())))
        elif x.shape[-1] == 1024:
            # Format: [B, ..., 1024]
            return x.mean(dim=tuple(range(1, x.dim()-1)))
        else:
            # Fallback for flattened vectors
            if x.shape[1] % 1024 == 0:
                return x.view(x.shape[0], 1024, -1).mean(dim=2)
            elif x.shape[-1] % 1024 == 0:
                return x.view(x.shape[0], -1, 1024).mean(dim=1)
            else:
                raise ValueError(f"Cannot determine feature dimension in shape {x.shape}")

    def forward(self, x):
        # --- CONVNEXT BRANCH ---
        # Use forward_features to bypass timm's internal head
        cnn_feat = self.cnn_backbone.forward_features(x)
        
        # Robustly flatten to [B, 1024]
        cnn_feat = self._robust_flatten(cnn_feat)
        
        # Apply LayerNorm from checkpoint
        cnn_feat = self.cnn_backbone.head.norm(cnn_feat)
        cnn_feat = self.proj_cnn(cnn_feat)
        
        # --- SWIN BRANCH ---
        # Use forward_features to get raw patch tokens
        swin_feat = self.swin_backbone.forward_features(x)
        
        # Robustly flatten to [B, 1024]
        swin_feat = self._robust_flatten(swin_feat)
        swin_feat = self.proj_swin(swin_feat)
        
        # --- CROSS-ATTENTION FUSION ---
        q = cnn_feat.unsqueeze(1)  # [B, 1, 512]
        k = swin_feat.unsqueeze(1) # [B, 1, 512]
        v = k
        
        attn_out, _ = self.cross_attn(q, k, v)
        out = self.layer_norm(attn_out.squeeze(1))
        
        return self.classifier(out)
