import torch
import torch.nn as nn
import timm

# ==========================================
# 1. RESNET50 ONLY
# ==========================================
class ResNet50Only(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.classifier = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# ==========================================
# 2. SWIN TRANSFORMER (FIXED TO TINY)
# ==========================================
class SwinOnly(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # FIXED: Changed from swin_base to swin_tiny to match checkpoint dimensions
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True, 
            num_classes=0
        )
        # FIXED: Swin-Tiny output dimension is 768, not 1024
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# ==========================================
# 3. CONVNEXT ONLY
# ==========================================
class ConvNeXtOnly(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# ==========================================
# 4. TWO-STREAM SPATIAL RESNET50
# ==========================================
class TwoStream_Spatial_ResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.classifier = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# ==========================================
# 5. NEXTSWIN-X (FIXED DIMENSIONS)
# ==========================================
class NextSwinX(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # FIXED: Swin Tiny (768 dim)
        self.swin_backbone = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True, 
            num_classes=0
        )
        # ConvNeXt Tiny also outputs 768 dim, ensuring perfect fusion alignment
        self.convnext = timm.create_model(
            'convnext_tiny', 
            pretrained=True, 
            num_classes=0
        )
        
        # Fusion layers (expects 768 dim input)
        self.ln = nn.LayerNorm(768)
        self.mlp = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        swin_feat = self.swin_backbone(x)   # [B, 768]
        conv_feat = self.convnext(x)        # [B, 768]
        
        # Average fusion
        fused = (swin_feat + conv_feat) / 2.0
        fused = self.ln(fused)
        out = self.mlp(fused)
        return out

# ==========================================
# 6. EFFICIENTNET V2 ONLY (FIXED TIMM NAME)
# ==========================================
class EfficientNetV2Only(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # FIXED: Use 'tf_efficientnetv2_b3' instead of 'efficientnetv2_b3'
        self.backbone = timm.create_model('tf_efficientnetv2_b3', pretrained=True, num_classes=0)
        # tf_efficientnetv2_b3 has 1536 features
        self.classifier = nn.Linear(1536, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# ==========================================
# 7. BIFNN RESNET50 (FIXED TIMM NAME)
# ==========================================
class BiFNN_ResNet50(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Branch 1: ResNet50 (2048 features)
        self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=0)
        
        # Branch 2: EfficientNet V2 B3 (FIXED name, 1536 features)
        self.efficientnet = timm.create_model('tf_efficientnetv2_b3', pretrained=True, num_classes=0)
        
        # Fusion: 2048 + 1536 = 3584
        self.fusion = nn.Sequential(
            nn.Linear(3584, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        feat1 = self.resnet(x)      # [B, 2048]
        feat2 = self.efficientnet(x) # [B, 1536]
        
        # Concatenate features
        fused = torch.cat((feat1, feat2), dim=1) # [B, 3584]
        out = self.fusion(fused)
        return out
