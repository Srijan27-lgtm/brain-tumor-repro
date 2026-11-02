import torch
import torch.nn as nn
import timm

# ==============================================================================
# 1. SINGLE BACKBONE MODELS
# ==============================================================================
class SingleBackbone(nn.Module):
    def __init__(self, model_name, feature_dim, num_classes=4):
        super(SingleBackbone, self).__init__()
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
# 2. CONCATENATION FUSION BASELINES
# ==============================================================================
class ConcatenationFusion(nn.Module):
    def __init__(self, model_name_1, dim_1, model_name_2, dim_2, num_classes=4):
        super(ConcatenationFusion, self).__init__()
        self.backbone1 = timm.create_model(model_name_1, pretrained=True, num_classes=0)
        self.backbone2 = timm.create_model(model_name_2, pretrained=True, num_classes=0)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim_1 + dim_2, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.backbone1(x)
        f2 = self.backbone2(x)
        return self.mlp(torch.cat((f1, f2), dim=1))

class BiFNN_ResNet50(ConcatenationFusion):
    def __init__(self, num_classes=4):
        super().__init__('resnet50', 2048, 'efficientnetv2_b3', 1536, num_classes)

class TwoStream_Spatial_ResNet50(ConcatenationFusion):
    def __init__(self, num_classes=4):
        super().__init__('resnet50', 2048, 'convnext_base', 1024, num_classes)

# ==============================================================================
# 3. NEXTSWIN-X (Paper's Proposed Architecture)
# ==============================================================================
class NextSwinX(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Backbones (num_classes=0 removes the ImageNet head)
        self.cnn_backbone = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        self.swin_backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
        
        # Linear projections to shared 512-dim space (Eq 3 & 4)
        self.proj_cnn = nn.Linear(1024, 512)
        self.proj_swin = nn.Linear(1024, 512)
        
        # Cross-attention: 8 heads, 512 dim (64 per head). batch_first=True
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.ln = nn.LayerNorm(512)
        
        # Classification Head (Eq 7 & 8)
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 1. Extract features
        f_cnn = self.cnn_backbone(x)   # [B, 1024]
        f_swin = self.swin_backbone(x) # [B, 1024]
        
        # 2. Project to shared space
        p_cnn = self.proj_cnn(f_cnn)   # [B, 512]
        p_swin = self.proj_swin(f_swin) # [B, 512]
        
        # 3. Cross-attention (CNN is Query, Swin is Key & Value)
        # Add sequence dim for MHA: [B, 1, 512]
        q = p_cnn.unsqueeze(1)
        k = p_swin.unsqueeze(1)
        v = p_swin.unsqueeze(1)
        
        attn_out, _ = self.cross_attn(q, k, v) # [B, 1, 512]
        attn_out = attn_out.squeeze(1)         # [B, 512]
        
        # 4. Residual connection + LayerNorm (Eq 6)
        fused = self.ln(p_cnn + attn_out)
        
        # 5. Classification
        return self.mlp(fused)
