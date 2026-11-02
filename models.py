import torch
import torch.nn as nn
import timm

class NeXtSwinX(nn.Module):
    def __init__(self, num_classes=4):
        super(NeXtSwinX, self).__init__()
        
        # 1. Parallel Feature Extraction (Eq 1 & 2)
        # num_classes=0 removes the default ImageNet classification head, 
        # leaving us with the 1024-dim feature vector.
        self.cnn_backbone = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        self.swin_backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
        
        # 2. Linear Projections (Eq 3 & 4)
        # Maps 1024-dim features to a shared 512-dim latent space.
        self.proj_cnn = nn.Linear(1024, 512)
        self.proj_swin = nn.Linear(1024, 512)
        
        # 3. Cross-Attention Fusion (Eq 5)
        # Paper: "multi-head attention module with eight heads. Each head operates on a 64-dimensional subspace (512/8)"
        # batch_first=True is crucial: it expects inputs in (Batch, Seq_Len, Embed_Dim) format.
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        
        # 4. Residual Connection & Layer Normalization (Eq 6)
        self.layer_norm = nn.LayerNorm(512)
        
        # 5. Classification Head / MLP (Eq 7 & 8)
        # Paper: "two-layer Multi-Layer Perceptron (MLP)... intermediate dimension of 256"
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(), # GELU is standard for modern backbones like ConvNeXt/Swin
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Step A: Extract Features
        # Output shape for both: [Batch_Size, 1024]
        f_cnn = self.cnn_backbone(x)
        f_swin = self.swin_backbone(x)
        
        # Step B: Project to shared latent space
        # Output shape: [Batch_Size, 512]
        p_cnn = self.proj_cnn(f_cnn)
        p_swin = self.proj_swin(f_swin)
        
        # Step C: Prepare for Cross-Attention
        # nn.MultiheadAttention expects 3D tensors: (Batch, Seq_Len, Embed_Dim)
        # Since we have single feature vectors, we treat them as sequences of length 1.
        # Output shape: [Batch_Size, 1, 512]
        p_cnn_3d = p_cnn.unsqueeze(1)
        p_swin_3d = p_swin.unsqueeze(1)
        
        # Step D: Asymmetric Cross-Attention
        # Query = CNN features, Key = Swin features, Value = Swin features
        # attn_out shape: [Batch_Size, 1, 512]
        attn_out, _ = self.cross_attn(p_cnn_3d, p_swin_3d, p_swin_3d)
        
        # Step E: Residual Connection + LayerNorm
        # Output shape: [Batch_Size, 1, 512]
        fused_3d = self.layer_norm(p_cnn_3d + attn_out)
        
        # Step F: Squeeze back to 2D for the MLP
        # Output shape: [Batch_Size, 512]
        fused_2d = fused_3d.squeeze(1)
        
        # Step G: Final Classification
        # Output shape: [Batch_Size, num_classes]
        out = self.classifier(fused_2d)
        
        return out
