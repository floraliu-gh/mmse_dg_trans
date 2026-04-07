import torch
import torch.nn as nn

class ServerModel(nn.Module):
    """
    Server 端的 Vision Transformer (ViT) 模型
    輸入: [Batch, 64, 16, 16] 的 4D 特徵圖 (來自 Client 端)
    輸出: [Batch, 7, 64, 64] 的 Segmentation Mask
    """
    def __init__(self, embed_dim=64, num_heads=4, depth=2, num_classes=7):
        super().__init__()
        
        # Transformer Encoder 層 (後 N 層)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4, 
            dropout=0.1, 
            activation='gelu', 
            batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # ViT 分割頭 (將 16x16 空間特徵上採樣回 64x64 解析度)
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=2, stride=2), # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),        # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, kernel_size=1)                   # [B, 7, 64, 64]
        )

    def forward(self, A_k):
        # 取得 Batch(B), Channel(C), Height(H), Width(W)
        B, C, H, W = A_k.shape
        
        # 1. 展平空間維度並交換順序: [B, C, H, W] 轉換為 [B, H*W, C]
        x = A_k.view(B, C, H * W).permute(0, 2, 1)
        
        # 2. 接手計算剩餘的 Multi-Head Attention
        x = self.blocks(x)
        
        # 3. 轉回 4D 張量: [B, H*W, C] 轉換為 [B, C, H, W]
        spatial_feature = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        
        # 4. 通過 Decoder 上採樣至 [B, num_classes, 64, 64]
        out = self.segmentation_head(spatial_feature)
        return out