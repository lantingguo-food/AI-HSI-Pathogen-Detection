import torch
import torch.nn as nn
from einops import rearrange

class SpectralTransformer(nn.Module):
    """A lightweight spectral transformer that treats the spectral dimension as sequence."""
    def __init__(self, bands, dim=128, depth=4, heads=4, mlp_ratio=2.0, num_classes=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Conv2d(bands, dim, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=int(dim*mlp_ratio), dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(dim, num_classes)
    def forward(self, x):
        # x: (N, C(bands), H, W) treat H*W as sequence
        x = self.embed(x)
        N, D, H, W = x.shape
        seq = rearrange(x, 'n d h w -> n (h w) d')
        out = self.encoder(seq)
        out = out.mean(dim=1)
        return self.head(out)
