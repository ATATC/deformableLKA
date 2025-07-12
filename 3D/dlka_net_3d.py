import torch
from torch import nn
from typing import List

class ConvBlock(nn.Module):
    """A simple 3D convolutional block with BatchNorm and ReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding=pad)
        self.norm = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResidualBlock(nn.Module):
    """Two ConvBlocks with a residual connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels)
        )

    def forward(self, x):
        return x + self.block(x)

class LKA3d(nn.Module):
    """Large Kernel Attention used in D-LKA-Net."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv3d(dim, dim, 7, padding=9, dilation=3, groups=dim)
        self.conv1 = nn.Conv3d(dim, dim, 1)

    def forward(self, x):
        u = x
        x = self.conv0(x)
        x = self.conv_spatial(x)
        x = self.conv1(x)
        return u * x

class LKA_Attention3d(nn.Module):
    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.proj_1 = nn.Conv3d(dim, dim, 1, groups=num_heads)
        self.act = nn.GELU()
        self.lka = LKA3d(dim)
        self.proj_2 = nn.Conv3d(dim, dim, 1, groups=num_heads)

    def forward(self, x, B, C, H, W, D):
        x = x.permute(0,2,1).reshape(B, C, H, W, D)
        shortcut = x
        x = self.proj_1(x)
        x = self.act(x)
        x = self.lka(x)
        x = self.proj_2(x)
        x = x + shortcut
        x = x.reshape(B, C, H*W*D).permute(0,2,1)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(torch.ones(hidden_size) * 1e-6)
        self.attn = LKA_Attention3d(hidden_size, num_heads)
        self.conv = nn.Sequential(
            ResidualBlock(hidden_size),
            nn.Conv3d(hidden_size, hidden_size, 1)
        )

    def forward(self, x):
        B,C,H,W,D = x.shape
        inp = x.reshape(B,C,-1).permute(0,2,1)
        attn = inp + self.gamma * self.attn(self.norm(inp), B,C,H,W,D)
        attn = attn.permute(0,2,1).reshape(B,C,H,W,D)
        out = self.conv(attn)
        return out

class EncoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int):
        super().__init__()
        layers = [ConvBlock(in_ch, out_ch, stride=2)]
        layers += [ResidualBlock(out_ch) for _ in range(depth)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.blocks = nn.Sequential(*[TransformerBlock(out_ch) for _ in range(depth)])

    def forward(self, x, skip):
        x = self.up(x) + skip
        x = self.blocks(x)
        return x

class DLKANet3D(nn.Module):
    """Simplified implementation of the 3D D-LKA-Net."""
    def __init__(self, in_channels: int =1, out_channels: int =14, base_channels: int =32, depths: List[int]=[2,2,2,2]):
        super().__init__()
        self.enc1 = EncoderStage(in_channels, base_channels, depths[0])
        self.enc2 = EncoderStage(base_channels, base_channels*2, depths[1])
        self.enc3 = EncoderStage(base_channels*2, base_channels*4, depths[2])
        self.enc4 = EncoderStage(base_channels*4, base_channels*8, depths[3])

        self.dec3 = DecoderStage(base_channels*8, base_channels*4, depths[2])
        self.dec2 = DecoderStage(base_channels*4, base_channels*2, depths[1])
        self.dec1 = DecoderStage(base_channels*2, base_channels, depths[0])

        self.out = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.dec3(e4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.out(d1)

