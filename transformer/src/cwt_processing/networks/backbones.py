"""Neural network backbones for spectrum feature extraction."""
from __future__ import annotations

import math
from typing import Any
import torch
import torch.nn.functional as F
from torch import nn


class Conv1DBackbone(nn.Module):
    """1-D Convolutional neural network backbone.

    Transforms feature tensor (batch, in_channels, length) -> (batch, out_features).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_features: int = 4,
        hidden_channels: tuple[int, ...] = (32, 64, 128),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        layers: list[nn.Module] = []
        cur_in = in_channels
        for ch in hidden_channels:
            layers.extend([
                nn.Conv1d(cur_in, ch, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(ch),
                nn.GELU(),
            ])
            cur_in = ch

        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(cur_in, 64),
            nn.GELU(),
            nn.Linear(64, out_features),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, context: dict[str, Any] | None = None) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.net(x)


class MLPBackbone(nn.Module):
    """Multi-layer perceptron regression backbone."""

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 4,
        signal_length: int = 352,
        in_features: int | None = None,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        total_in = in_features if in_features is not None else in_channels * signal_length
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor, context: dict[str, Any] | None = None) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """Transformer encoder block matching GaussianDualWaveletTransformer reference."""

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        dim_feedforward: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.projection = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, tokens, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch, tokens, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch, tokens, self.nhead, self.head_dim).transpose(1, 2)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(float(self.head_dim))).softmax(dim=-1)
        attended = (weights @ v).transpose(1, 2).contiguous().view(batch, tokens, dim)
        return self.projection(attended)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self._attention(x))
        return self.norm2(x + self.feed_forward(x))


class PatchTransformerBackbone(nn.Module):
    """Patch-based Transformer encoder backbone.

    Accepts feature tensor (batch, in_channels, length), divides into non-overlapping
    patches, applies learned positional embeddings and Transformer blocks, and pools
    across tokens into (batch, out_features).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_features: int = 6,
        patch_size: int = 16,
        signal_length: int = 352,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.patch_size = patch_size
        self.n_tokens = signal_length // patch_size
        self.patch_embedding = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.empty(1, self.n_tokens, d_model))
        nn.init.uniform_(self.position, -0.02, 0.02)

        self.blocks = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, out_features),
        )

    def forward(self, x: torch.Tensor, context: dict[str, Any] | None = None) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z = self.patch_embedding(x).transpose(1, 2) + self.position
        for block in self.blocks:
            z = block(z)
        pooled = self.final_norm(z).mean(dim=1)
        return self.head(pooled)


class UNet1DBackbone(nn.Module):
    """1D U-Net encoder-decoder backbone reproducing model_full_cnn_v4.

    Accepts (batch, in_channels, length) and produces dual-channel heatmap (batch, 2, length):
        Channel 0: Peak classification / detection heatmap (sigmoid).
        Channel 1: Normalized peak height / amplitude regression (sigmoid).
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 2

        # Encoder Stage 1: (C, 352) -> (32, 176)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
        )

        # Encoder Stage 2: (32, 176) -> yc (64, 176) -> (64, 88)
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.pool2 = nn.MaxPool1d(2)

        # Encoder Stage 3: (64, 88) -> yb (128, 88) -> (128, 44)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )
        self.pool3 = nn.MaxPool1d(2)

        # Bottleneck: (128, 44) -> (256, 44)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        # Decoder Stage 1: (256, 44) -> (256, 88) + skip yb (256, 88) = (512, 88)
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        self.skip_b = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Decoder Stage 2: (512, 88) -> (128, 176) + skip yc (128, 176) = (256, 176)
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(512, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
        )
        self.skip_c = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Decoder Stage 3: (256, 176) -> (64, 352) -> (64, 352)
        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # Dual heads: classification heatmap & normalized height regression
        self.cls_head = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )
        self.reg_head = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, context: dict[str, Any] | None = None) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        y1 = self.conv1(x)
        yc = self.conv2(y1)
        y2 = self.pool2(yc)
        yb = self.conv3(y2)
        y3 = self.pool3(yb)
        y_bot = self.bottleneck(y3)

        y_up1 = self.up1(y_bot)
        y_cat1 = torch.cat([y_up1, self.skip_b(yb)], dim=1)
        y_up2 = self.up2(y_cat1)
        y_cat2 = torch.cat([y_up2, self.skip_c(yc)], dim=1)
        y_up3 = self.up3(y_cat2)

        cls = self.cls_head(y_up3)
        reg = self.reg_head(y_up3)
        return torch.cat([cls, reg], dim=1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, nhead, seq_len, head_dim)
    # cos, sin: (1, 1, seq_len, head_dim)
    return (x * cos) + (_rotate_half(x) * sin)


class RoPETransformerEncoderLayer(nn.Module):
    """Transformer encoder layer using Rotary Position Embeddings and F.scaled_dot_product_attention.

    Non-causal bidirectional self-attention layer adhering to the PyTorch TransformerEncoderLayer interface.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim % 2 == 0, f"head_dim ({self.head_dim}) must be even for RoPE"


        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)


        # Normalization and Feed-Forward using native PyTorch layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_p = dropout


        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )


        # Precompute RoPE rotary frequency buffers (Su et al., 2021)
        theta = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / float(self.head_dim)))
        positions = torch.arange(max_len).float()
        freqs = torch.outer(positions, theta)
        self.register_buffer("cos", freqs.cos().repeat_interleave(2, dim=-1))
        self.register_buffer("sin", freqs.sin().repeat_interleave(2, dim=-1))


    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = src.shape


        q = self.q_proj(src).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(src).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(src).view(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)


        # Apply Rotary Position Embedding to queries and keys
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)


        # Native PyTorch scaled dot-product attention (non-causal)
        dropout_p = self.dropout_p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=src_mask, dropout_p=dropout_p, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)


        # Residual + Norm
        src = self.norm1(src + self.dropout1(attn_out))
        src = self.norm2(src + self.dropout2(self.mlp(src)))
        return src




class RoPETransformerEncoderBackbone(nn.Module):
    """Transformer encoder backbone with linear embedding from 256 to 64,


    four non-causal encoder layers with RoPE, and dual output heads matching ReferenceCNN.


    Transforms feature tensor (batch, in_channels=256, length=352) -> (batch, 2, length=352):
        Channel 0: Peak detection heatmap (sigmoid).
        Channel 1: Normalized peak height regression (sigmoid).
    """


    def __init__(
        self,
        in_channels: int = 256,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        signal_length: int = 352,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.signal_length = signal_length


        # First layer: Linear embedding from in_channels (256) to 64
        self.embedding = nn.Linear(in_channels, d_model)

        # 4 non-causal Transformer encoder layers with RoPE
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_len=max(signal_length, 512),
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Dual output heads matching ReferenceCNN
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, context: dict[str, Any] | None = None) -> torch.Tensor:
        # Input: (batch, in_channels, length) -> transpose to (batch, length, in_channels)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        z = x.transpose(1, 2)
        # Linear embedding: (batch, length, in_channels) -> (batch, length, d_model)
        z = self.embedding(z)

        # 4 non-causal encoder layers with RoPE
        for layer in self.layers:
            z = layer(z)
        z = self.norm(z)

        # Dual heads: (batch, length, 1) -> transpose to (batch, 1, length)
        cls = self.cls_head(z).transpose(1, 2)
        reg = self.reg_head(z).transpose(1, 2)
        return torch.cat([cls, reg], dim=1)


