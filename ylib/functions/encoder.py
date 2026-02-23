import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from .utils import position_encoding_init, FFN_Network

class patch_embedding(nn.Module):
    def __init__(self, image_size, patch_size, emb_dim, in_channel=3):
        super().__init__()
        self.image_h, self.image_w = image_size
        self.patch_h, self.patch_w = patch_size
        self.in_channel = in_channel
        self.num_patches = (self.image_h//self.patch_h)*(self.image_w//self.patch_w)
        self.patch_dim = self.in_channel * self.patch_h * self.patch_w
        
        self.patch_emb = nn.Conv2d(
            in_channels=in_channel,
            out_channels=emb_dim,
            kernel_size=self.patch_h,
            stride=self.patch_h
        )

    def forward(self, x):
        """
        引数
        x: 入力画像（B, C, H, W）
        attn_mask:（B, 1, H, W）
        返り値
        埋め込みベクトル（B, N, D）
        attn_mask_patch:(B, N)
        """
        # パッチ分割と埋め込み
        # (B, C, H, W) -> (B, D, H/P, W/P)
        out = self.patch_emb(x)
        # パッチのflatten
        # (B, D, H/P, W/P) -> (B, D, N) ※N = パッチ数
        out = out.flatten(2)
        # 軸の入れ替え
        # (B, D, N) -> (B, N, D)
        out = out.transpose(1, 2)

        return out

class encoder_block(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        self.MHA = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.FFN = FFN_Network(hidden_size=hidden_size,
                                ffn_size=hidden_size*4,
                                dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, key_mask):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        attn_out, attn_weight = self.MHA(
            query=q, key=k, value=v,
            key_padding_mask=key_mask
        )
        out = self.layer_norm1(x + attn_out)
        ffn_out = self.FFN(out)

        return self.layer_norm2(out + ffn_out)

class encoder(nn.Module):
    def __init__(self, image_size, patch_size=(16, 16), emb_dim=768,
                hidden_size=768, num_heads=12, num_layers=8, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.image_h, self.image_w = image_size
        self.patch_h, self.patch_w = patch_size
        self.num_patches = (self.image_h//self.patch_h)*(self.image_w//self.patch_w)

        self.patch_emb = patch_embedding(image_size=image_size,
                                        patch_size=patch_size,
                                        emb_dim=emb_dim,
                                        in_channel=3)
        
        self.pos_enc = nn.Embedding(self.num_patches, emb_dim, padding_idx=0)
        self.pos_enc.weight.data = position_encoding_init(self.num_patches, 
                                                                emb_dim)
        self.enc_linear = nn.Linear(emb_dim, hidden_size)
        self.Encoder_Layer = nn.ModuleList([
            encoder_block(num_heads=num_heads, hidden_size=hidden_size, dropout=dropout)
            for _ in range(num_layers)])
    
    def forward(self, x, pos_x, key_mask):
        patch_emb = self.patch_emb(x)
        pos_enc = self.pos_enc(pos_x)
        features = self.layer_norm(patch_emb + pos_enc)
        if self.emb_dim != self.hidden_size:
            features = self.enc_linear(features)

        for encoder in self.Encoder_Layer:
            features = encoder(features, key_mask)
        
        return features