import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from .utils import position_encoding_init, FFN_Network

class dec_embedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, max_len):
        super().__init__()
        self.word_emb = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=3
        )
        self.pos_enc = nn.Embedding(max_len, emb_dim, padding_idx=0)
        self.pos_enc.weight.data = position_encoding_init(max_len, emb_dim)
    def forward(self, x, x_pos):
        word_emb = self.word_emb(x)
        pos_enc = self.pos_enc(x_pos)

        return word_emb + pos_enc
    
class GPT_Block(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.slf_MHA = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
            # batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.FFN = FFN_Network(hidden_size=hidden_size,
                                ffn_size=hidden_size*4,
                                dropout=dropout)
    
    def forward(self, x, key_mask, attn_mask):

        x = self.layer_norm1(self.dropout(x))
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        attn_out, attn_weight = self.slf_MHA(
            query=q, key=k, value=v,
            key_padding_mask=key_mask,
            attn_mask=attn_mask
        )
        attn_out = self.layer_norm2(x+attn_out)
        out = self.FFN(attn_out)

        return out

class target_source_block(nn.Module):
    def __init__(self, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        self.ts_MHA = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
            # batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        self.FFN = FFN_Network(hidden_size=hidden_size,
                                ffn_size=hidden_size*4,
                                dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, encoder_out, encoder_mask, dec_input):
        
        q = self.q_linear(dec_input)
        k = self.k_linear(encoder_out)
        v = self.v_linear(encoder_out)
        attn_out, attn_weight = self.ts_MHA(
            query=q, key=k, value=v,
            key_padding_mask=encoder_mask
        )
        out = self.layer_norm1(attn_out + dec_input)
        ffn_out = self.FFN(attn_out)
        
        return self.layer_norm2(out + ffn_out)
    
class decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, max_len, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.embedding = dec_embedding(vocab_size, emb_dim, max_len)
        self.dec_linear = nn.Linear(emb_dim, hidden_size)
        self.gpt_Layer = nn.ModuleList([
            GPT_Block(num_heads=num_heads, hidden_size=hidden_size, dropout=dropout)
            for _ in range(num_layers)])
        self.ts_attn_Layer = nn.ModuleList([
            target_source_block(num_heads=num_heads, hidden_size=hidden_size, dropout=dropout)
            for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    
    def forward(self, encoder_out, encoder_mask, input_ids, pos_ids, slf_key_mask, attn_mask):
        dec_out = self.embedding(input_ids, pos_ids)
        if self.emb_dim != self.hidden_size:
            dec_out = self.dec_linear(dec_out)
        for decoder in self.gpt_Layer:
            dec_out = decoder(dec_out, slf_key_mask, attn_mask)
        
        ts_out = sum([(ts_attn(encoder_out, encoder_mask, dec_out)) for ts_attn in self.ts_attn_Layer])

        return self.layer_norm(ts_out)