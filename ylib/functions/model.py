import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from .encoder import encoder
from .decoder import decoder

class transformer_model(nn.Module):
    def __init__(self, image_size, vocab_size, word_emb_dim=768, max_len=60, patch_size=(16, 16), emb_dim=768, 
                hidden_size=768, num_heads=12, num_layers=8, dropout=0.1):
        super().__init__()
        self.encoder = encoder(image_size=image_size, patch_size=patch_size,
                                emb_dim=emb_dim, hidden_size=hidden_size, num_heads=num_heads,
                                num_layers=num_layers, dropout=dropout)
        self.decoder = decoder(vocab_size=vocab_size, emb_dim=word_emb_dim, hidden_size=hidden_size,
                                max_len=max_len, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        self.out_linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, imgs, encoder_mask, pos_img, input_ids, pos_ids, slf_key_mask, attn_mask):
        encoder_out = self.encoder(imgs, pos_img, encoder_mask)
        decoder_out = self.decoder(encoder_out, encoder_mask, input_ids, pos_ids, slf_key_mask, attn_mask)
        out = self.out_linear(decoder_out)

        return out