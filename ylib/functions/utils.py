from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from PIL import Image as Img
from PIL import ImageDraw as imgd
from PIL import ImageFont as imgf
import random

class make_input_img():
    def __init__(self):
        self.fontname = ["datasets/font/HGRGY.TTC", "datasets/font/HGRPP1.TTC", "datasets/font/HGRKK.TTC", "datasets/font/HGRME.TTC"]
    def __call__(self, text, img):
        # 背景画像の調整
        draw = imgd.Draw(img)
        # テキスト整形
        target = text

        text = text.replace('、', '\'')
        text = text.replace('-', 'I')
        text = text.replace('ー', 'I')
        r_fname = random.sample(self.fontname, 1)
        old_size = 30
        text_len = len(text)
        y_min = 15
        y_max = 19 + 30*(50-text_len)
        for n_index, t in enumerate(text):
            r = random.randint(0, 64)
            g = random.randint(0,64)
            b = random.randint(0,64)
            
            if n_index == 0:
                x = random.randint(25, 40)
                y = random.randint(y_min, y_max)
                # フォントサイズ38までは必ずすべての文字が収まる
                fontsize= random.randint(30, 38)
                old_size = random.randint(35, 38)
            else:
            # 中心は(13, y) 高さはテキストがすべて収まる位置
                if t == '\'':
                    x = random.randint(45, 50)
                    y = y_max+5
                    fontsize = random.randint(38, 40)
                    old_size = 30
                else:
                    x = random.randint(25, 40)
                    y = random.randint(y_min, y_max)
                    # フォントサイズ38までは必ずすべての文字が収まる
                    fontsize= random.randint(28, 38)
                    old_size = random.randint(35, 38)
            textRGB = (r, g, b)
            font = imgf.truetype(r_fname[0], fontsize)
            draw.text((x, y), t, fill=textRGB, font=font, anchor='mm')
            y_min = y + old_size
            y_max = y_min+4
            #bboxが欲しいとき return (左上の点のx座標, 左上の点のy座標, 右下の点のx座標, 右下の点のy座標)
            #bbox = draw.textbbox((x, y), t, font=font, anchor='md'))
        
        return img, target

def position_encoding_init(n_position, d_pos_vec):
    """
    Positional Encodingのための行列の初期化を行う
    :param n_position: int, 系列長
    :param d_pos_vec: int, 隠れ層の次元数
    :return: torch.tensor, size=(n_position, d_pos_vec)
    """
    # PADがある単語の位置はpos=0にしておき、position_encも0にする
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.tensor(position_enc, dtype=torch.float)

class FFN_Network(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(hidden_size, ffn_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Conv1d(ffn_size, hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        residual = x
        output = self.gelu(self.w_1(x.transpose(1, 2)))
        output = self.dropout(output)
        output = self.w_2(output).transpose(2, 1)
        return self.layer_norm(output + residual)

def get_trainable_parameters(model, multi=False):
        # Positional Encoding以外のパラメータを更新する
        if multi:
            enc_freezed_param_ids = set(map(id, model.module.encoder.pos_enc.parameters()))
            dec_freezed_param_ids = set(map(id, model.module.decoder.embedding.pos_enc.parameters()))
        else:
            freezed_param_ids = set(map(id, model.decoder.embedding.pos_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in model.parameters() if id(p) not in freezed_param_ids)
    
def get_attn_subsequent_mask(num_heads, input_ids, target_len, key_len, device):
    """
    未来の情報に対するattentionを0にするためのマスクを作成する
    :param input_ids: tensor, size=(batch_size, length)
    :return subsequent_mask: tensor, size=(batch_size, length, length)
    """
    attn_shape = (target_len, key_len)
    # 上三角行列(diagonal=1: 対角線より上が1で下が0)
    subsequent_mask = torch.triu(torch.ones(attn_shape, dtype=torch.bool, device=device), diagonal=1)
    subsequent_mask = subsequent_mask.repeat(num_heads*input_ids.size(0), 1, 1)
    return subsequent_mask