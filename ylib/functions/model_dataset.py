import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from PIL import Image as Img
import random
from .utils import make_input_img

class kindai_dataset(torch.utils.data.Dataset):
    def __init__(self, text, img_path, img_transforms, max_len, tokenizer, patch_h, patch_w):
        self.texts = text
        self.img_path = img_path
        self.transforms = img_transforms
        self.pad_transform = transforms.Compose([
                        transforms.Resize((patch_h, patch_w), antialias=True)
                        ])
        self.make_img = make_input_img()
        self.max_len = max_len
        self.h = 1952
        self.w = 64
        self.tokenizer = tokenizer
        self.PAD = tokenizer.label_2_id('[PAD]')

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        # make_images
        path = random.sample(self.img_path, 1)
        img = Img.open(path[0])
        img, target = self.make_img(text, img)
        
        # enocer_input
        tensor_img = self.transforms(img)
        image_h, image_w = tensor_img[0].shape
        mask_h = image_h+16
        encoder_input = tensor_img.new_zeros((3, self.h, self.w))
        encoder_mask = tensor_img.new_ones((1, self.h, self.w), dtype=torch.bool)
        encoder_input[:, :image_h, :image_w] = tensor_img
        encoder_mask[:, :mask_h, :image_w] = False
        encoder_mask = self.pad_transform(encoder_mask)
        encoder_mask = torch.flatten(encoder_mask[0])
        pos_img = torch.arange(1, encoder_mask.size(0)+1) * ((encoder_mask == False).to(int))
        
        # decoder_input
        input_ids = self.tokenizer.encode_texts('[BOS]'+target+'[EOS]')
        target = torch.tensor(input_ids)
        pad_target = torch.ones(self.max_len+1, dtype=target.dtype)*self.PAD
        pad_key_mask = torch.ones(self.max_len, dtype=torch.bool)
        pad_target[:target.size(0)] = target
        pad_key_mask[:target.size(0)] = False
        dec_input = pad_target[:-1]
        pos_ids = torch.arange(1, self.max_len+1) * ((pad_key_mask == False).to(torch.int))
        # pad_target : target, pad_key_mask : slf_key_mask
        return encoder_input, encoder_mask, pos_img, dec_input, pos_ids, pad_target, pad_key_mask


class dhp_dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, img_transforms, max_len, tokenizer, patch_h, patch_w):
        self.dataframe = dataframe
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.transforms = img_transforms
        self.pad_transform = transforms.Compose([
                        transforms.Resize((patch_h, patch_w), antialias=True)
                        ])
        self.h = 1952
        self.w = 64
        self.PAD = tokenizer.label_2_id('[PAD]')

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
        # images
        img = self.dataframe.iloc[index]['images']
        # enocer_input
        tensor_img = self.transforms(img)
        image_h, image_w = tensor_img[0].shape
        mask_h = image_h+16
        encoder_input = tensor_img.new_zeros((3, self.h, self.w))
        encoder_mask = tensor_img.new_ones((1, self.h, self.w), dtype=torch.bool)
        encoder_input[:, :image_h, :image_w] = tensor_img
        encoder_mask[:, :mask_h, :image_w] = False
        encoder_mask = self.pad_transform(encoder_mask)
        encoder_mask = torch.flatten(encoder_mask[0])
        pos_img = torch.arange(1, encoder_mask.size(0)+1) * ((encoder_mask == False).to(int))
        
        # decoder_input
        target = self.dataframe.iloc[index]['targets']
        input_ids = self.tokenizer.encode_texts('[BOS]'+target+'[EOS]')
        target = torch.tensor(input_ids)
        pad_target = torch.ones(self.max_len+1, dtype=target.dtype)*self.PAD
        pad_key_mask = torch.ones(self.max_len, dtype=torch.bool)
        pad_target[:target.size(0)] = target
        pad_key_mask[:target.size(0)] = False
        dec_input = pad_target[:-1]
        pos_ids = torch.arange(1, self.max_len+1) * ((pad_key_mask == False).to(torch.int))
        # pad_target : target, pad_key_mask : slf_key_mask
        return encoder_input, encoder_mask, pos_img, dec_input, pos_ids, pad_target, pad_key_mask
