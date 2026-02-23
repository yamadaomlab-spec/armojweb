# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from ylib.util import box_ops
from ylib.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
import math


'''
Sin based positional encoding (not learnable)
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        scale = d_model ** -0.5
        pe = pe*scale
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]

        return self.dropout(x)
 

'''
ラベルスムージングを用いたloss関数
'''
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None

    def forward(self, x, target):
        size =  x.size(1) 
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        
        return self.criterion(x, Variable(true_dist, requires_grad=False))
    
#CrossEntropy
class LabelSmoothingCE(nn.Module):
    "Implement label smoothing."
    def __init__(self, padding_idx=0, dim=-1, smoothing=0.0):
        super(LabelSmoothingCE, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.true_dist = None
        self.dim = dim
        
    def forward(self, x, target):
        size =  x.size(1) 
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        
        return torch.mean(torch.sum(-self.true_dist * x, dim=self.dim)) 
    
'''
DHPDETR: 物体検出DETRを文書認識用にカスタマイズ（テキストとBBoxを予測）
'''
class DHPDETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_classes, hidden_dim)
        self.query_pos = PositionalEncoding(hidden_dim, .2, max_len=1024)
        
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    
    '''
    return mask like as
    tensor([[0., -inf, -inf, -inf, -inf],
            [0., 0., -inf, -inf, -inf],
            [0., 0., 0., -inf, -inf],
            [0., 0., 0., 0., -inf],
            [0., 0., 0., 0., 0.]])
    
    '''
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
    

    def forward(self, samples: NestedTensor, targets: NestedTensor):
        
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
 
        assert mask is not None
    
        # src:(bs, c, h, w), pos:(bs, hidden_dim, h, w), mask(bs, h, w)       
        src = self.input_proj(src) + pos[-1] #(bs, hidden_dim, h, w)
        src = src.flatten(2).permute(2, 0, 1) #(S=hw, bs, hidden_dim)
        mask = mask.flatten(1)#(bs, hw)
        
        if isinstance(targets, (list, torch.Tensor)):
            targets = nested_tensor_from_tensor_list(targets)  
        trg, trg_pad_mask = targets.decompose()    
            
        trg_mask = self.generate_square_subsequent_mask(trg.shape[1]).to(trg.device)
    
        
        trg = self.query_embed(trg)#(bs, T, hidden_dim)
        trg = self.query_pos(trg).permute(1,0,2)#(T, bs, hidden_dim)
        
        
        hs = self.transformer(src, #(S, bs, hidden_dim)
                              trg, #(T, bs, hidden_dim)
                              tgt_mask=trg_mask, #(T, T)
                              src_key_padding_mask=mask, #(bs, S)
                              memory_key_padding_mask=mask,#(bs, S)
                              tgt_key_padding_mask=trg_pad_mask)#(bs, T)
        
        #hs: (T, bs, hidden_dim)
        
        hs = hs.transpose(0,1)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out
    

'''
DHPDETR: 物体検出DETRを文書認識用にカスタマイズ（テキストとBBoxを予測）
'''
class DHPDETR_EXT(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        self.query_embed = nn.Embedding(num_classes, hidden_dim*15//16)
        self.query_pos = PositionalEncoding(hidden_dim, .2, max_len=128)
        
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

        self.cursor_embed = nn.Linear(2, hidden_dim//16)
        
    '''
    return mask like as
    tensor([[0., -inf, -inf, -inf, -inf],
            [0., 0., -inf, -inf, -inf],
            [0., 0., 0., -inf, -inf],
            [0., 0., 0., 0., -inf],
            [0., 0., 0., 0., 0.]])
    
    '''
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask
    
            
    def forward(self, samples: NestedTensor, targets: NestedTensor, cursors):
        
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
 
        assert mask is not None
    
        # src:(bs, c, h, w), pos:(bs, hidden_dim, h, w), mask(bs, h, w)       
        src = self.input_proj(src) + pos[-1] #(bs, hidden_dim, h, w)
        src = src.flatten(2).permute(2, 0, 1) #(S=hw, bs, hidden_dim)
        mask = mask.flatten(1)#(bs, hw)
        
        if isinstance(targets, (list, torch.Tensor)):
            targets = nested_tensor_from_tensor_list(targets)  
        trg, trg_pad_mask = targets.decompose()    
            
        trg_mask = self.generate_square_subsequent_mask(trg.shape[1]).to(trg.device)
        
        trg = self.query_embed(trg)#(bs, T, hidden_dim*15/16)
#         csr = self.cursor_embed(cursors).permute(1,0,2) #(T, bs, hidden_dim/16)
        csr = self.cursor_embed(cursors)#(bs, T, hidden_dim/16)
        trg = torch.cat([trg, csr],dim=2)        
        trg = self.query_pos(trg).permute(1,0,2)#(T, bs, hidden_dim*15/16)
        
        hs = self.transformer(src, #(S, bs, hidden_dim)
                              trg, #(T, bs, hidden_dim)
                              tgt_mask=trg_mask, #(T, T)
                              src_key_padding_mask=mask, #(bs, S)
                              memory_key_padding_mask=mask,#(bs, S)
                              tgt_key_padding_mask=trg_pad_mask)#(bs, T)
        
        #hs: (T, bs, hidden_dim)
        
        hs = hs.transpose(0,1)
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out    
    
class SetCriterion(nn.Module):
    """ This class computes the loss 
    """
    def __init__(self, weight_dict, lossfunctype= 'CE', padding_idx = 0, eos_idx = 2, smoothing = 0.0, losses = ['labels', 'boxes']):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict
        self.padding_idx = padding_idx
        self.eos_idx = eos_idx
        self.smoothing = smoothing
#         self.labelsmoothingloss = nn.CrossEntropyLoss(ignore_index=padding_idx, label_smoothing=smoothing) # pytorch 2.0>

        self.lossfunctype = lossfunctype
    
        if self.lossfunctype != 'CE':
            #KLDivLoss
            self.labelsmoothingloss = LabelSmoothing(padding_idx, smoothing)
        else:
            #CrossEntropy
            self.labelsmoothingloss = LabelSmoothingCE(padding_idx=padding_idx, dim=-1, smoothing=smoothing)

    def loss_labels(self, outputs, targets, log=True):
        assert 'pred_logits' in outputs
        
        #t: label of [<SOS>, a, b, c, d, <EOS>]
        t, m = targets['labels'].decompose()
        
        #target_classes: label of [a, b, c, d <EOS>]
        target_classes = t[:, 1:]
        mask = ~m[:, 1:]
    
        #src_logits: logis of [a, b, c, d]
        src_logits = (outputs['pred_logits'])[:, :-1, :]
        
        #class error
        if self.lossfunctype != 'CE':
            # KLDivLoss
            loss_ce = self.labelsmoothingloss(src_logits[mask].log_softmax(-1), target_classes[mask])/len(mask)
        else:
            # ClossEntropyLoss
            loss_ce = self.labelsmoothingloss(src_logits[mask].log_softmax(-1), target_classes[mask])
        
        losses = {'loss_ce': loss_ce}
        
        
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[mask], target_classes[mask])[0]
            
        return losses


    def loss_boxes(self, outputs, targets):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        
        t, m = targets['labels'].decompose()
        target_classes = t[:, 1:]
        mask = (~m[:, 1:])*(target_classes != self.eos_idx)
        
        src_boxes = (outputs['pred_boxes'])[:, :-1, :]
        target_boxes = targets['boxes']

        #By default, the losses are averaged over each loss element
        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes)

        losses = {}
        losses['loss_bbox'] = loss_bbox

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / len(target_boxes)
        return losses



    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses


# 使ってないかも
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, targets, orig_target_sizes, padding_idx, eos_idx):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        t, m = targets['labels'].decompose()
        target_classes = t[:, 1:]
        mask_classes = ~m[:, 1:]
        mask_boxes = (~m[:, 1:])*(target_classes != eos_idx)
        len_target_classes = target_classes.shape[1]
        out_logits = (outputs['pred_logits'])[:, :len_target_classes]
        out_boxes = (outputs['pred_boxes'])[:, :len_target_classes, :]
        

        assert len(out_boxes) == len(orig_target_sizes)
        assert orig_target_sizes.shape[1] == 2

        prob = F.softmax(out_logits[mask_classes], -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_boxes[mask_boxes])
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_w, img_h = orig_target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results
    

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


    
def build(args, num_classes):

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = nn.Transformer(
        d_model = args.hidden_dim, 
        nhead = args.nheads, 
        num_encoder_layers = args.enc_layers, 
        num_decoder_layers = args.dec_layers)


    if args.method == 'crbbcu':
        model = DHPDETR_EXT(
            backbone,
            transformer,
            num_classes=num_classes
        )
    else:
        model = DHPDETR(
            backbone,
            transformer,
            num_classes=num_classes
        )        

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    losses = ['labels', 'boxes']
    criterion = SetCriterion(weight_dict, lossfunctype ='CE', padding_idx = 0, eos_idx = 3, smoothing = 0.1, losses=losses)
    criterion.to(device)
    
    postprocessors = {'bbox': PostProcess()}
    
    return model, criterion, postprocessors
