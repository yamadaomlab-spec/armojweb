import sys
import pprint

pprint.pprint(sys.path)

import torch
import torchvision.transforms as transforms
import torchvision
import torchvision.transforms.v2 as v2
import torchvision.tv_tensors as tv_tensors

from ylib.datasets.tokenizer import Tokenizer
from ylib.util.misc import nested_tensor_from_tensor_list
from ylib.models import build_model
import ylib.models
sys.modules['models'] = ylib.models

import argparse
import pickle
from PIL import Image, ImageDraw
from pathlib import Path
import os
import glob

from ylib.config import get_args_parser

class LineInferenceWapper():
    def __init__(self, model_basename):
        super(LineInferenceWapper, self).__init__()
        parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        
        self.args = parser.parse_args([
            '--name_dataset', 'dhp20231110',
            # '--charset', './data/dicts/dhp20231110_charset.pkl',
            '--charset', 'marge_and_dhp20231110_katsujisample20250606_and_copus_charset.pkl',
            # '--charset', 'marge_and_dhp20231110_katsujisample20250606_charset.pkl',
            # '--charset', 'marge_and_dhp20231110_charset.pkl',
            '--kfold', '10',
            '--fold_id', '0',
            '--transformer_type', 'original',
            '--backbone', 'swin',
            '--method', 'crbb',
            '--enc_layers', '6',
            '--dec_layers', '6',
            '--batch_size', '1',
            '--decoder_input_is_representative_chars',
            '--addstr', 'pretrain_marge39_kz10_r2i'
            ])

        self.accuracy_mode =  'standard'
        self.w = 64

        # charset_filename = './data/dicts/dhp20231110_charset.pkl'
        charset_filename = './data/dicts/marge_and_dhp20231110_katsujisample20250606_and_copus_charset.pkl'
        # charset_filename = './data/dicts/marge_and_dhp20231110_katsujisample20250606_charset.pkl'
        # charset_filename = './data/dicts/marge_and_dhp20231110_charset.pkl'
        # charset_filename = './data/dicts/minhon_and_dhp20231110_charset.pkl'
        # charset_filename = './data/dicts/charset_base.pkl'
        with open(charset_filename, 'rb') as f:
            charset_base = pickle.load(f)

        num_classes = len(charset_base)+4
        self.max_text_length = 128
        
        #20240909
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = 'cpu'

        # Tokenizer
        path_to_itaiji = f'./data/dicts/itaiji.txt'
        itaiji = {}
        with open(path_to_itaiji, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                line = line.split(',')
                itaiji[line[0]] = line[1:]
        self.tokenizer = Tokenizer(charset_base, itaiji, self.max_text_length)
        num_classes, num_ext_classes = self.tokenizer.num_representative_chars, self.tokenizer.num_chars

        self.model, _, _ = build_model(self.args, num_classes, num_ext_classes)

        
        with open('./data/modelL/' + model_basename, 'rb') as f:
            self.model.load_state_dict(torch.load(f))
            # self.model = pickle.load(f)
            # print(self.model)

        # 20240909
        _= self.model.to(self.device)

        self.infer_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # self.infer_transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ]
        # )


    def get_memory(self, model, samples):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = model.backbone(samples)

        src, mask = features[-1].decompose()
        
        src = model.input_proj(src) + pos[-1] #(bs, hidden_dim, h, w)
        src = src.flatten(2).permute(2, 0, 1) #(S=hw, bs, hidden_dim)
        mask = mask.flatten(1)#(bs, hw)
        
        return model.transformer.encoder(src, src_key_padding_mask = mask), mask

    @torch.no_grad()
    def inference(self, model, images, device, max_text_length=128, beam_width=3, topk=10):
        if self.accuracy_mode == 'standard':
            beam_width = 1
            topk = 1

        model.eval()
        
        predict_classes = []
        predict_class_confidences = []
        predict_text = []
        predict_boxes = []
        
        predict_ext_classes = []
        predict_ext_class_confidences = []
        predict_ext_text = []

        for img in images:     
            img = self.infer_transform(img)
            img = img.to(device)    
            memory, mask = self.get_memory(model, [img])

            best_seq, best_ext_seq, best_box_seq, best_cur_seq, best_score = self.beam_search_decode_batched(
                model, memory, mask, self.tokenizer,
                beam_width=beam_width,  # 例としてビーム幅5
                topk = topk,
                max_text_length=max_text_length,
                device=device,
                decoder_input_is_representative_chars=self.args.decoder_input_is_representative_chars,
                decoder_output_representative_chars=self.args.decoder_output_representative_chars
            )

                
            if self.args.decoder_output_representative_chars:
                predict_classes.append(self.tokenizer.representative_decode_as_list(best_seq))
                predict_class_confidences.append(best_score)
                predict_text.append(self.tokenizer.representative_decode(best_seq))
            
            predict_ext_classes.append(self.tokenizer.decode_as_list(best_ext_seq, 1, len(best_ext_seq)-1))
            predict_ext_class_confidences.append(best_score)
            predict_ext_text.append(self.tokenizer.decode(best_ext_seq))
            predict_boxes.append(best_box_seq)

        predict_text = list(map(lambda x : x.replace('SOS','').replace('EOS',''),predict_text))
        predict_ext_text = list(map(lambda x : x.replace('SOS','').replace('EOS',''),predict_ext_text))
        
        return predict_classes, predict_class_confidences, predict_text, \
            predict_ext_classes, predict_ext_class_confidences, predict_ext_text, \
            predict_boxes
    
    # @torch.no_grad()
    # def inference(self, model, images, device, max_text_length=128):
    #     model.eval()
        
    #     predict_classes = []
    #     predict_class_confidences = []
    #     predict_text = []
    #     predict_boxes = []
        
    #     predict_ext_classes = []
    #     predict_ext_class_confidences = []
    #     predict_ext_text = []

    #     for img in images:     
    #         img = self.infer_transform(img)
    #         img = img.to(device)    
    #         memory, mask = self.get_memory(model, [img])
    #         out_indexes = [self.tokenizer.chars.index('SOS'), ]
    #         out_conficences = []
    #         out_ext_indexes = [self.tokenizer.chars.index('SOS'), ]
    #         out_ext_conficences = []
    #         out_boxes = []

    #         if model.__class__.__name__ == 'DHPDETR_EXT':
    #             out_cursors = [[1.0, 0.0]]

    #         for i in range(self.max_text_length):#<-ここだけmax_text_lengthを使ってる
    #             trg_mask = model.generate_square_subsequent_mask(i+1).to(device)
    #             if self.args.decoder_input_is_representative_chars:
    #                 trg = torch.LongTensor(out_indexes).unsqueeze(1).to(device) #trg_tensor:[i+1,1]
    #             else:
    #                 trg = torch.LongTensor(out_ext_indexes).unsqueeze(1).to(device) #trg_tensor:[i+1,1]
    #             trg = model.query_embed(trg)

    #             if model.__class__.__name__ == 'DHPDETR_EXT':
    #                 csr = torch.Tensor(out_cursors).unsqueeze(1).to(device) #[i+1, 1, 2]
    #                 csr = model.cursor_embed(csr) #[i+1, 1, hidden_dim/16]
    #                 trg = torch.cat([trg, csr],dim=2)#[i+1, 1, hidden_dim]

    #             trg = model.query_pos(trg)
    #             hs = model.transformer.decoder(trg, memory, memory_key_padding_mask = mask, tgt_mask=trg_mask)
    #             hs = hs.transpose(0, 1)

    #             # 異体字の出力
    #             out_ext_logits = model.ext_class_embed(hs)
    #             last_ext_label = out_ext_logits.argmax(2)[:,-1].item()
    #             last_ext_confidences = torch.nn.functional.softmax(out_ext_logits[:, -1], dim=1)# logits to probability (only last charactor) 
    #             last_ext_label_confidence = last_ext_confidences[:, last_ext_label].item()
    #             out_ext_conficences.append(last_ext_label_confidence)
    #             out_ext_indexes.append(last_ext_label)

    #             # 代表字の出力
    #             if self.args.decoder_output_representative_chars:
    #                 out_logits = model.class_embed(hs)
    #                 last_label = out_logits.argmax(2)[:,-1].item()
    #                 last_confidences = torch.nn.functional.softmax(out_logits[:, -1], dim=1)# logits to probability (only last charactor) 
    #                 last_label_confidence = last_confidences[:, last_label].item()
    #                 out_conficences.append(last_label_confidence)
    #                 out_indexes.append(last_label)            
    #             else:
    #                 # 代表字出力がない場合は、異体字出力から代表字出力を求める
    #                 out_indexes.append(self.tokenizer.representative_index(self.tokenizer.chars[last_ext_label]))

    #             #bboxの出力
    #             out_coords = model.bbox_embed(hs).sigmoid()
    #             last_boxes = out_coords[:, -1].to('cpu').detach().numpy().copy()
                
    #             if model.__class__.__name__ == 'DHPDETR_EXT':
    #                 box = last_boxes[0]
    #                 out_cursors.append([box[0], box[1]+0.5*box[3]])

    #             # 20250213     
    #             if self.args.decoder_input_is_representative_chars:
    #                 if self.args.decoder_output_representative_chars:
    #                     if last_label == self.tokenizer.representative_chars.index('EOS'):
    #                         break
    #                 else:
    #                     if last_ext_label == self.tokenizer.chars.index('EOS'):
    #                         break
    #             elif last_ext_label == self.tokenizer.chars.index('EOS'):
    #                 break
                    
    #             # 20250204
    #             if i< max_text_length-1:
    #                 out_boxes.append(last_boxes[0])   
                
    #         if self.args.decoder_output_representative_chars:
    #             predict_classes.append(self.tokenizer.representative_decode_as_list(out_indexes))
    #             predict_class_confidences.append(out_conficences)
    #             predict_text.append(self.tokenizer.representative_decode(out_indexes))
            
    #         predict_ext_classes.append(self.tokenizer.decode_as_list(out_ext_indexes, 1, len(out_indexes)-1))
    #         predict_ext_class_confidences.append(out_ext_conficences)
    #         predict_ext_text.append(self.tokenizer.decode(out_ext_indexes))
    #         predict_boxes.append(out_boxes)

    #     predict_text = list(map(lambda x : x.replace('SOS','').replace('EOS',''),predict_text))
    #     predict_ext_text = list(map(lambda x : x.replace('SOS','').replace('EOS',''),predict_ext_text))
        
    #     return predict_classes, predict_class_confidences, predict_text, \
    #         predict_ext_classes, predict_ext_class_confidences, predict_ext_text, \
    #         predict_boxes
    
    def sequence_length_penalty(self, length, alpha=0.6):
        return ((5 + length) / 6) ** alpha

    def beam_search_decode_batched(self, model, memory, mask, tokenizer, beam_width, topk, max_text_length, device,
                                decoder_input_is_representative_chars, decoder_output_representative_chars):
        # print(beam_width, topk)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            
        # トークンのインデックス取得
        sos_token = tokenizer.representative_chars.index('SOS')
        eos_token = tokenizer.representative_chars.index('EOS')
        sos_ext_token = tokenizer.chars.index('SOS')
        eos_ext_token = tokenizer.chars.index('EOS')

        # 初期ビーム：各ビームは辞書で管理
        beams = [{
            "seq": [sos_token],
            "ext_seq": [sos_ext_token],
            "box_seq": [],
            "cur_seq": [[1.0, 0.0]],  # 初期カーソル（例）
            "score": 0.0
        }]
        completed = []

        for step in range(max_text_length):
            # EOSに達していないビームだけを抽出
            active_beams = [beam for beam in beams if beam["ext_seq"][-1] != eos_ext_token]
            if len(active_beams) == 0:
                break


            # バッチ処理のため、全ビームの ext_seq は同じ長さ (step+1) となるので、テンソルに変換
            if decoder_input_is_representative_chars:
                input_seqs = [beam["seq"] for beam in active_beams]
            else:
                input_seqs = [beam["ext_seq"] for beam in active_beams]
            trg_tensor = torch.tensor(input_seqs, dtype=torch.long, device=device)  # shape: (batch, seq_len)
            trg_tensor = trg_tensor.transpose(0, 1)  # shape: (seq_len, batch)

            # active_beams の数＝現在のバッチサイズ（beam 数）
            beam_batch_size = trg_tensor.size(1)
            # memory を各ビーム候補に対応させるために、バッチ次元を繰り返す
            memory_expanded = memory.repeat(1, beam_batch_size, 1)
            mask_expanded = mask.repeat(beam_batch_size, 1)

            # with torch.cuda.amp.autocast():
            with torch.amp.autocast('cuda'):
            # デコーダーへの入力（代表文字列か異体字列かで分岐）
                trg_emb = model.query_embed(trg_tensor)
                if model.__class__.__name__ == 'DHPDETR_EXT':
                    # 各ビームのカーソルシーケンスも同様にテンソルへ変換（shape: (batch, seq_len, 2)）
                    cur_seqs = [beam["cur_seq"] for beam in active_beams]
                    cur_tensor = torch.tensor(cur_seqs, dtype=torch.float, device=device)  # (batch, seq_len, 2)
                    cur_tensor = cur_tensor.transpose(0, 1)  # (seq_len, batch, 2)
                    cur_emb = model.cursor_embed(cur_tensor)
                    trg_emb = torch.cat([trg_emb, cur_emb], dim=2)
                trg_emb = model.query_pos(trg_emb)
                tgt_mask = model.generate_square_subsequent_mask(trg_tensor.size(0)).to(device)
                hs = model.transformer.decoder(trg_emb, memory_expanded, memory_key_padding_mask=mask_expanded, tgt_mask=tgt_mask)
                # hs = model.transformer.decoder(trg_emb, memory, memory_key_padding_mask=mask, tgt_mask=tgt_mask)
                hs = hs.transpose(0, 1)  # shape: (batch, seq_len, hidden_dim)
                
                # 異体字の出力を計算
                ext_logits = model.ext_class_embed(hs)  # shape: (batch, seq_len, vocab_size)
                last_logits = ext_logits[:, -1, :]      # shape: (batch, vocab_size)
                last_log_probs = torch.log_softmax(last_logits, dim=-1)

                # bbox出力の取得
                bbox_logits = model.bbox_embed(hs).sigmoid()  # (batch, seq_len, bbox_dim)
                last_boxes = bbox_logits[:, -1, :]              # (batch, bbox_dim)
                last_boxes_np = last_boxes.detach().cpu().numpy()

            new_beams = []
            # バッチ内の各ビーム候補に対して、全語彙に対する展開を行う
            for i, beam in enumerate(active_beams):
                beam_score = beam["score"]
                # 現在の last_log_probs は shape: (batch, vocab_size)
                # 各ビームで上位候補 (例：beam_width 以上) を一括取得する
                # topk = 10  # beam_width より少し多く取るなど調整可能
                topk_log_probs, topk_tokens = last_log_probs.topk(topk, dim=-1)
                for token, log_prob in zip(topk_tokens[i], topk_log_probs[i]):
                    # トークンごとの対数確率に長さペナルティを適用
                    token_score = log_prob.item() / self.sequence_length_penalty(step + 2)
                    new_score = beam_score + token_score
                    new_ext_seq = beam["ext_seq"] + [token]
                    new_seq = beam["seq"] + [tokenizer.representative_index(tokenizer.chars[token])]
                    new_box_seq = beam["box_seq"] + [last_boxes_np[i]]
                    # カーソル情報の更新（モデルが DHPDETR_EXT の場合）
                    new_cur_seq = beam["cur_seq"]
                    if model.__class__.__name__ == 'DHPDETR_EXT':
                        # 例：bbox の x 座標と y 座標＋高さの半分を使用
                        new_cur_seq = beam["cur_seq"] + [[last_boxes_np[i][0], last_boxes_np[i][1] + 0.5 * last_boxes_np[i][3]]]
                    new_beams.append({
                        "seq": new_seq,
                        "ext_seq": new_ext_seq,
                        "box_seq": new_box_seq,
                        "cur_seq": new_cur_seq,
                        "score": new_score
                    })


                # for token in range(last_log_probs.size(-1)):
                #     # トークンごとの対数確率に長さペナルティを適用
                #     token_score = last_log_probs[i, token].item() / sequence_length_penalty(step + 2)
                #     new_score = beam_score + token_score
                #     new_ext_seq = beam["ext_seq"] + [token]
                #     new_seq = beam["seq"] + [tokenizer.representative_index(tokenizer.chars[token])]
                #     new_box_seq = beam["box_seq"] + [last_boxes_np[i]]
                #     # カーソル情報の更新（モデルが DHPDETR_EXT の場合）
                #     new_cur_seq = beam["cur_seq"]
                #     if model.__class__.__name__ == 'DHPDETR_EXT':
                #         # 例：bbox の x 座標と y 座標＋高さの半分を使用
                #         new_cur_seq = beam["cur_seq"] + [[last_boxes_np[i][0], last_boxes_np[i][1] + 0.5 * last_boxes_np[i][3]]]
                #     new_beams.append({
                #         "seq": new_seq,
                #         "ext_seq": new_ext_seq,
                #         "box_seq": new_box_seq,
                #         "cur_seq": new_cur_seq,
                #         "score": new_score
                #     })

            # スコアが高い上位 beam_width 件を選択
            new_beams = sorted(new_beams, key=lambda b: b["score"], reverse=True)[:beam_width]

            # EOSに到達したビームは completed に、未到達のものは次回の展開用に beams に格納
            beams = []
            for beam in new_beams:
                if beam["ext_seq"][-1] == eos_ext_token:
                    completed.append(beam)
                else:
                    beams.append(beam)

            if len(beams) == 0:
                break

        # 完了候補があれば、その中で最もスコアが高いものを選択
        if len(completed) > 0:
            best_beam = max(completed, key=lambda b: b["score"])
        else:
            best_beam = beams[0]

        return best_beam["seq"], best_beam["ext_seq"], best_beam["box_seq"], best_beam["cur_seq"], best_beam["score"]

    def add_bboxes_to_pilimage(self, img, bboxes):
        w, h = img.size
        new_img = img.copy()
        draw=ImageDraw.Draw(new_img)
        
        for bb in bboxes:
            x0, y0 = int((bb[0]-bb[2]/2)*w), int((bb[1]-bb[3]/2)*h)
            x1, y1 = int((bb[0]+bb[2]/2)*w), int((bb[1]+bb[3]/2)*h)
            draw.rectangle([(x0,y0),(x1,y1)],outline="blue")

        return new_img


    def inference_one(self, pil_image, pil_bbox):
        cropped_image = pil_image.crop( pil_bbox )
        
        cw, ch = cropped_image.size
        cropped_image.save('./data/images/cropped_image.jpg')
        resized_cropped_image =  cropped_image.resize((self.w, int(ch*self.w/cw)))
        resized_cropped_image.save('./data/images/resized_cropped_image.jpg')
        
        predict_classes, predict_class_confidences, predict_text, \
            predict_ext_classes, predict_ext_class_confidences, predict_ext_text, \
            predict_boxes  = self.inference(self.model,
                            [resized_cropped_image], 
                            self.device, 
                            self.max_text_length)

        # predict_boxes is list of [center_x, ceneter_y, width, height] 
        bbox_image = self.add_bboxes_to_pilimage(cropped_image, predict_boxes[0])
        bbox_image.save('./data/images/bbox_image.jpg')

        if self.args.decoder_output_representative_chars:
            return predict_classes[0], predict_class_confidences[0], predict_text[0],\
                predict_ext_classes[0], predict_ext_class_confidences[0], predict_ext_text[0], predict_boxes[0]
        else:    
            return predict_ext_classes[0], predict_ext_class_confidences[0], predict_ext_text[0], predict_boxes[0]


