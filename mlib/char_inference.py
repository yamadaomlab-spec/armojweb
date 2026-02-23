import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import pickle

import numpy as np
import timm


def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x)/u

class CharInferenceWapper():
    def __init__(self, model_basename):
        super(CharInferenceWapper, self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
    
        # model_names = ['full_ViT_addRandAffine_30.pth','full_ViT_addRandAffine_test_33.pth' 
        #             ]

        # 学習データのカテゴリの辞書をゲット　データ数＿文字　というラベル名になっているので、認識結果の出力時に最後の１文字を返すようにしています。
        with open('./data/dicts/c2i_dict.pkl', 'rb') as f:
            c2i_dict = pickle.load(f)

        self.cat_list = list(c2i_dict.keys())
        
        print(len(c2i_dict))
        
        self.transform_val = transforms.Compose(
            [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],)
            ]
        )

        n_cat= len(c2i_dict)

        # カテゴリ数を指定して元となるモデルを用意
        self.model = timm.create_model('vit_relpos_base_patch16_gapcls_224', pretrained=False, num_classes=n_cat)

        # 学習済みの重みをロード　モデルを選択できるようにしています。
        # weight_name = model_names[1]
        weight_name = model_basename

        self.model.load_state_dict(torch.load('./data/modelC/'+weight_name, map_location=torch.device('cpu')))
        # self.model.load_state_dict(torch.load('./data/modelC/'+weight_name))
        self.model = self.model.to(self.device)
        self.model.eval()

    def resize_bbox(self, sx, sy, img_w, img_h, bbox):
        left, upper, right, lower = bbox
        bw = right - left
        bh = lower - upper

        new_left = left - int(sx*bw)
        new_left = new_left if new_left >= 0 else 0
        new_right = right + int(sx*bw)
        new_right = new_right if new_right < img_w else img_w -1 
        new_upper = upper - int(sy*bh)
        new_upper = new_upper if new_upper >= 0 else 0
        new_lower = lower + int(sy*bh)
        new_lower = new_lower if new_lower < img_h else img_h -1 

        return (new_left, new_upper, new_right, new_lower)

    def add_bbox_to_pilimage(self, img, bbox):
        new_img = img.copy()
        draw=ImageDraw.Draw(new_img)
        
        x0, y0, x1, y1 = bbox
        draw.rectangle([(x0,y0),(x1,y1)],outline="blue")

        return new_img

    def inference_one(self, pil_image, pil_bbox):

        # アプリ表示用画像範囲の切り出し
        bbox_image = self.add_bbox_to_pilimage(pil_image, pil_bbox)
        img_w, img_h = pil_image.size
        bbox_image = bbox_image.crop(self.resize_bbox( 0.25, 1.5, img_w, img_h, pil_bbox ))
        bbox_image.save('./data/images/bbox_image.jpg')        
        
        # モデル入力用画像範囲の切り出し
        cropped_image = pil_image.crop( pil_bbox )
        cropped_image.save('./data/images/cropped_image.jpg')
        

        # バリデーション用の変換をかけておく
        t_image = self.transform_val(cropped_image)

        # 上のままだとバッチ分の次元が足りないので、最初に足しておく（batch, channel, height, width)
        t_image = t_image.view(1,3,224,224)
    
        max_k=10  # top-k の最大値
        
        with torch.no_grad():
            output = self.model(t_image.to(self.device))

            # topkの値とインデックスを求める
            val, indices = output.topk(max_k)
            # 今使っているモデルが最終層にsoftmaxがかかっていないので自前で書ける
            confidence = softmax(np.squeeze(output.cpu().detach().numpy()))*100
            
            out_cat = []
            out_conf = []
            for i in range(max_k):
                index = indices[0][i].cpu()
                out_cat.append(self.cat_list[index][-1:])
                out_conf.append(confidence[index]/100)
                # print(self.cat_list[index][-1:], f'{confidence[index]:6.2f}', '%')

        return out_cat, out_conf
        
    