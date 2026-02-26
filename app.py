from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
import os
# import pytesseract

from ylib.line_inference import LineInferenceWapper
from mlib.char_inference import CharInferenceWapper
from ultralytics import YOLO
import cv2
import json
from threading import Thread
from pdf2image import convert_from_path # PDFページを画像に変換
import numpy as np
import torch

class Armoj():
    def __init__(self):
        # GPU推論の再現性を確保
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        with open('./data/setting.json') as f:
            self.setting = json.load(f)

        self.modelL_filebasename = self.setting['modelL']
        self.modelC_filebasename = self.setting['modelC']

        # 行認識クラスインスタンス
        self.lineTranscriptor = LineInferenceWapper(
            self.setting['modelL'],
            charset_path=self.setting.get('dictL_charset'),
            itaiji_path=self.setting.get('dictL_itaiji')
        )
        # 文字認識クラスインスタンス
        self.charTranscriptor = CharInferenceWapper(
            self.setting['modelC'],
            dict_path=self.setting.get('dictC')
        )
        # YOLOインスタンス
        self.yolo = YOLO(self.setting.get('modelY', './data/modelY/best.pt'))
        # self.yolo = YOLO('./data/modelY/last.pt')
        self.yolo_xyxy_bboxes = []
        self.maxheight = 768
        self.imagescale = 1.0

    # 10.17
    def process_image(self, pil_image):
        # 画像をリサイズしてOCRにかける
        pil_image = self.image_resize(pil_image)
        self.lineTranscriptor.inference_one(pil_image)  # 行認識
        self.charTranscriptor.inference_one(pil_image)  # 文字認識
        return "認識完了"
    
    #----------------
    # Image load with Drag&Drop
    #----------------
    # 20240809
    def image_resize(self, pil_image):
        self.imagescale = 1.0
        w, h = pil_image.size
        if h > self.maxheight:
            self.imagescale = self.maxheight/h
            w = int(w*self.imagescale)
            pil_image = pil_image.resize((w, self.maxheight), Image.BICUBIC)
        
        return pil_image
    
    # 10.17
    def process_pdf(self, pdf_path):
        # PDFを画像に変換してページごとに処理
        pages = convert_from_path(pdf_path)
        results = []
        for page in pages:
            page_text = self.process_image(page)
            results.append(page_text)
        return "\n".join(results)

    # 個別文字認識結果の表示用テキスト
    def char_inference_result_to_text(self, cls, confidence = None):
        vtext = ''
        if confidence is not None:
            i = 0
            for c, cfd in zip(cls, confidence):
                vtext += f'第{i}候補： {c}  {cfd:6.1%}\n'
                # vtext += f'第{i}候補： {c}  ({cfd:6.1%})\n'
                i += 1
        else:
            for i, c in enumerate(cls):
                vtext += f'第{i}候補： {c}\n'
            
        return vtext, len(cls)
    
    def _lineDetection(self, pil_image, conf=0.25):
        # 毎回同じ結果を得るためシードをリセット
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)

        results = self.yolo(pil_image, conf=conf)
        # YOLOの出力から行のバウンディングボックスを取得し、boxesのy座標を2だけ増やす
        boxes = results[0].boxes.xyxy
        offset = torch.tensor([0, 2, 0, 0], device=boxes.device)
        yolo_xyxy_bboxes = (results[0].boxes.xyxy + offset).tolist()  # y座標を1だけ増やす 2026.2.23
        # yolo_xyxy_bboxes = results[0].boxes.xyxy.tolist()
        pbboxes = yolo_xyxy_bboxes
        pbb = pbboxes
        def my_func(b):
            return b[0] + b[2]
        
        pbb.sort(key=my_func,reverse=True)
        # for b in pbb:
        #     print(my_func(b))

        total = len(yolo_xyxy_bboxes)
        multiline_text = ""
        multiline_cbboxes = [] #10.17
        multiline_lbboxes = [] #10.17
 
        for i, pil_bbox in enumerate(yolo_xyxy_bboxes):
            cls, confidence, text, bboxes = self.lineTranscriptor.inference_one(pil_image, pil_bbox)
            multiline_text += text + '\n'
            multiline_cbboxes.append(bboxes) #10.17 一個一個の文字の四角形
            multiline_lbboxes.append(pil_bbox) #10.17 yoloの行単位の四角形

        # このスレッド中でメインメソッドのレンダーを呼び出す
        self.res_text = multiline_text
        self.res_cbboxes = multiline_cbboxes #10.17
        self.res_lbboxes = multiline_lbboxes #10.17
        
    def LineDetection(self, pil_image, conf=0.25, user_id=None):
        # t = Thread(target = self._lineDetection, args=(pil_image,))
        # t.start()
        self._lineDetection(pil_image, conf=conf)
        return

armoj = Armoj()

app = Flask(__name__)



# 画像ディレクトリのパス
IMAGE_FOLDER = os.path.join(app.root_path, 'static/images')


# ウェブページを表示するルート
@app.route('/')
# def index():
def home():
    return render_template('index.html', result=None)

# 新しい画像リストページのルート
@app.route('/images')
def images():
    images = os.listdir(IMAGE_FOLDER)  # static/imagesディレクトリの画像ファイルを取得
    return render_template('images.html', images=images)

# 個別の画像ファイルを表示するルート
@app.route('/images/<filename>')
def image_page(filename):
    return render_template('image_page.html', filename=filename)

# 画像ファイルをダウンロード可能にするルート
@app.route('/download/<filename>')
def download_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename, as_attachment=True)


# char OCR
@app.route('/run_charocr', methods=['POST'])
def run_charocr():
    try:
        scaleFactor = float(request.form.get('scaleFactor'))
        x = int(request.form.get('x'))
        y = int(request.form.get('y'))
        width = int(request.form.get('width'))
        height = int(request.form.get('height'))

        file = request.files.get('file')
        if not file:
            return jsonify({'status': 'error', 'message': 'No image file provided'}), 400

        # ストリームの先頭に戻す
        file.stream.seek(0)

        try:
            img = Image.open(file)
            if img.size == (0, 0):
                raise ValueError("Image is empty.")

            img = img.convert("RGB")  # JPEGに保存するならRGBに変換
            
        except Exception as e:
            print("Failed to load image:", e)
            return jsonify({'status': 'error', 'message': f'Failed to load image: {e}'}), 400
        
        img = armoj.image_resize(img)
        rs = armoj.imagescale

        x = int(x * rs)
        y = int(y * rs)
        width = int(width * rs)
        height = int(height * rs)

        cls, confidence = armoj.charTranscriptor.inference_one(img, (x, y, x + width, y + height))
        text_result, _ = armoj.char_inference_result_to_text(cls, confidence)
        return jsonify({'status': 'success', 'ocr_text': text_result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


# line OCR
@app.route('/run_lineocr', methods=['POST'])
def run_lineocr():
    try:
        accuracy = request.form.get('accuracy', 'standard')
        armoj.lineTranscriptor.accuracy_mode = accuracy

        # 選択範囲のパラメータ取得
        scaleFactor = float(request.form.get('scaleFactor'))
        x = int(request.form.get('x'))
        y = int(request.form.get('y'))
        width = int(request.form.get('width'))
        height = int(request.form.get('height'))

        file = request.files.get('file')
        if not file:
            return jsonify({'status': 'error', 'message': 'No image file provided'}), 400

        # ストリームの先頭に戻す
        file.stream.seek(0)

        # 画像を読み込む（回転済みの画像）
        # PILで開く
        try:
            img = Image.open(file)
            if img.size == (0, 0):
                raise ValueError("Image is empty.")

            img = img.convert("RGB")  # JPEGに保存するならRGBに変換
        except Exception as e:
            print("Failed to load image:", e)
            return jsonify({'status': 'error', 'message': f'Failed to load image: {e}'}), 400

        # OCR用にリサイズ
        img = armoj.image_resize(img)
        rs = armoj.imagescale

        # canvas座標を画像座標に変換
        x = int(x * rs)
        y = int(y * rs)
        width = int(width * rs)
        height = int(height * rs)

        _, _, text_result, boxes = armoj.lineTranscriptor.inference_one(img, (x, y, x + width, y + height))

        return jsonify({'status': 'success', 
                        'ocr_text': text_result
                        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


# page OCR
@app.route('/run_pageocr', methods=['POST'])
def run_pageocr():
    try:
        accuracy = request.form.get('accuracy', 'standard')
        page_number = int(request.form.get('page', '1'))
        yolo_conf = float(request.form.get('yolo_conf', '0.25'))
        armoj.lineTranscriptor.accuracy_mode = accuracy
        print("OCR accuracy mode:", accuracy, "YOLO conf:", yolo_conf)

        # 保存先ディレクトリが存在しない場合は作成
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # ファイルが送信されているか確認
        file = request.files.get('file')
        if not file:
            return jsonify({'status': 'error', 'message': 'No image file provided'}), 400
        
        # ストリームの先頭に戻す
        file.stream.seek(0)

        if file.mimetype.startswith('image/'):  # ← canvas からの画像（image/png など）
            # PILで開く
            try:
                img = Image.open(file)
                if img.size == (0, 0):
                    raise ValueError("Image is empty.")

                img = img.convert("RGB")  # JPEGに保存するならRGBに変換
            except Exception as e:
                print("Failed to load image:", e)
                return jsonify({'status': 'error', 'message': f'Failed to load image: {e}'}), 400

        else:
            return jsonify({'status': 'error', 'message': 'Unsupported file format'}), 400


        img = armoj.image_resize(img)
        rs = armoj.imagescale

        # 選択範囲パラメータの取得（オプション）
        sel_x = request.form.get('sel_x')
        sel_y = request.form.get('sel_y')
        sel_width = request.form.get('sel_width')
        sel_height = request.form.get('sel_height')

        crop_offset_x = 0
        crop_offset_y = 0

        if sel_x is not None and sel_y is not None and sel_width is not None and sel_height is not None:
            # canvas座標を画像座標に変換
            cx = int(int(sel_x) * rs)
            cy = int(int(sel_y) * rs)
            cw = int(int(sel_width) * rs)
            ch = int(int(sel_height) * rs)

            # 画像範囲内にクランプ
            iw, ih = img.size
            cx = max(0, min(cx, iw))
            cy = max(0, min(cy, ih))
            cw = min(cw, iw - cx)
            ch = min(ch, ih - cy)

            if cw > 0 and ch > 0:
                img = img.crop((cx, cy, cx + cw, cy + ch))
                crop_offset_x = cx
                crop_offset_y = cy
                print(f"Cropped to selection: x={cx}, y={cy}, w={cw}, h={ch}")

        armoj.LineDetection(img, conf=yolo_conf)

        print(armoj.res_text)
        print(armoj.res_lbboxes)

        # バウンディングボックス座標にクロップオフセットを加算し、canvas座標系に戻す
        lbboxes_jsonable = [
            (int((bbox[0] + crop_offset_x) / rs), int((bbox[1] + crop_offset_y) / rs),
             int((bbox[2] + crop_offset_x) / rs), int((bbox[3] + crop_offset_y) / rs))
            for bbox in armoj.res_lbboxes
        ]
        cbboxes_jsonable = [
            [(int((box[0] + crop_offset_x) / rs), int((box[1] + crop_offset_y) / rs),
              int((box[2] + crop_offset_x) / rs), int((box[3] + crop_offset_y) / rs))
             for box in line]
            for line in armoj.res_cbboxes
        ]

        return jsonify({
            'status': 'success',
            'ocr_text': armoj.res_text,
            'lbboxes': lbboxes_jsonable,
            'cbboxes': cbboxes_jsonable
        })
    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'message': str(e)}), 400
    
   
# 追加
@app.route('/submit', methods=['GET','POST'])
def submit():
    if request.method == 'POST': 
        user_input = request.form['textbox']
        # テキストをファイルに保存
        with open('input_text_file.txt', 'a', encoding='utf-8') as file:
            input = file.write(user_input + '\n')  # テキストを1行ごとに追加 10.24
        return f"あなたの入力: {input}"  # 10.24
    return render_template('index.html', result=user_input)

if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument('--port', type=int, default=11111)
    _args = _parser.parse_args()
    app.run(host='0.0.0.0', port=_args.port)
