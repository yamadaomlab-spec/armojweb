import sentencepiece as spm
# モデルの作成

class kindai_Tokenizer():
    def __init__(self, path):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(path)

    def get_vocab_size(self):
        return self.tokenizer.GetPieceSize()

    # ラベルを受け取りidを返す
    def label_2_id(self, label):
        return self.tokenizer.PieceToId(label)

    # idを受け取りラベルを返す
    def id_2_label(self, id):
        return self.tokenizer.IdToPiece(int(id))

    def encode_texts(self, datas):
        input_ids = self.tokenizer.EncodeAsIds(datas)
        return input_ids
    def decode_text(self, ids):
        text = self.tokenizer.DecodeIds(ids)
        return text
    def decode_as_piece(self, ids):
        text = self.tokenizer.decode(ids)
        piece = self.tokenizer.EncodeAsPieces(text)
        return piece