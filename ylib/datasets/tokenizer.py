import re
import html
import string
import numpy as np

"""
DeepSpell based text cleaning process.
    Tal Weiss.
    Deep Spelling.
    Medium: https://machinelearnings.co/deep-spelling-9ffef96a24f6#.2c9pu8nlm
    Github: https://github.com/MajorTal/DeepSpell
"""

RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(
    chr(768), chr(769), chr(832), chr(833), chr(2387),
    chr(5151), chr(5152), chr(65344), chr(8242)), re.UNICODE)
RE_RESERVED_CHAR_FILTER = re.compile(r'[¶¤«»]', re.UNICODE)
RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
RE_BASIC_CLEANER = re.compile(r'[^\w\s{}]'.format(re.escape(string.punctuation)), re.UNICODE)

LEFT_PUNCTUATION_FILTER = """!%&),.:;<=>?@\\]^_`|}~"""
RIGHT_PUNCTUATION_FILTER = """"(/<=>@[\\^_`{|~"""
NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE)


def text_standardize(text):
    """Organize/add spaces around punctuation marks"""

    if text is None:
        return ""

    text = html.unescape(text).replace("\\n", "").replace("\\t", "")

    text = RE_RESERVED_CHAR_FILTER.sub("", text)
    text = RE_DASH_FILTER.sub("-", text)
    text = RE_APOSTROPHE_FILTER.sub("'", text)
    text = RE_LEFT_PARENTH_FILTER.sub("(", text)
    text = RE_RIGHT_PARENTH_FILTER.sub(")", text)
    text = RE_BASIC_CLEANER.sub("", text)

    text = text.lstrip(LEFT_PUNCTUATION_FILTER)
    text = text.rstrip(RIGHT_PUNCTUATION_FILTER)
    text = text.translate(str.maketrans({c: f" {c} " for c in string.punctuation}))
    text = NORMALIZE_WHITESPACE_REGEX.sub(" ", text.strip())

    return text


class Tokenizer_ori():
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK,self.SOS,self.EOS = "¶", "¤", "SOS", "EOS"
        self.chars = [self.PAD_TK] + [self.UNK_TK ]+ [self.SOS] + [self.EOS] + chars
        self.PAD = self.chars.index(self.PAD_TK)
        self.UNK = self.chars.index(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""

        encoded = []

        text = ['SOS'] + text + ['EOS']
        for item in text:
            index = self.chars.index(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""
        
        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
#         decoded = text_standardize(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")


    def decode_as_list(self, text):
        """Decode vector to text"""
        
        decoded = [self.chars[int(x)] for x in text if x > 3]

        return decoded
    

class Tokenizer():
    """Manager tokens functions and charset/dictionary properties"""

    def __init__(self, chars, itaiji=None, max_text_length=128):
        self.PAD_TK, self.UNK_TK,self.SOS,self.EOS = "¶", "¤", "SOS", "EOS"
        self.chars = [self.PAD_TK] + [self.UNK_TK ]+ [self.SOS] + [self.EOS] + chars
        self.PAD = self.chars.index(self.PAD_TK)
        self.UNK = self.chars.index(self.UNK_TK)

        self.maxlen = max_text_length

        if itaiji is not None:
            self.make_representative_chars(itaiji)
        else:
            self.reprezentative_chars = self.chars

    # itaiji は 漢字k を代表漢字とする異体字の辞書dic_itai[k]=[k1,k2, ...]
    def make_representative_chars(self, itaiji):
        self.itaiji = itaiji
        self.itaiji_to_representative = {}
        if itaiji is not None:
            for k, v in itaiji.items():
                for vi in v:
                    self.itaiji_to_representative[vi] = k
        
        representative_chars = []
        for c in self.chars:
            if c in self.itaiji_to_representative.keys():
                k = self.itaiji_to_representative[c]
                if k not in representative_chars:
                    representative_chars.append(k)
            else:
                representative_chars.append(c)

        self.representative_chars = representative_chars 

        self.num_representative_chars = len(self.representative_chars)
        self.num_chars = len(self.chars)

    # def index(self, c):
    #     return self.chars.index(c)
    
    # def index_to_representative_index(self, idx):
    #     return self.representative_index(self.chars[idx])
    
    def representative_index(self, c):
        if c not in self.representative_chars:
            return self.representative_chars.index(self.itaiji_to_representative[c])
        else:
            return self.representative_chars.index(c)
        
    def encode(self, text):
        """Encode text to vector"""

        encoded = []

        text = ['SOS'] + text + ['EOS']
        for item in text:
            index = self.chars.index(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def representative_encode(self, text):
        """Encode text to vector"""

        encoded = []
        
        text = ['SOS'] + text + ['EOS']
        for item in text:
            index = self.representative_index(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)
    
    def decode(self, text):
        """Decode vector to text"""
        
        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
#         decoded = text_standardize(decoded)

        return decoded
    
    def representative_decode(self, text):
        """Decode vector to text"""
        
        decoded = "".join([self.representative_chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
#         decoded = text_standardize(decoded)

        return decoded
    
    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")


    def decode_as_list(self, text, from_sos=None, to_eos=None):
        """Decode vector to text"""
        
        if from_sos is None:
            decoded = [self.chars[int(x)] for x in text if x > 3]
        else:
            decoded = [self.chars[int(x)] for x in text[from_sos:to_eos]]
        return decoded


    def representative_decode_as_list(self, text):
        """Decode vector to text"""
        
        decoded = [self.representative_chars[int(x)] for x in text if x > 3]

        return decoded
    
    
# class Tokenizer():
#     """Manager tokens functions and charset/dictionary properties"""

#     def __init__(self, chars, max_text_length=128):
#         self.PAD_TK, self.SOS,self.EOS = "¶", "SOS", "EOS"
#         self.chars = [self.PAD_TK] + [self.UNK_TK ]+ [self.SOS] + [self.EOS] + chars
#         self.PAD = self.chars.index(self.PAD_TK)        
# #         self.PAD_TK, self.UNK_TK,self.SOS,self.EOS = "¶", "¤", "SOS", "EOS"
# #         self.chars = [self.PAD_TK] + [self.UNK_TK ]+ [self.SOS] + [self.EOS] + chars
# #         self.PAD = self.chars.index(self.PAD_TK)
# #         self.UNK = self.chars.index(self.UNK_TK)

#         self.vocab_size = len(self.chars)
#         self.maxlen = max_text_length

#     def encode(self, text):
#         """Encode text to vector"""

#         encoded = []

#         text = ['SOS'] + text + ['EOS']
#         for item in text:
#             index = self.chars.index(item)
# #             index = self.UNK if index == -1 else index
#             encoded.append(index)

#         return np.asarray(encoded)

#     def decode(self, text):
#         """Decode vector to text"""
        
#         decoded = "".join([self.chars[int(x)] for x in text if x > -1])
#         decoded = self.remove_tokens(decoded)
# #         decoded = text_standardize(decoded)

#         return decoded

#     def remove_tokens(self, text):
#         """Remove tokens (PAD) from text"""

#         return text.replace(self.PAD_TK, "")
# #         return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")


#     def decode_as_list(self, text):
#         """Decode vector to text"""
        
#         decoded = [self.chars[int(x)] for x in text if x > 3]

#         return decoded