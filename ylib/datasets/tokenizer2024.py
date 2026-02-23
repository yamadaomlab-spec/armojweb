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


class Tokenizer():
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