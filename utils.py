# utils.py
import random
import numpy as np
import torch
import pymorphy2
from nltk import ngrams
from sklearn.metrics.pairwise import cosine_similarity

morph = pymorphy2.MorphAnalyzer(lang='ru')

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tokenize(text: str):
    return text.lower().split()

def lemmatize(text: str):
    return [morph.parse(w)[0].normal_form for w in tokenize(text)]

def cosine_sim(a: list, b: list):
    if not a or not b:
        return 0.0
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return cosine_similarity(a, b)[0][0]

def char_ngram_overlap(src: str, tgt: str, n: int = 3):
    src_ng = set(''.join(g) for g in ngrams(src.lower(), n))
    tgt_ng = set(''.join(g) for g in ngrams(tgt.lower(), n))
    if not src_ng and not tgt_ng:
        return 0.0
    return len(src_ng & tgt_ng) / len(src_ng | tgt_ng)

def filter_pair(src: str, tgt: str, tokenizer):
    src_tok = len(tokenizer.encode(src, add_special_tokens=False))
    tgt_tok = len(tokenizer.encode(tgt, add_special_tokens=False))
    if src_tok < 10 or tgt_tok < 3:
        return False
    if tgt.strip() in src:
        return False
    if tgt_tok >= src_tok:
        return False
    if tgt_tok > src_tok // 2:
        return False

    return True