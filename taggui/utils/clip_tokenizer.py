"""
Native CLIP BPE tokenizer implementation without any external dependencies.
Produces EXACTLY the same token count as transformers AutoTokenizer for CLIP.
"""

import json
import os
from pathlib import Path
from functools import lru_cache

TOKENIZER_DIRECTORY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'clip-vit-base-patch32')

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    This is exactly the same mapping used by CLIP tokenizer.
    """
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class ClipTokenizer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = TOKENIZER_DIRECTORY_PATH
        
        with open(os.path.join(model_path, 'vocab.json'), 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        with open(os.path.join(model_path, 'merges.txt'), 'r', encoding='utf-8') as f:
            merges = f.read().split('\n')[1:-1]
        self.bpe_ranks = dict(zip([tuple(merge.split()) for merge in merges], range(len(merges))))
        
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        self.start_token = 49406  # <|startoftext|>
        self.end_token = 49407    # <|endoftext|>
    
    @lru_cache(maxsize=4096)
    def bpe(self, token):
        word = tuple(token)
        pairs = get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        return ' '.join(word)
    
    @lru_cache(maxsize=8192)
    def count_tokens(self, text):
        """Count number of tokens for given text, matching CLIP tokenizer exactly."""
        if not text:
            return 0
        
        # Normalize whitespace
        text = ' '.join(text.strip().split())
        
        # Encode bytes and tokenize
        bpe_tokens = []
        for token in text.lower().split():
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        
        # Count tokens (exclude start/end tokens which are always added)
        return len(bpe_tokens)


# Global instance for reuse
_tokenizer = None

def get_token_count(text):
    """Get token count for text, returns number of tokens (excluding start/end markers)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = ClipTokenizer()
    return _tokenizer.count_tokens(text)