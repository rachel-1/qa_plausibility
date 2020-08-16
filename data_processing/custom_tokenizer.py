import re
from emoji import *
import string 
import Levenshtein as lev
import pandas as pd

EMOJI_SET = set()

for emoji in UNICODE_EMOJI:
    EMOJI_SET.add(emoji)

def split_emojis(s):
    """
    Split emojis in a sentence into separate tokens. Note that some complex emojis are still mangled.
    """
    tokens = []
    token = ""
    for letter in s:
        if letter in EMOJI_SET:
            if token != "": 
                tokens.append(token)
                token = ""
            tokens.append(letter)
        else:
            token += letter
    if token != "": tokens.append(token)
    return tokens

def basic_tokenize(s):
    """
    Remove stop words, skip links and split on punctuation.
    """
    s = s.lower()
    # replace certain punctuation w/ space
    for substr in [":","'",",", "(", '"', ")", "’", "‘", "‼", "”", "？", "“", "‚"]:
        s = s.replace(substr, " ")

    tokens = s.strip().split()
    
    return tokens

def question_tokenize(s):
    if(isinstance(s, list)): return(s)
    
     # replace certain punctuation w/ space
    for substr in ["?", "!"]:
        s = s.replace(substr, " "+substr)
    
    tokens = basic_tokenize(s)
    # remove introductory exclaimation if present
    try:
        index = tokens.index('!')
        tokens = tokens[index+1:]
    except ValueError:
        pass
        
    return tokens

def response_tokenize(s):
    """
    Split to preserve emojis as their own tokens.
    """
    if(isinstance(s, list)): return(s)
    tokens = basic_tokenize(s)
    
    # remove special cases
    cleaned_tokens = []
    for token in tokens:
        if "http" in token or "www" in token: continue
        if "@" in token: continue
        new_token = token.translate(str.maketrans('', '', string.punctuation))
        if new_token != "":
            cleaned_tokens.append(new_token)
    tokens = cleaned_tokens
    
    true_tokens = []
    for token in tokens:
        true_tokens.extend(split_emojis(token))
    return [token for token in true_tokens if token != '']

def find_start_end(response_tokens, answer):
    assert(type(response_tokens) == list)
    if pd.isnull(answer) or answer == "not possible": return None

    answer_tokens = response_tokenize(answer)
    answer_length = len(answer_tokens)
    if answer_length == 0: return None
    start_token = answer_tokens[0]
    if start_token in ['a', 'an', 'the', 'in', 'on']:
        if len(answer_tokens) == 1: return None
        start_token = answer_tokens[1]
        answer_length -= 1
        
    def find_token(token_list, target_token, threshold=0.9):
        for idx, token in enumerate(token_list):
            # Note: tried the partial_ratio from fuzzywuzzy package 
            # but it gives false positives for extremely short strings
            similarity = lev.ratio(token, target_token)
            #print("{} vs {}: {}".format(token, target_token, similarity))
            if similarity >= threshold:
                return idx  
        return None

    start_idx = find_token(response_tokens, start_token)
    if start_idx is None:
        return None
    if answer_length == 1: return start_idx, start_idx
    end_idx = find_token(response_tokens[start_idx+1:], answer_tokens[-1])         
    if end_idx is None: 
        return None
    end_idx += start_idx + 1 # reference relative to start of string
    
    return start_idx, end_idx

def extract_answer(response_tokens, span):
    if span is None: return None
    return response_tokens[span[0]:span[1]+1]