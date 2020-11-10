# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer, AutoModel
from fairseq.data.encoders import register_tokenizer

def get_sentence(tokens):
    sentence = []
    length = len(tokens)
    i=0
    while(i < length):
        index = [i]
        index = find_pound_key(tokens, i, index)
        i = index[-1]+1
        word = [tokens[j].strip("##") for j in index]
        word = "".join(word)
        sentence.append(word)
    return sentence

def find_pound_key(tokens, i, index):
    if i == len(tokens)-1:
        return index
    if "##" not in tokens[i+1]:
        return index
    if "##" in tokens[i+1]:
        index.append(i+1)
        return find_pound_key(tokens, i+1, index)

@register_tokenizer("transformer_tokenizer")
class TransformerTokenizer(object):
    def __init__(self, *unused):
        self.tokenizer = AutoTokenizer.from_pretrained("lanwuwei/GigaBERT-v4-Arabic-and-English")

    def encode(self, x: str) -> str:
    	toks = self.tokenizer.tokenize(x)
        return " ".join(toks)

    def decode(self, x: str) -> str:
    	x = get_sentence(x.split(" "))
        return " ".join(x)
