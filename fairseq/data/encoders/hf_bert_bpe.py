# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass

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


@dataclass
class BertBPEConfig(FairseqDataclass):
    bpe_cased: bool = field(default=False, metadata={"help": "set for cased BPE"})
    pretrained_bpe: Optional[str] = field(
        default=None, metadata={"help": "pre-trained bpe model"}
    )


@register_bpe("bert", dataclass=BertBPEConfig)
class BertBPE(object):
    def __init__(self, cfg):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers with: pip install transformers"
            )
        self.pretrained_bpe = cfg.pretrained_bpe
        if cfg.pretrained_bpe:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_bpe)
            if "de" in cfg.pretrained_bpe:
                self.t = AutoTokenizer.from_pretrained("Geotrend/bert-base-en-de-cased")
            else:
                self.t = AutoTokenizer.from_pretrained("Geotrend/bert-base-en-fr-cased")
        else:
            vocab_file_name = (
                "bert-base-cased" if cfg.bpe_cased else "bert-base-uncased"
            )
            self.bert_tokenizer = AutoTokenizer.from_pretrained(vocab_file_name)

    def encode(self, x: str) -> str:
        return " ".join(self.bert_tokenizer.tokenize(x))

    def decode(self, x: str) -> str:
        if self.pretrained_bpe in ["Geotrend/bert-base-en-de-cased", "Geotrend/bert-base-en-fr-cased", "../models/lang-model/wp_models/"] or "wordpiece" in self.pretrained_bpe:
            return " ".join(get_sentence(x.split(" ")))
        else:
            tokens = self.bert_tokenizer.convert_tokens_to_string(x.split(" "))
            return " ".join(get_sentence(self.t.tokenize(tokens)))
        # return self.bert_tokenizer.clean_up_tokenization(
        #     self.bert_tokenizer.convert_tokens_to_string(x.split(" "))
        # )

    def is_beginning_of_word(self, x: str) -> bool:
        return not x.startswith("##")
