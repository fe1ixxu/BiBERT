# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass


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
        if cfg.pretrained_bpe:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_bpe)
        else:
            vocab_file_name = (
                "bert-base-cased" if cfg.bpe_cased else "bert-base-uncased"
            )
            self.bert_tokenizer = AutoTokenizer.from_pretrained(vocab_file_name)

    def encode(self, x: str) -> str:
        return " ".join(self.bert_tokenizer.tokenize(x))

    def decode(self, x: str) -> str:
        return self.bert_tokenizer.convert_tokens_to_string(x.split(" "))
        # return self.bert_tokenizer.clean_up_tokenization(
        #     self.bert_tokenizer.convert_tokens_to_string(x.split(" "))
        # )

    def is_beginning_of_word(self, x: str) -> bool:
        return not x.startswith("##")
