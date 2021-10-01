
from tokenizers import ByteLevelBPETokenizer
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str,required=True)
parser.add_argument("--size", type=int,required=True)
parser.add_argument("--output", type=str,required=True)
args = parser.parse_args()

tokenizer = BertWordPieceTokenizer(lowercase=False)
tokenizer.train(files=args.data, vocab_size=args.size, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>"
])
tokenizer.save_model(args.output) 

