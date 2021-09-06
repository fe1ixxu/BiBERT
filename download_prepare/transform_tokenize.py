from transformers import AutoTokenizer, AutoModel, BertTokenizer
from sacremoses import MosesTokenizer, MosesDetokenizer

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input file')
    parser.add_argument('--output', type=str, required=True, help='output file')
    #parser.add_argument('--lang', type=str, required=True, help='language')
    parser.add_argument('--pretrained_model', type=str, required=True, help='pretrained language model')  
    args = parser.parse_args()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    except:
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    fo = open(args.input, encoding="utf-8")
    fw = open(args.output, "w", encoding="utf-8")
    #detok = MosesDetokenizer(args.lang)
    line = fo.readline()
    while(line):
        line = line.strip()
        #line = detok.detokenize(line.split())
        toks = tokenizer.tokenize(line)
        toks = " ".join(toks)
        fw.writelines([toks, "\n"])
        line = fo.readline()

    fo.close()
    fw.close()

if __name__ == "__main__":
  main()
