from transformers import AutoTokenizer, BertTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, required=True, help='path or name of the tokenizer')
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    try:
        t = AutoTokenizer.from_pretrained(args.tokenizer)
    except:
        t = BertTokenizer.from_pretrained(args.tokenizer)

    with open(args.output, "w", encoding="utf-8") as f:
        size = t.vocab_size
        for i in range(size):
            f.writelines([t.convert_ids_to_tokens(i), "\n"])

if __name__ == "__main__":
    main()
