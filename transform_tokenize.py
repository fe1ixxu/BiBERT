from transformers import AutoTokenizer, AutoModel
from sacremoses import MosesTokenizer, MosesDetokenizer

tokenizer = AutoTokenizer.from_pretrained("lanwuwei/GigaBERT-v4-Arabic-and-English")
fo = open("/export/c12/haoranxu/fairseq/arabic_bitext/data_temp/train.32k.ar", encoding="utf-8")
fw = open("/export/c12/haoranxu/fairseq/arabic_bitext/data_transbpe/train.32k.ar", "w", encoding="utf-8")
detok = MosesDetokenizer("en")
line = fo.readline()
while(line):
    # line = detok.detokenize(line.split()) # if en, do not need if ar
    toks = tokenizer.tokenize(line)
    toks = " ".join(toks)
    fw.writelines([toks, "\n"])
    line = fo.readline()

fo.close()
fw.close()


# fo = open("/export/c12/haoranxu/fairseq/arabic_bitext/data_test/vocab.txt", encoding="utf-8")
# fw = open("/export/c12/haoranxu/fairseq/arabic_bitext/data_test/vocab2.txt", "w", encoding="utf-8")
# lines = fo.readlines()
# num = len(lines)
# for line in lines:
#     line = line.strip()
#     fw.writelines([line, " ", str(num), "\n"])
#     num -= 1

# fo.close()
# fw.close()