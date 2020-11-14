from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("lanwuwei/GigaBERT-v4-Arabic-and-English")
fo = open("/export/c12/haoranxu/fairseq/arabic_bitext/data_transbpe/dev.32k.en2", encoding="utf-8")
fw = open("/export/c12/haoranxu/fairseq/arabic_bitext/data_transbpe/dev.32k.en", "w", encoding="utf-8")

line = fo.readline()
while(line):
    toks = tokenizer.tokenize(line)
    toks = " ".join(toks)
    fw.writelines([toks, "\n"])
    line = fo.readline()

fo.close()
fw.close()