## Download IWSLT'14 dataset from fairseq
wget https://raw.githubusercontent.com/pytorch/fairseq/master/examples/translation/prepare-iwslt14.sh
bash prepare-iwslt14.sh

## de-subnmt data
mkdir data_desubnmt
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/train.en > data_desubnmt/train.en
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/train.de > data_desubnmt/train.de
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/valid.en > data_desubnmt/valid.en
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/valid.de > data_desubnmt/valid.de
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/test.en > data_desubnmt/test.en
sed -r 's/(@@ )|(@@ ?$)//g' iwslt14.tokenized.de-en/test.de > data_desubnmt/test.de

## de-mose data
mkdir data_demose
perl mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/train.en > data_demose/train.en
perl mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/train.de > data_demose/train.de
perl mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/valid.en > data_demose/valid.en
perl mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/valid.de > data_demose/valid.de
perl mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/test.en > data_demose/test.en
perl mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < data_desubnmt/test.de > data_demose/test.de

## train 8k tokenizer:
cat data_demose/train.en data_demose/valid.en data_demose/test.en | shuf > data_demose/train.all
mkdir 8k-vocab-models
python vocab_trainer.py --data data_demose/train.all --size 8000 --output 8k-vocab-models


## tokenize
mkdir data
for prefix in "valid" "test" "train" ;
do
    python transform_tokenize.py --input data_demose/${prefix}.de --output data/${prefix}.de --pretrained_model Coran/bibert-ende
done

for prefix in "valid" "test" "train" ;
do
    python transform_tokenize.py --input data_demose/${prefix}.en --output data/${prefix}.en --pretrained_model 8k-vocab-models
done


## get src and tgt vocabulary
python get_vocab.py --tokenizer Coran/bibert-ende --output data/src_vocab.txt
python get_vocab.py --tokenizer 8k-vocab-models --output data/tgt_vocab.txt

## fairseq preprocess
TEXT=data
fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25

## remove useless files
rm -rf data_*
rm -rf iwslt14.tokenized.de-en
rm -rf orig
rm -rf subword-nmt
rm -rf mosesdecoder
rm -rf prepare-iwslt14.sh









