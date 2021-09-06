## download IWSLT'14 dataset from fairseq
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

## train 8K tokenizer for ordinary translation:
cat data_demose/train.en data_demose/valid.en data_demose/test.en | shuf > data_demose/train.all
mkdir 8k-vocab-models
python vocab_trainer.py --data data_demose/train.all --size 8000 --output 8k-vocab-models

## train 12K tokenizer for dual-directional translation
cat data_demose/train.en data_demose/valid.en data_demose/test.en data_demose/train.de data_demose/valid.de data_demose/test.de | shuf > data_demose/train.all.dual
mkdir 12k-vocab-models
python vocab_trainer.py --data data_demose/train.all.dual --size 12000 --output 12k-vocab-models



## tokenize translation data
mkdir bibert_tok
mkdir 8k_tok
mkdir 12k_tok

for prefix in "valid" "test" "train" ;
do
    for lang in "en" "de" ;
    do
        python transform_tokenize.py --input data_demose/${prefix}.${lang} --output bibert_tok/${prefix}.${lang} --pretrained_model Coran/bibert-ende
    done
done

for prefix in "valid" "test" "train" ;
do
    python transform_tokenize.py --input data_demose/${prefix}.en --output 8k_tok/${prefix}.en --pretrained_model 8k-vocab-models
done


for prefix in "valid" "test" "train" ;
do
    for lang in "en" "de";
    do
    python transform_tokenize.py --input data_demose/${prefix}.${lang} --output 12k_tok/${prefix}.${lang} --pretrained_model 12k-vocab-models
    done
done


mkdir data   # for one-way translation data
cp bibert_tok/*.de data/ 
cp 8k_tok/*.en data/

mkdir data_mixed_ft # for dual-directional fine-tuning data. we first preprocess this because it will be easier to finish
cp bibert_tok/*.de data_mixed_ft/
cp 12k_tok/*.en data_mixed_ft/

mkdir data_mixed # preprocess dual-directional data

cd data_mixed
cat ../bibert_tok/train.en ../bibert_tok/train.de > train.all.en
cat ../12k_tok/train.de ../12k_tok/train.en > train.all.de
paste -d '@@@' train.all.en /dev/null /dev/null train.all.de | shuf > train.all
cat train.all | awk -F'@@@' '{print $1}' > train.de
cat train.all | awk -F'@@@' '{print $2}' > train.en
rm train.all*

cat ../bibert_tok/valid.en ../bibert_tok/valid.de > valid.all.en
cat ../12k_tok/valid.de ../12k_tok/valid.en > valid.all.de
paste -d '@@@' valid.all.en /dev/null /dev/null valid.all.de | shuf > valid.all
cat valid.all | awk -F'@@@' '{print $1}' > valid.de
cat valid.all | awk -F'@@@' '{print $2}' > valid.en
rm valid.all*

cp ../bibert_tok/test.de .
cp ../12k_tok/test.en .
cd ..




## get src and tgt vocabulary
python get_vocab.py --tokenizer Coran/bibert-ende --output data/src_vocab.txt
python get_vocab.py --tokenizer Coran/bibert-ende --output data_mixed/src_vocab.txt
python get_vocab.py --tokenizer Coran/bibert-ende --output data_mixed_ft/src_vocab.txt
python get_vocab.py --tokenizer 8k-vocab-models --output data/tgt_vocab.txt
python get_vocab.py --tokenizer 12k-vocab-models --output data_mixed/tgt_vocab.txt
python get_vocab.py --tokenizer 12k-vocab-models --output data_mixed_ft/tgt_vocab.txt


## fairseq preprocess
TEXT=data
fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25

TEXT=data_mixed
fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25

TEXT=data_mixed_ft
fairseq-preprocess --source-lang de --target-lang en  --trainpref $TEXT/train --validpref $TEXT/valid \
--testpref $TEXT/test --destdir ${TEXT}/de-en-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25

## remove useless files
rm -rf data_desubnmt
rm -rf data_demose
rm -rf iwslt14.tokenized.de-en
rm -rf orig
rm -rf subword-nmt
rm -rf mosesdecoder
rm -rf prepare-iwslt14.sh
rm -rf bibert_tok
rm -rf 8k_tok
rm -rf 12k_tok









