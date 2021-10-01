DATA_PATH=/export/c01/haoranxu/BiBERT/data/en-fa/
LM=/export/c01/haoranxu/LMs/EnFa-large-128K-v1.0/
## de-subnmt data
# No need to de-subnmt

## de-mose data
mkdir ${DATA_PATH}data_demose
perl ${HOME}/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < ${DATA_PATH}orig/train.en > ${DATA_PATH}data_demose/train.en
perl ${HOME}/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < ${DATA_PATH}orig/train.fa > ${DATA_PATH}data_demose/train.fa
perl ${HOME}/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < ${DATA_PATH}orig/dev.en > ${DATA_PATH}data_demose/dev.en
perl ${HOME}/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < ${DATA_PATH}orig/dev.fa > ${DATA_PATH}data_demose/dev.fa
perl ${HOME}/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < ${DATA_PATH}orig/test.en > ${DATA_PATH}data_demose/test.en
perl ${HOME}/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en -q < ${DATA_PATH}orig/test.fa > ${DATA_PATH}data_demose/test.fa

## train 52K tokenizer for dual-directional translation
cat ${DATA_PATH}data_demose/train.en ${DATA_PATH}data_demose/dev.en ${DATA_PATH}data_demose/test.en \
${DATA_PATH}data_demose/train.fa ${DATA_PATH}data_demose/dev.fa ${DATA_PATH}data_demose/test.fa | shuf \
> ${DATA_PATH}data_demose/train.all.dual

mkdir ${DATA_PATH}52k-vocab-models
python vocab_trainer_bpe.py --data ${DATA_PATH}data_demose/train.all.dual --size 52000 --output ${DATA_PATH}52k-vocab-models



## tokenize translation data
mkdir ${DATA_PATH}bibert_tok
mkdir ${DATA_PATH}52k_tok

for prefix in "dev" "test" "train" ;
do
    for lang in "en" "fa" ;
    do
        python transform_tokenize.py --input ${DATA_PATH}data_demose/${prefix}.${lang} --output ${DATA_PATH}bibert_tok/${prefix}.${lang} --pretrained_model ${LM}
    done
done


for prefix in "dev" "test" "train" ;
do
    for lang in "en" "fa";
    do
    python transform_tokenize.py --input ${DATA_PATH}data_demose/${prefix}.${lang} --output ${DATA_PATH}52k_tok/${prefix}.${lang} --pretrained_model ${DATA_PATH}52k-vocab-models
    done
done


for one-way translation data
cp ${DATA_PATH}bibert_tok/*.en ${DATA_PATH}
cp ${DATA_PATH}52k_tok/*.fa ${DATA_PATH}



## get src and tgt vocabulary
python get_vocab.py --tokenizer ${LM} --output ${DATA_PATH}/src_vocab.txt
python get_vocab.py --tokenizer ${DATA_PATH}52k-vocab-models --output ${DATA_PATH}/tgt_vocab.txt


## fairseq preprocess
TEXT=${DATA_PATH}
fairseq-preprocess --source-lang en --target-lang fa  --trainpref $TEXT/train --validpref $TEXT/dev \
--testpref $TEXT/test --destdir ${TEXT}/en-fa-databin --srcdict $TEXT/src_vocab.txt \
--tgtdict $TEXT/tgt_vocab.txt --vocab_file $TEXT/src_vocab.txt --workers 25

## remove useless files
rm -rf data_demose
rm -rf bibert_tok
rm -rf 52k_tok









