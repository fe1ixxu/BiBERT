INPUT=$1
OUTPUT=$2
FAIRSEQ=$3
MT_MODEL=$4
PRETRAIN_MODEL=$5

TEMP_DIR=temp_dir_for_mt
mkdir $TEMP_DIR  # create temp MT workplace

python ${FAIRSEQ}/download_prepare/transform_tokenize.py --input $INPUT --output ${TEMP_DIR}/test.en --pretrained_model $PRETRAIN_MODEL

## create dummy files for preprocessing
cp ${TEMP_DIR}/test.en ${TEMP_DIR}/test.fa
cp ${TEMP_DIR}/test.en ${TEMP_DIR}/train.en
cp ${TEMP_DIR}/test.en ${TEMP_DIR}/train.fa
cp ${TEMP_DIR}/test.en ${TEMP_DIR}/dev.en
cp ${TEMP_DIR}/test.en ${TEMP_DIR}/dev.fa

## fairseq preprocess
TEXT=$TEMP_DIR
fairseq-preprocess --source-lang en --target-lang fa  --trainpref $TEXT/train --validpref $TEXT/dev \
--testpref $TEXT/test --destdir ${TEXT}/en-fa-databin --srcdict $FAIRSEQ/data/en-fa-remove-u200/src_vocab.txt \
--tgtdict $FAIRSEQ/data/en-fa-remove-u200/tgt_vocab.txt --vocab_file $FAIRSEQ/data/en-fa-remove-u200/src_vocab.txt --workers 25

## Translation
STPATH=${TEMP_DIR}/en-fa-databin/
MODELPATH=${MT_MODEL}
PRE_SRC=${PRETRAIN_MODEL}
PRE=${FAIRSEQ}/data/en-fa-remove-u200/52k-vocab-models
CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH} --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
--beam 4 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.fa.txt --pretrain ${PRETRAIN_MODEL} \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out 

cat ${STPATH}/generate.out | grep '^D-' | sed 's/^..//' | sort -n | awk 'BEGIN{FS="\t"}{print $3}' > $OUTPUT

rm -rf $TEMP_DIR