DATAPATH=./data/en-fa/
STPATH=${DATAPATH}en-fa-databin/
MODELPATH=./models/en-fa-52k-v1/
PRE_SRC=/export/c01/haoranxu/LMs/EnFa-large-128K-v1.0/
PRE=/export/c01/haoranxu/BiBERT/data/en-fa/52k-vocab-models
CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
--beam 4 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.fa.txt \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out
