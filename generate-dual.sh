DATAPATH=./download_prepare/data_mixed_ft/
STPATH=${DATAPATH}de-en-databin/
MODELPATH=./models/dual-ft/ 
PRE_SRC=haoranxu/bibert-ende
PRE=./download_prepare/12k-vocab-models/
CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
--beam 4 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.en.txt \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out
