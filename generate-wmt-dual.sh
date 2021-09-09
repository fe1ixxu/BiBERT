DATAPATH=./download_prepare/wmt-data/
STPATH=${DATAPATH}en-de-databin/
MODELPATH=./models/dual-wmt-ft/ 
PRE_SRC=jhu-clsp/bibert-ende
PRE=jhu-clsp/bibert-ende
CUDA_VISIBLE_DEVICES=0 fairseq-generate \
${STPATH} --path ${MODELPATH}checkpoint_best.pt --bpe bert --pretrained_bpe ${PRE} --pretrained_bpe_src ${PRE_SRC} \
--beam 4 --lenpen 0.6 --remove-bpe --vocab_file=${STPATH}/dict.de.txt \
--max-len-a 1 --max-len-b 50|tee ${STPATH}/generate.out
