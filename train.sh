TEXT=./download_prepare/data/
SAVE_DIR=./models/one-way/

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train ${TEXT}de-en-databin/ --arch transformer_iwslt_de_en --ddp-backend no_c10d --optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 1.0 --lr 0.0004 --lr-scheduler inverse_sqrt --warmup-updates 1000 --warmup-init-lr 1e-07 --dropout 0.3 \
--weight-decay 0.00002 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 2048 --update-freq 4 \
--attention-dropout 0.1 --activation-dropout 0.1 --max-epoch 75 --save-dir ${SAVE_DIR}  --encoder-embed-dim 768 --decoder-embed-dim 768 \
--no-epoch-checkpoints --save-interval 1 --pretrained_model haoranxu/bibert-ende --use_drop_embedding 8 
