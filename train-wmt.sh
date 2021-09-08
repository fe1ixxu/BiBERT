TEXT=./download_prepare/wmt-data/
SAVE_DIR=./models/one-way-wmt/

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train ${TEXT}en-de-databin/ --arch transformer_vaswani_wmt_en_de_big --ddp-backend no_c10d --optimizer adam \
--adam-betas '(0.9, 0.98)' --clip-norm 1.0 --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 --dropout 0.3 \
--weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --update-freq 32 --attention-dropout 0.1 \
--activation-dropout 0.1 --max-epoch 100 --save-dir ${SAVE_DIR}  --encoder-embed-dim 768 --decoder-embed-dim 768 --no-epoch-checkpoints \
--max-source-positions 512 --max-target-positions 512 --pretrained_model haoranxu/bibert-ende --use_drop_embedding 8 
