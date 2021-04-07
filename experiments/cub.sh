python -u train.py \
    --config  configs/cub/glow/glow-base-uni.json \
    --epochs 100 --valid_epochs 10 \
    --batch_size 6 --batch_steps 1 --eval_batch_size 100 --init_batch_size 100 \
    --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --warmup_steps 50 --weight_decay 1e-6 --grad_clip 0 \
    --image_size 128 --n_bits 8 \
    --data_path data/birds --model_path cub_model --dataset cub