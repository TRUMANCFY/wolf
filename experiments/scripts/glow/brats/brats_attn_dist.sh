cd ../../../
python -u distributed.py \
    --nproc_per_node 2 \
    --config  configs/brats/glow/glow-cat-uni-attn.json \
    --epochs 10 --valid_epochs 10 \
    --batch_size 64 --batch_steps 2 --eval_batch_size 1000 --init_batch_size 2048 \
    --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --warmup_steps 50 --weight_decay 1e-6 --grad_clip 0 \
    --image_size 64 --n_bits 8 \
    --data_path data/brats --model_path models/brats_attn --dataset brats

# time per iteration: 
# linear: 250s
# attn: 300s