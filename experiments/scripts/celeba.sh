cd ..
python -u distributed.py \
    --nproc_per_node 3 \
    --config  configs/celebA-HQ/glow/glow-base-uni-simple.json \
    --epochs 50 --valid_epochs 10 \
    --batch_size 24 --batch_steps 2 --eval_batch_size 3 --init_batch_size 512 \
    --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --warmup_steps 10 --weight_decay 5e-4 --grad_clip 0 \
    --image_size 256 --n_bits 8 \
    --data_path data/celeba_data --model_path models/celeba_model --dataset celeba

# time per iteration: 
# linear: 250s
# attn: 300s
# 70714159