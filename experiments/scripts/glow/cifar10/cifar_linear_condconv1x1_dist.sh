cd ../../../
python -u distributed.py \
    --nproc_per_node 3 \
    --config  configs/cifar10/glow/glow-cat-uni-condconv1x1.json \
    --epochs 200 --valid_epochs 10 \
    --batch_size 120 --batch_steps 2 --eval_batch_size 300 --init_batch_size 1024 \
    --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --warmup_steps 50 --weight_decay 1e-6 --grad_clip 0 \
    --image_size 32 --n_bits 8 \
    --data_path data/cifar_data --model_path models/glow/cifar_linear_condconv1x1_model --dataset cifar10

# the previous model is cifar_linear_attn_model, which only add simple attn in the first coupling of first block
# cifar_linear_multi_attn_model: add multi-head attn in the frist coupling of first block