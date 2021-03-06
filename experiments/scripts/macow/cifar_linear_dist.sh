cd ../..
python -u distributed.py \
    --nproc_per_node 3 \
    --config  configs/cifar10/macow/macow-cat-uni.json \
    --epochs 100 --valid_epochs 10 \
    --batch_size 512 --batch_steps 2 --eval_batch_size 512 --init_batch_size 1800 \
    --lr 0.001 --beta1 0.9 --beta2 0.999 --eps 1e-8 --warmup_steps 50 --weight_decay 1e-6 --grad_clip 0 \
    --image_size 32 --n_bits 8 \
    --data_path data/cifar_data --model_path models/macow/cifar_linear_model --dataset cifar10 --recover 1