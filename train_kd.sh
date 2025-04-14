python train_kd.py \
    --batch_size 2 \
    --num_steps 1000000 \
    --lr 0.00005 \
    --train_datasets VKITTI Hypersim tartan_air \
    --ckpt InSDA_checkpoints/BriGeS_DAv2_large_best.pth \
    --student_ckpt InSDA_checkpoints/depth_anything_v2_vits.pth \
    --save_dir kd_fitnet_style \
    --train_style trans

