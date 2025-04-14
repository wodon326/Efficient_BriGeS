python train_BriGeS.py \
    --batch_size 4 \
    --num_steps 1000000 \
    --lr 0.00005 \
    --train_datasets VKITTI Hypersim tartan_air BlendedMVS IRS \
    --save_dir Efficient_BriGeS_residual \
    --train_style trans \
    --restore_ckpt /home/wodon326/project/Efficient_BriGeS/checkpoint_Efficient_BriGeS_residual/44500_AsymKD_new_loss.pth

