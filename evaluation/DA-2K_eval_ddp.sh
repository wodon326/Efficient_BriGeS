CUDA_VISIBLE_DEVICES=7 python DA-2K_eval_ddp.py \
    --model Efficient_BriGeS_residual \
    --root_dir /home/wodon326/data/DA-2K \
    --output_dir evaluation/output/da_2k \
    --checkpoint_dir /home/wodon326/project/Efficient_BriGeS/checkpoint_Efficient_BriGeS_naive