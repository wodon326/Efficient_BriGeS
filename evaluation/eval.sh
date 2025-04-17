set -e
set -x

# python AsymKD_evaluate_ddp_cache_ver.py \
#     --model Efficient_BriGeS_residual_refine \
#     --base_data_dir ~/data/AsymKD \
#     --dataset_config evaluation/config/data_kitti_eigen_test.yaml \
#     --alignment least_square_disparity \
#     --output_dir evaluation/output/kitti_eigen_test \
#     --checkpoint_dir /home/wodon326/project/Efficient_BriGeS/checkpoint_Efficient_BriGeS_residual_refine

# CUDA_VISIBLE_DEVICES=5 python AsymKD_evaluate_ddp_cache_ver.py \
#     --model Efficient_BriGeS_residual_refine \
#     --base_data_dir ~/data/AsymKD \
#     --dataset_config evaluation/config/data_nyu_test.yaml \
#     --alignment least_square_disparity \
#     --output_dir evaluation/output/nyu_test \
#     --checkpoint_dir /home/wodon326/project/Efficient_BriGeS/checkpoint_Efficient_BriGeS_residual_refine

# CUDA_VISIBLE_DEVICES=0,1,2 python AsymKD_evaluate_ddp_cache_ver.py \
#     --model Efficient_BriGeS_residual_refine \
#     --base_data_dir ~/data/AsymKD \
#     --dataset_config evaluation/config/data_eth3d.yaml \
#     --alignment least_square_disparity \
#     --output_dir evaluation/output/eth3d \
#     --alignment_max_res 1024 \
#     --checkpoint_dir /home/wodon326/project/Efficient_BriGeS/checkpoint_Efficient_BriGeS_residual_refine

# CUDA_VISIBLE_DEVICES=6 python AsymKD_evaluate_ddp_cache_ver.py \
#     --model Efficient_BriGeS_residual_refine \
#     --base_data_dir ~/data/AsymKD \
#     --dataset_config evaluation/config/data_diode_all.yaml \
#     --alignment least_square_disparity \
#     --output_dir evaluation/output/diode \
#     --checkpoint_dir /home/wodon326/project/Efficient_BriGeS/checkpoint_Efficient_BriGeS_residual_refine

CUDA_VISIBLE_DEVICES=7 python AsymKD_evaluate_ddp_cache_ver.py \
    --model Efficient_BriGeS_residual_refine \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_scannet_val.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/scannet \
    --checkpoint_dir /home/wodon326/project/Efficient_BriGeS/checkpoint_Efficient_BriGeS_residual_refine