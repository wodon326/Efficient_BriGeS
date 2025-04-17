set -e
set -x

python AsymKD_evaluate_ddp_cache_ver.py \
    --model Efficient_BriGeS_residual_refine \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_kitti_eigen_test.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/kitti_eigen_test \
    --start_step 250 \
    --end_step 77750 \
    --save_step 250 \
    --checkpoint_dir /home/wodon326/project/Efficient_BriGeS/checkpoint_Efficient_BriGeS_residual_refine