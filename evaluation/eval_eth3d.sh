set -e
set -x

CUDA_VISIBLE_DEVICES=2 python AsymKD_evaluate_affine_inv_gpu.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_eth3d.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/eth3d \
    --alignment_max_res 1024