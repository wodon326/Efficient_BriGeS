set -e
set -x

CUDA_VISIBLE_DEVICES=3 python AsymKD_evaluate_affine_inv_gpu.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_diode_all.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/diode