set -e
set -x

# CUDA_VISIBLE_DEVICES=6 python infer.py \
#     --input_root_path ~/data/AsymKD/kitti_eigen_split_test \
#     --input_filename_path evaluation/data_split/kitti/eigen_test_files_with_gt.txt \
#     --outdir evaluation/output/kitti_eigen_test/briges_depth_anythingv2 \
#     --bfm_checkpoint /home/wodon326/project/BriGeS-Depth-Anything-V2/best_checkpoint/1209_3000_AsymKD_new_loss.pth \
#     --encoder vitl \
#     --infer_width 1792 \
#     --infer_height 518

# CUDA_VISIBLE_DEVICES=5 python infer.py \
#     --input_root_path ~/data/AsymKD/nyu_labeled_extracted \
#     --input_filename_path evaluation/data_split/nyu/labeled/filename_list_test.txt \
#     --outdir evaluation/output/nyu_test/briges_depth_anythingv2 \
#     --bfm_checkpoint /home/wodon326/project/BriGeS-Depth-Anything-V2/best_checkpoint/1209_3000_AsymKD_new_loss.pth \
#     --encoder vitl \
#     --infer_width 686 \
#     --infer_height 518

# CUDA_VISIBLE_DEVICES=7 python infer.py \
#     --input_root_path ~/data/AsymKD/eth3d \
#     --input_filename_path evaluation/data_split/eth3d/eth3d_filename_list.txt \
#     --outdir evaluation/output/eth3d/briges_depth_anythingv2 \
#     --bfm_checkpoint /home/wodon326/project/BriGeS-Depth-Anything-V2/best_checkpoint/1209_3000_AsymKD_new_loss.pth \
#     --encoder vitl \
#     --infer_width 770 \
#     --infer_height 518

# CUDA_VISIBLE_DEVICES=7 python infer.py \
#     --input_root_path ~/data/AsymKD/diode_val \
#     --input_filename_path evaluation/data_split/eth3d/diode_val_all_filename_list.txt \
#     --outdir evaluation/output/diode/briges_depth_anythingv2 \
#     --bfm_checkpoint /home/wodon326/project/BriGeS-Depth-Anything-V2/best_checkpoint/1209_3000_AsymKD_new_loss.pth \
#     --encoder vitl \
#     --infer_width 686 \
#     --infer_height 518

CUDA_VISIBLE_DEVICES=1 python infer.py \
    --input_root_path ~/data/AsymKD/scannet_val_sampled_800_1 \
    --input_filename_path evaluation/data_split/scannet/scannet_val_sampled_list_800_1.txt \
    --outdir evaluation/output/scannet/briges_depth_anythingv2 \
    --bfm_checkpoint /home/wodon326/project/BriGeS-Depth-Anything-V2/best_checkpoint/1209_3000_AsymKD_new_loss.pth \
    --encoder vitl \
    --infer_width 686 \
    --infer_height 518
