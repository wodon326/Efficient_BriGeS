python DA-2K_eval_tau_ddp.py \
    --model bfm \
    --root_dir /home/wodon326/data/DA-2K \
    --output_dir evaluation/output/da_2k \
    --checkpoint_dir /home/wodon326/project/BriGeS-Depth-Anything-V2/checkpoints_new_loss_001_smooth

    
python AsymKD_evaluate_affine_ddp_tau.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_kitti_eigen_test.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/kitti_eigen_test \
    --checkpoint_dir /home/wodon326/project/BriGeS-Depth-Anything-V2/checkpoints_new_loss_001_smooth

python AsymKD_evaluate_affine_ddp_tau.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_nyu_test.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/nyu_test \
    --checkpoint_dir /home/wodon326/project/BriGeS-Depth-Anything-V2/checkpoints_new_loss_001_smooth

python AsymKD_evaluate_affine_ddp_tau.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_eth3d.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/eth3d \
    --alignment_max_res 1024 \
    --checkpoint_dir /home/wodon326/project/BriGeS-Depth-Anything-V2/checkpoints_new_loss_001_smooth

python AsymKD_evaluate_affine_ddp_tau.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_diode_all.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/diode \
    --checkpoint_dir /home/wodon326/project/BriGeS-Depth-Anything-V2/checkpoints_new_loss_001_smooth

python AsymKD_evaluate_affine_ddp_tau.py \
    --model bfm \
    --base_data_dir ~/data/AsymKD \
    --dataset_config evaluation/config/data_scannet_val.yaml \
    --alignment least_square_disparity \
    --output_dir evaluation/output/scannet \
    --checkpoint_dir /home/wodon326/project/BriGeS-Depth-Anything-V2/checkpoints_new_loss_001_smooth