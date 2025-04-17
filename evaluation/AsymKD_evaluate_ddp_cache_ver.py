# Last modified: 2024-03-11
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import argparse
import logging
import os
import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_SPACE)
print(CODE_SPACE)

import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
)
from dataset.util import metric
from dataset.util.alignment_gpu import (
    align_depth_least_square,
    depth2disparity,
    disparity2depth,
)
from dataset.util.metric import MetricTracker

import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor


from Efficient_BriGeS.Efficient_BriGeS_residual_refine import Efficient_BriGeS_residual_refine
from torch.multiprocessing import Manager
import torch.distributed as dist
import torch.multiprocessing as mp

@torch.no_grad()
def infer(model, image, seg_image=None, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    if seg_image is None:
        pred1 = model(image, **kwargs)
        pred2 = model(torch.flip(image, [3]), **kwargs)
    else:
        pred1 = model(image, seg_image, **kwargs)
        pred2 = model(torch.flip(image, [3]), torch.flip(seg_image, [3]), **kwargs)

    pred1 = get_depth_from_prediction(pred1)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred

eval_metrics = [
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
    "abs_relative_difference",
    "squared_relative_difference",
    "rmse_linear",
    "rmse_log",
    "log10",
    "i_rmse",
    "silog_rmse",
]




def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12364'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

import queue


def eval(rank, world_size, queue, args):
    try:
        setup(rank, world_size)
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()

        output_dir = args.output_dir

        model_type = args.model

        dataset_config = args.dataset_config
        base_data_dir = args.base_data_dir

        alignment = args.alignment
        alignment_max_res = args.alignment_max_res

        no_cuda = args.no_cuda
        pred_suffix = ".npy"

        os.makedirs(output_dir, exist_ok=True)

        cfg_data = OmegaConf.load(dataset_config)

        depth_any_a1 = None

        if(cfg_data.name == 'kitti'):
            depth_any_a1 = 0.9472893651094905
            depth_any_abs = 0.07969481239175322
        elif(cfg_data.name == 'nyu_v2'):
            depth_any_a1 = 0.9810679190384868
            depth_any_abs = 0.04232252059099962
        elif(cfg_data.name == 'eth3d'):
            depth_any_a1 = 0.9819326794620128
            depth_any_abs = 0.05689661835611589
        elif(cfg_data.name == 'scannet'):
            depth_any_a1 = 0.9816831358522177
            depth_any_abs = 0.04227282495703548
        elif(cfg_data.name == 'diode'):
            depth_any_a1 = 0.7583718170374438
            depth_any_abs = 0.25980516974892753

        cfg_data.filenames =  CODE_SPACE + '/evaluation/' + cfg_data.filenames
        dataset: BaseDepthDataset = get_dataset(
            cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.EVAL
        )

        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
        cache_dataloader = []
        cache_dataloader_flag = False
        # -------------------- Model --------------------
        while not queue.empty():
            restore_ckpt = str(queue.get())
            print(restore_ckpt)
            torch.cuda.empty_cache()
            if "depth_anything" in model_type:
                if "small" in model_type:
                    encoder = "vits"
                elif "base" in model_type:
                    encoder = "vitb"
                elif "large" in model_type:
                    encoder = "vitl"
                model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(rank)
            elif model_type == "kd_naive_depth_latent4_split_adapter":
                model = kd_naive_depth_latent4_split_adapter().to(rank)
                if restore_ckpt is not None:
                    logging.info("Loading checkpoint...")
                    checkpoint = torch.load(restore_ckpt, map_location=torch.device('cuda', rank))
                    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            elif model_type == "Efficient_BriGeS_residual_refine":
                segment_anything = sam_model_registry["vit_b"](checkpoint="Efficient_BriGeS_checkpoints/sam_vit_b_01ec64.pth").to(rank)
                segment_anything_predictor = SamPredictor(segment_anything)

                for child in segment_anything.children():
                    ImageEncoderViT = child
                    break
                model = Efficient_BriGeS_residual_refine(ImageEncoderViT=ImageEncoderViT).to(rank)
                if restore_ckpt is not None:
                    logging.info("Loading checkpoint...")
                    checkpoint = torch.load(restore_ckpt, map_location=torch.device('cuda', rank))
                    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
                

            # -------------------- Eval metrics --------------------
            metric_funcs = [getattr(metric, _met) for _met in eval_metrics]

            metric_tracker = MetricTracker(*[m.__name__ for m in metric_funcs])
            metric_tracker.reset()

            # -------------------- Per-sample metric file head --------------------
            # per_sample_filename = os.path.join(output_dir, "per_sample_metrics.csv")
            # # write title
            # with open(per_sample_filename, "w+") as f:
            #     f.write("filename,")
            #     f.write(",".join([m.__name__ for m in metric_funcs]))
            #     f.write("\n")

            # -------------------- Evaluate --------------------
            model.eval()
            if cache_dataloader_flag == True:
                dataloader = cache_dataloader
                print(f'cache_dataloader_flag {cache_dataloader_flag}')
            for data in dataloader:
                if cache_dataloader_flag == False:
                    # GT data
                    depth_raw_ts = data["depth_raw_linear"].squeeze().to(rank)
                    valid_mask_ts = data["valid_mask_raw"].squeeze().to(rank)
                    rgb = data["rgb_norm"].to(rank)
                    rgb_float = data["rgb_float"].to(rank)

                    depth_raw = depth_raw_ts
                    valid_mask = valid_mask_ts

                    depth_raw_ts = depth_raw_ts.to(rank)
                    valid_mask_ts = valid_mask_ts.to(rank)

                    # Get prediction
                    # (352, 1216)
                    if "kitti" in dataset_config:
                        pred_size = (518, 1792)
                    # (480, 640), (768, 1024), (480, 640)
                    elif "nyu" in dataset_config or "diode" in dataset_config or "scannet" in dataset_config:
                        pred_size = (518, 686)
                    # (4032, 6048)
                    elif "eth3d" in dataset_config:
                        pred_size = (518, 770)

                    rgb_resized = F.interpolate(rgb, size=pred_size, mode='bilinear', align_corners=False)
                    rgb_float_resized = F.interpolate(rgb_float, size=pred_size, mode='bilinear', align_corners=False)
                    if('BriGeS' in model_type):
                        rgb_resized_seg = segment_anything_predictor.set_image(rgb_float_resized.squeeze())
                        
                        append_data = {
                            'rgb_resized' : rgb_resized.detach().cpu(),
                            'rgb_resized_seg' : rgb_resized_seg.detach().cpu(),
                            'depth_raw_ts' : depth_raw_ts.detach().cpu(),
                            'valid_mask_ts' : valid_mask_ts.detach().cpu()
                        }
                    else:
                        append_data = {
                            'rgb_resized' : rgb_resized.detach().cpu(),
                            'depth_raw_ts' : depth_raw_ts.detach().cpu(),
                            'valid_mask_ts' : valid_mask_ts.detach().cpu()
                        }
                    cache_dataloader.append(append_data)
                else:
                    # GT data
                    depth_raw_ts = data["depth_raw_ts"].to(rank)
                    valid_mask_ts = data["valid_mask_ts"].to(rank)
                    rgb_resized = data["rgb_resized"].to(rank)
                    if('BriGeS' in model_type):
                        rgb_resized_seg = data["rgb_resized_seg"].to(rank)
                    
                    depth_raw = depth_raw_ts
                    valid_mask = valid_mask_ts

                if 'BriGeS' in model_type:
                    pred = infer(model, rgb_resized, rgb_resized_seg)
                else:
                    pred = infer(model, rgb_resized)


                depth_pred_ts = F.interpolate(pred, size=depth_raw_ts.shape, mode='bilinear', align_corners=False)
                depth_pred = depth_pred_ts.squeeze()
                # Align with GT using least square
                if "least_square" == alignment:
                    depth_pred, scale, shift = align_depth_least_square(
                        gt_arr=depth_raw,
                        pred_arr=depth_pred,
                        valid_mask_arr=valid_mask,
                        return_scale_shift=True,
                        max_resolution=alignment_max_res,
                    )
                elif "least_square_disparity" == alignment:
                    # convert GT depth -> GT disparity
                    gt_disparity, gt_non_neg_mask = depth2disparity(
                        depth=depth_raw, return_mask=True
                    )
                    # LS alignment in disparity space
                    pred_non_neg_mask = depth_pred > 0
                    valid_nonnegative_mask = valid_mask & gt_non_neg_mask & pred_non_neg_mask

                    disparity_pred, scale, shift = align_depth_least_square(
                        gt_arr=gt_disparity,
                        pred_arr=depth_pred,
                        valid_mask_arr=valid_nonnegative_mask,
                        return_scale_shift=True,
                        max_resolution=alignment_max_res,
                    )
                    # convert to depth
                    disparity_pred = torch.clamp(
                        disparity_pred, min=1e-3
                    )  # avoid 0 disparity
                    depth_pred = disparity2depth(disparity_pred)  # 이 함수는 이미 GPU에서 동작하도록 수정됨

                # Clip to dataset min max
                depth_pred = torch.clamp(
                    depth_pred, min=dataset.min_depth, max=dataset.max_depth
                )

                # clip to d > 0 for evaluation
                depth_pred = torch.clamp(depth_pred, min=1e-6)

                # Evaluate (using CUDA if available)
                sample_metric = []
                depth_pred_ts = depth_pred

                for met_func in metric_funcs:
                    _metric_name = met_func.__name__
                    _metric = met_func(depth_pred_ts, depth_raw_ts, valid_mask_ts).item()
                    sample_metric.append(_metric.__str__())
                    metric_tracker.update(_metric_name, _metric)

            # -------------------- Save metrics to file --------------------
            

            print_str = f'#######AsymKD {restore_ckpt} evaluate result#############\n'

            for key in metric_tracker.result().keys():
                if(depth_any_a1-metric_tracker.result()['delta1_acc']<0 and depth_any_abs - metric_tracker.result()['abs_relative_difference']>0):
                    print_str += f'@@@@{key} : {round(metric_tracker.result()[key], 3)}\n'
                else:
                    print_str += f'{key} : {round(metric_tracker.result()[key], 3)}\n'
            
            print(print_str)


            metrics_filename = f"eval_metrics-{model_type}-ddp.txt"

            _save_to = os.path.join(output_dir, metrics_filename)
            with open(_save_to, "a") as f:
                f.write(f'{print_str}\n')

            
            cache_dataloader_flag = True

    finally:
        cleanup()


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # model
    parser.add_argument(
        "--model", type=str, required=True, help="Model to evaluate."
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    # LS depth alignment
    parser.add_argument(
        "--alignment",
        choices=[None, "least_square", "least_square_disparity"],
        default=None,
        help="Method to estimate scale and shift between predictions and ground truth.",
    )
    parser.add_argument(
        "--alignment_max_res",
        type=int,
        default=None,
        help="Max operating resolution used for LS alignment",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to base checkpoint directory.",
    )
    parser.add_argument(
        "--start_step",
        type=int,
    )
    parser.add_argument(
        "--end_step",
        type=int,
    )
    parser.add_argument(
        "--save_step",
        type=int,
    )

    parser.add_argument("--no_cuda", action="store_true", help="Run without cuda")

    args = parser.parse_args()




    # -------------------- Data --------------------


    world_size = torch.cuda.device_count()
    manager = Manager()
    queue = manager.Queue()    
    # start_num = args.start_step
    # end_num = args.end_step
        
    # for i in range(end_num,start_num-1,-args.save_step):
    #     queue.put(f'{args.checkpoint_dir}/{i}_AsymKD_new_loss.pth')

    arr = ['22000', '21750', '15250', '14250', '12000', '8750', '8250', '7750', '7500', '6750', '6500', '5500', '5250', '5000', '6000', '4250', '4000', '3750', '5750', '3250', '3000', '2250', '1500', '1000']
    for i in arr:
        queue.put(f'{args.checkpoint_dir}/{i}_AsymKD_new_loss.pth')

    os.chdir(CODE_SPACE)

    
    mp.spawn(eval, args=(world_size,queue, args,), nprocs=world_size, join=True)
