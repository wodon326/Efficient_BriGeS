import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib
import logging
from torchvision.transforms import Compose
from tqdm import tqdm

from BriGeS.util.transform import Resize, NormalizeImage, PrepareForNet
from segment_anything import SamPredictor, sam_model_registry
from BriGeS.dpt import BriGeS_DepthAnythingV2, BriGeS_DepthAnythingV2_Infer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root_path', type=str)
    parser.add_argument('--input_filename_path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, choices=['vitb', 'vitl'])
    parser.add_argument('--bfm_checkpoint', type=str)
    parser.add_argument('--infer_width', type=int)
    parser.add_argument('--infer_height', type=int, default=518)

    args = parser.parse_args()

    margin_width = 50
    caption_height = 60

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    segment_anything = sam_model_registry['vit_l'](checkpoint="/home/wodon326/datasets/AsymKD_checkpoints/sam_vit_l_0b3195.pth").to(DEVICE)
    segment_anything_predictor = SamPredictor(segment_anything)
    for child in segment_anything.children():
        ImageEncoderViT = child
        break
    model = BriGeS_DepthAnythingV2_Infer(ImageEncoderViT=ImageEncoderViT).to(DEVICE)
    restore_ckpt = args.bfm_checkpoint

    if restore_ckpt is not None:
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(restore_ckpt, map_location=DEVICE)
        model_state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in checkpoint.items():
            new_key = k.replace('module.', '')
            if new_key in model_state_dict:
                new_state_dict[new_key] = v

        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
    else:
        raise ValueError("To evaluate bfm, a checkpoint is needed.")

    total_params = sum(param.numel() for param in model.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    transform = Compose([
        Resize(
            width=args.infer_width,
            height=args.infer_height,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    filenames = []
    save_paths = []
    with open(args.input_filename_path, 'r') as file:
        for line in file:
            img_path = line.split()[0]
            filenames.append(os.path.join(args.input_root_path, img_path))
            save_paths.append(img_path)

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('magma')

    for idx, filename in enumerate(tqdm(filenames)):
        raw_image = cv2.imread(filename)
        image_seg = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image_depth = image_seg / 255.0

        h, w = image_depth.shape[:2]

        image_depth = transform({'image': image_depth})['image']
        image_depth = torch.from_numpy(image_depth).unsqueeze(0).to(DEVICE)

        image_seg = cv2.resize(image_seg, (args.infer_width, args.infer_height))
        image_seg = segment_anything_predictor.set_image(image_seg).to(DEVICE)

        with torch.no_grad():
            depth = model(image_depth, image_seg)

        depth = F.interpolate(depth, (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)

        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        os.makedirs(
            os.path.join(args.outdir, os.path.dirname(save_paths[idx])),
            exist_ok=True
        )
        filename = os.path.join(args.outdir, save_paths[idx])
        cv2.imwrite(filename, depth)
 