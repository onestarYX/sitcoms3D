import os
from sitcoms3D.nerf.src.opt import get_opts
import torch
from collections import defaultdict

from tqdm import tqdm

# models
from models.nerf import (
    PosEmbedding
)
# from models.lerf_ngpcore import Lerfw
from sitcoms3D.nerf.models.rendering import (
    render_rays
)

# from datasets.sitcom_lerf import SitcomLerfDataset
# from datasets.blender import BlenderDataset
# from datasets.replica import ReplicaDataset
# from datasets.front import ThreeDFrontDataset
# from datasets.kitti360 import Kitti360Dataset
import numpy as np

from sitcoms3D.nerf.src.metrics import psnr

from utils.visualization import get_image_summary_from_vis_data, np_visualize_depth

import cv2
from utils import load_ckpt
import imageio
import argparse
from pathlib import Path
import json
# from omegaconf import OmegaConf
import pickle
from sklearn.decomposition import PCA


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, ts, N_samples, N_importance, use_disp,
                      chunk, predict_label, num_classes, white_back, **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for chunk_idx in range(0, B, chunk):
        inputs = {'rays': rays[chunk_idx:chunk_idx+chunk],
                  'ts': ts[chunk_idx:chunk_idx+chunk]}
        rendered_ray_chunks = \
            render_rays(models=models,
                        embeddings=embeddings,
                        rays=rays[chunk_idx:chunk_idx+chunk],
                        ts=ts[chunk_idx:chunk_idx + chunk],
                        predict_label=predict_label,
                        num_classes=num_classes,
                        N_samples=N_samples,
                        use_disp=use_disp,
                        perturb=0,
                        N_importance=N_importance,
                        chunk=chunk,
                        white_back=white_back,
                        test_time=True,
                        **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k].append(v)

    for k, v in results.items():
        results[k] = torch.cat(v, 0).cpu()

    return results


def compute_iou(pred, gt, num_cls):
    iou = []
    for cls_idx in range(num_cls):
        denom = np.logical_or((pred == cls_idx), (gt == cls_idx)).astype(int).sum()
        if denom == 0:
            iou.append(1)
        else:
            numer = np.logical_and((pred == cls_idx), (gt == cls_idx)).astype(int).sum()
            iou.append(numer / denom)
    return np.mean(iou)


def render_to_path(path, dataset, idx, models, embeddings, all_training_img_ids, config, save_img=False):
    sample = dataset[idx]
    rays = sample['rays']
    ts = sample['ts'].squeeze()
    results = batched_inference(models=models, embeddings=embeddings, rays=rays.cuda(), ts=ts.cuda(),
                                N_samples=config.N_samples, N_importance=config.N_importance,
                                use_disp=config.use_disp, chunk=config.chunk,
                                predict_label=config.predict_label, num_classes=config.num_classes,
                                white_back=dataset.white_back, all_img_ids=all_training_img_ids)

    rows = []
    metrics = {}

    # GT image and predicted image
    if config.dataset_name == 'sitcom3D':
        w, h = sample['img_wh']

    # GT image and predicted combined image
    img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
    img_pred_ = (img_pred * 255).astype(np.uint8)
    rgbs = sample['rgbs']
    img_gt = rgbs.view(h, w, 3)
    psnr_ = psnr(img_gt, img_pred).item()
    print(f"PSNR: {psnr_}")
    metrics['psnr'] = psnr_
    img_gt_ = np.clip(img_gt.cpu().numpy(), 0, 1)
    img_gt_ = (img_gt_ * 255).astype(np.uint8)
    rows.append(np.concatenate([img_gt_, img_pred_], axis=1))

    # Predicted static image and predicted static depth
    if config.encode_t:
        img_static = np.clip(results['rgb_fine_static'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_static_ = (img_static * 255).astype(np.uint8)
        static_depth = results['depth_fine_static_exp'].cpu().numpy()
    else:
        img_static_ = np.zeros((h, w, 3), dtype=np.ubyte)
        static_depth = results['depth_fine'].cpu().numpy()
    depth_static = np.array(np_visualize_depth(static_depth, cmap=cv2.COLORMAP_BONE))
    depth_static = depth_static.reshape(h, w, 1)
    depth_static_ = np.repeat(depth_static, 3, axis=2)
    rows.append(np.concatenate([img_static_, depth_static_], axis=1))

    res_img = np.concatenate(rows, axis=0)
    if save_img:
        imageio.imwrite(path, res_img)

    return metrics, res_img