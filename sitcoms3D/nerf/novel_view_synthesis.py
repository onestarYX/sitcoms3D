import os
from sitcoms3D.nerf.src.opt import get_opts
import torch
from collections import defaultdict

from tqdm import tqdm

# models
from sitcoms3D.nerf.models.nerf import (
    PosEmbedding,
    NeRF
)
from sitcoms3D.nerf.models.rendering import (
    render_rays
)

from sitcoms3D.nerf.datasets.sitcom3D import Sitcom3DDataset
from datasets.ray_utils import get_ray_directions, get_rays
import numpy as np
from sitcoms3D.nerf.src.metrics import psnr

from utils.visualization import get_image_summary_from_vis_data, np_visualize_depth

import cv2
from utils import load_ckpt
import imageio
import argparse
from pathlib import Path
import json
from omegaconf import OmegaConf
import pickle

@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, ts, N_samples, N_importance, use_disp,
                      chunk, predict_label, num_classes, white_back, **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for chunk_idx in range(0, B, chunk):
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


def infer_results(rays, ts, models, embeddings, config):
    results = batched_inference(models=models, embeddings=embeddings, rays=rays.cuda(), ts=ts.cuda(),
                                N_samples=config.N_samples, N_importance=config.N_importance,
                                use_disp=config.use_disp, chunk=config.chunk,
                                predict_label=config.predict_label, num_classes=config.num_classes,
                                white_back=dataset.white_back, all_img_ids=dataset.img_ids)
    return results

def render_to_path(output_dir, all_results, img_wh, save_img=False):
    out_img_list = []
    for i, results in enumerate(all_results):
        rows = []
        w, h = img_wh

        img_static = np.clip(results['rgb_fine_static'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_static_ = (img_static * 255).astype(np.uint8)

        static_depth = results['depth_fine_static_exp'].cpu().numpy()
        depth_static = np.array(np_visualize_depth(static_depth, cmap=cv2.COLORMAP_BONE))
        depth_static = depth_static.reshape(h, w, 1)
        depth_static_ = np.repeat(depth_static, 3, axis=2)
        rows.append(np.concatenate([img_static_, depth_static_], axis=1))

        res_img = np.concatenate(rows, axis=0)
        out_img_list.append(res_img)
        if save_img:
            path = output_dir / f'{i}.jpg'
            imageio.imwrite(path, res_img)

    return out_img_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='results/novel_view')
    parser.add_argument('--use_ckpt', type=str)
    parser.add_argument('--select_part_idx', type=int)
    parser.add_argument('--sweep_parts', action='store_true', default=False)
    parser.add_argument('--num_parts', type=int, default=-1)
    parser.add_argument('--num_images', type=int, default=-1)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--remap_dir', type=str, default=None)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = exp_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.remap_dir is not None:
        input_dir = output_dir
        img_path_list = []
        for file in input_dir.iterdir():
            if file.suffix == '.gif':
                continue
            img_path_list.append(file)
        img_path_list.sort(key=lambda x: x.name)
        with imageio.get_writer(output_dir / 'seq.gif', mode='I', fps=4) as writer:
            for img_path in img_path_list:
                image = imageio.imread(img_path)
                writer.append_data(image)
        exit()

    config_path = exp_dir / 'config.json'
    with open(config_path, 'r') as f:
        file_config = json.load(f)
    file_config = OmegaConf.create(file_config)
    cli_config = OmegaConf.from_dotlist(args.opts)
    config = OmegaConf.merge(file_config, cli_config)

    kwargs = {}
    if config.dataset_name == 'sitcom3D':
        kwargs.update({'environment_dir': config.environment_dir,
                  'near_far_version': config.near_far_version,
                  'num_limit': config.num_limit})
        kwargs['use_cache'] = config.use_cache
        dataset = Sitcom3DDataset(split=args.split, img_downscale=config.img_downscale_val, **kwargs)
    w, h = dataset[0]['img_wh']


    embeddings = {}
    if config.encode_a:
        embedding_a = torch.nn.Embedding(config.N_vocab, config.N_a).cuda()
        load_ckpt(embedding_a, args.use_ckpt, model_name='embedding_a')
        embeddings['a'] = embedding_a
    if config.encode_t:
        embedding_t = torch.nn.Embedding(config.N_vocab, config.N_tau).cuda()
        load_ckpt(embedding_t, args.use_ckpt, model_name='embedding_t')
        embeddings['t'] = embedding_t
    embedding_xyz = PosEmbedding(config.N_emb_xyz - 1, config.N_emb_xyz)
    embedding_dir = PosEmbedding(config.N_emb_dir - 1, config.N_emb_dir)
    embeddings['xyz'] = embedding_xyz
    embeddings['dir'] = embedding_dir

    nerf_coarse = NeRF('coarse',
                            in_channels_xyz=6 * config.N_emb_xyz + 3,
                            in_channels_dir=6 * config.N_emb_dir + 3,
                            encode_appearance=False,
                            predict_label=config.predict_label,
                            num_classes=config.num_classes,
                            use_view_dirs=config.use_view_dirs).cuda()
    nerf_fine = NeRF('fine',
                          in_channels_xyz=6 * config.N_emb_xyz + 3,
                          in_channels_dir=6 * config.N_emb_dir + 3,
                          encode_appearance=config.encode_a,
                          in_channels_a=config.N_a,
                          encode_transient=config.encode_t,
                          in_channels_t=config.N_tau,
                          predict_label=config.predict_label,
                          num_classes=config.num_classes,
                          beta_min=config.beta_min,
                          use_view_dirs=config.use_view_dirs).cuda()

    load_ckpt(nerf_coarse, args.use_ckpt, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.use_ckpt, model_name='nerf_fine')
    models = {'coarse': nerf_coarse, 'fine': nerf_fine}

    cam_seq = []

    sample_img_idx = 0
    sample_img_id = dataset.img_ids[sample_img_idx]
    K = dataset.Ks[sample_img_id]
    all_poses = dataset.poses
    begin_pose = all_poses[sample_img_idx]
    # begin_pose[:, 3] += [0.1, 0, 0.5]
    begin_pose[:, 3] += [0, 0, 0]
    cam_seq.append(begin_pose)

    num_steps = 30
    step_delta = [0, 0, -0.06]
    for step in range(1, num_steps):
        new_pose = np.copy(cam_seq[-1])
        new_pose[:, 3] += step_delta
        cam_seq.append(new_pose)

    all_results = []
    with torch.no_grad():
        for pose in tqdm(cam_seq):
            pose = torch.tensor(pose, dtype=torch.float32)
            directions = get_ray_directions(h, w, K)
            rays_o, rays_d = get_rays(directions, pose)
            rays_t = torch.ones(len(rays_o), dtype=torch.long) * sample_img_idx
            rays_o = rays_o.cuda(); rays_d = rays_d.cuda(); rays_t = rays_t.cuda()
            nears, fars, inbbox_ray_mask = dataset.get_nears_fars_from_rays_or_cam(rays_o, rays_d, c2w=pose)
            rays = torch.cat([rays_o, rays_d, nears, fars], 1)


            results = infer_results(rays, rays_t, models, embeddings, config)
            all_results.append(results)
            torch.cuda.empty_cache()

    out_img_list = render_to_path(output_dir, all_results, (w, h), save_img=True)
    imageio.mimsave(output_dir / 'seq.gif', out_img_list, fps=4)

