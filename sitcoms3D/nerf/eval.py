import os

import os
from sitcoms3D.nerf.src.opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader

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

import numpy as np

from sitcoms3D.nerf.src.metrics import psnr

from sitcoms3D.nerf.utils.visualization import get_image_summary_from_vis_data, np_visualize_depth

import cv2
from sitcoms3D.nerf.utils import load_ckpt
import imageio


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, ts, predict_label, num_classes, N_samples, N_importance, use_disp,
                      chunk, white_back, **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        ts[i:i+chunk] if ts is not None else None,
                        predict_label,
                        num_classes,
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        validation_version=True,
                        **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
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


if __name__ == '__main__':
    args = get_opts()
    dataset_name = Sitcom3DDataset

    kwargs = {'environment_dir': args.environment_dir,
              'near_far_version': args.near_far_version}
    # kwargs['img_downscale'] = args.img_downscale
    kwargs['val_num'] = 5
    kwargs['use_cache'] = args.use_cache
    dataset = dataset_name(split='val', img_downscale=args.img_downscale_val, **kwargs)

    embedding_xyz = PosEmbedding(args.N_emb_xyz - 1, args.N_emb_xyz)
    embedding_dir = PosEmbedding(args.N_emb_dir - 1, args.N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    if args.encode_a:
        embedding_a = torch.nn.Embedding(args.N_vocab, args.N_a).cuda()
        load_ckpt(embedding_a, args.ckpt_path, model_name='embedding_a')
        embeddings['a'] = embedding_a
    if args.encode_t:
        embedding_t = torch.nn.Embedding(args.N_vocab, args.N_tau).cuda()
        load_ckpt(embedding_t, args.ckpt_path, model_name='embedding_t')
        embeddings['t'] = embedding_t

    nerf_coarse = NeRF('coarse',
                       in_channels_xyz=6 * args.N_emb_xyz + 3,
                       in_channels_dir=6 * args.N_emb_dir + 3,
                       encode_appearance=False,
                       predict_label=args.predict_label,
                       num_classes=args.num_classes,
                       use_view_dirs=args.use_view_dirs).cuda()
    nerf_fine = NeRF('fine',
                     in_channels_xyz=6 * args.N_emb_xyz + 3,
                     in_channels_dir=6 * args.N_emb_dir + 3,
                     encode_appearance=args.encode_a,
                     in_channels_a=args.N_a,
                     encode_transient=args.encode_t,
                     in_channels_t=args.N_tau,
                     predict_label=args.predict_label,
                     num_classes=args.num_classes,
                     beta_min=args.beta_min,
                     use_view_dirs=args.use_view_dirs).cuda()

    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')

    models = {'coarse': nerf_coarse, 'fine': nerf_fine}

    imgs, psnrs = [], []
    run_name = args.ckpt_path.split("/")[-4]
    dir_name = f'{args.environment_dir}/rendering/{run_name}'
    os.makedirs(dir_name, exist_ok=True)

    label_colors = np.random.rand(args.num_classes, 3)

    iou_combined = []
    iou_static = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays']
        ts = sample['ts']
        results = batched_inference(models, embeddings, rays.cuda(), ts.cuda(),
                                    args.predict_label, args.num_classes,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back,
                                    **kwargs)

        w, h = sample['img_wh']

        img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_pred_ = (img_pred * 255).astype(np.uint8)

        rgbs = sample['rgbs']
        img_gt = rgbs.view(h, w, 3)
        psnrs += [psnr(img_gt, img_pred).item()]
        img_gt_ = np.clip(img_gt.cpu().numpy(), 0, 1)
        img_gt_ = (img_gt_ * 255).astype(np.uint8)

        img_static = np.clip(results['rgb_fine_static'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_static_ = (img_static * 255).astype(np.uint8)

        depth_static = np.array(np_visualize_depth(results['depth_fine_static_med'].cpu().numpy(), cmap=cv2.COLORMAP_BONE))
        depth_static = depth_static.reshape(h, w, 1)
        depth_static_ = np.repeat(depth_static, 3, axis=2)

        if args.predict_label:
            label_gt = sample['labels'].to(torch.long).cpu().numpy()
            label_map_gt = label_colors[label_gt].reshape((h, w, 3))
            label_map_gt = (label_map_gt * 255).astype(np.uint8)

            label_pred = torch.argmax(results['label_fine'], dim=1).to(torch.long).cpu().numpy()
            label_map_pred = label_colors[label_pred].reshape((h, w, 3))
            label_map_pred = (label_map_pred * 255).astype(np.uint8)
            iou_combined.append(compute_iou(label_pred, label_gt, args.num_classes))

            label_static_pred = torch.argmax(results['label_fine_static'], dim=1).to(torch.long).cpu().numpy()
            label_map_static_pred = label_colors[label_static_pred].reshape((h, w, 3))
            label_map_static_pred = (label_map_static_pred * 255).astype(np.uint8)
            iou_static.append(compute_iou(label_static_pred, label_gt, args.num_classes))

            label_transient_pred = torch.argmax(results['label_fine_transient'], dim=1).to(torch.long).cpu().numpy()
            label_map_transient_pred = label_colors[label_transient_pred].reshape((h, w, 3))
            label_map_transient_pred = (label_map_transient_pred * 255).astype(np.uint8)

        row1 = np.concatenate([img_gt_, img_pred_], axis=1)
        row2 = np.concatenate([img_static_, depth_static_], axis=1)
        row3 = np.concatenate([label_map_gt, label_map_pred], axis=1)
        row4 = np.concatenate([label_map_static_pred, label_map_transient_pred], axis=1)
        res_img = np.concatenate([row1, row2, row3, row4], axis=0)
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), res_img)

    if args.predict_label:
        print('Mean IoU combined', iou_combined)
        print('Mean IoU static', iou_static)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')
