import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import wandb
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

# --- Additional metrics ---
import piq        # FSIM, GMSD, VIF
import cv2        # Canny, edge maps
from skimage import feature
from scipy.fft import fft2, fftshift

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

#New -> Compute fsim, gmsd, vif
def compute_additional_metrics(pred, target):
    """
    pred, target: tensors in [1, 3, H, W] (NCHW)
    Returns dict of FSIM, GMSD, VIF.
    """
    fsim = piq.fsim(pred, target).item()
    gmsd = piq.gmsd(pred, target).item()
    vif = piq.vif_p(pred, target).item()
    return {"fsim": fsim, "gmsd": gmsd, "vif": vif}
#New -> Compute EPI sharpness
def compute_epi_sharpness(img):
    """
    img: [H, W, 3]
    Returns mean horizontal EPI sharpness.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sharpness = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
    return sharpness
#New -> Compute FFT metrics -> HF ratio and PSD
def compute_fft_metrics(img):
    """
    img: [H, W, 3] float32
    Returns HF ratio and PSD.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    F = fftshift(fft2(gray))
    P = np.abs(F)

    # High frequency region = outer 30%
    h, w = P.shape
    cx, cy = w//2, h//2
    r = min(cx, cy)
    hf_mask = np.zeros_like(P)
    cv2.circle(hf_mask, (cx, cy), int(r*0.3), 1, thickness=-1)
    hf = P * (1 - hf_mask)

    hf_ratio = hf.sum() / P.sum()
    return hf_ratio, P

#New -> Compute Canny edge map similarity
def compute_canny_f1(pred_img, target_img):
    """
    pred_img, target_img: [H,W,3] in [0,1]
    """
    p = (pred_img * 255).astype(np.uint8)
    t = (target_img * 255).astype(np.uint8)
    p_edges = cv2.Canny(p, 100, 200)
    t_edges = cv2.Canny(t, 100, 200)

    tp = np.sum((p_edges > 0) & (t_edges > 0))
    fp = np.sum((p_edges > 0) & (t_edges == 0))
    fn = np.sum((p_edges == 0) & (t_edges > 0))

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    return f1

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    
    # Handle both module-based and function-based embedders
    if isinstance(embed_fn, nn.Module):
        embedded = embed_fn(inputs_flat)
    else:
        embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        
        # Handle both module-based and function-based embedders
        if isinstance(embeddirs_fn, nn.Module):
            embedded_dirs = embeddirs_fn(input_dirs_flat)
        else:
            embedded_dirs = embeddirs_fn(input_dirs_flat)
        
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # Create embedders (position encoding)
    embed_fn, input_ch = get_embedder(
        args.multires, args.i_embed,
        learnable=args.learnable_pe,
        learnable_phase=args.learnable_pe_phase,
        learnable_freqs=getattr(args, 'pe_learnable_freqs', True),
        init_scale=getattr(args, 'pe_init_scale', 1.0),
        pe_type=getattr(args, 'pe_type', 'baseline'),
        gaussian_num_feats=getattr(args, 'pe_num_feats', 30),
        gaussian_sigma=getattr(args, 'pe_sigma', 10.0),
    )

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed,
            learnable=args.learnable_pe,
            learnable_phase=args.learnable_pe_phase,
            learnable_freqs=getattr(args, 'pe_learnable_freqs', True),
            init_scale=getattr(args, 'pe_init_scale', 1.0),
            pe_type=getattr(args, 'pe_type', 'baseline'),
            gaussian_num_feats=getattr(args, 'pe_num_feats', 30),
            gaussian_sigma=getattr(args, 'pe_sigma', 10.0),
        )
    else:
        embeddirs_fn = None
        input_ch_views = 0
  
    # Move embedders to device if they are nn.Modules
    if isinstance(embed_fn, nn.Module):
        embed_fn = embed_fn.to(device)
    if embeddirs_fn is not None and isinstance(embeddirs_fn, nn.Module):
        embeddirs_fn = embeddirs_fn.to(device)
    
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    # Separate parameters for different learning rates
    network_params = list(model.parameters())
    if model_fine is not None:
        network_params += list(model_fine.parameters())
    
    pe_params = []
    if args.learnable_pe:
        if isinstance(embed_fn, nn.Module):
            pe_params += list(embed_fn.parameters())
        if embeddirs_fn is not None and isinstance(embeddirs_fn, nn.Module):
            pe_params += list(embeddirs_fn.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer with separate learning rates
    # PE parameters need moderate learning rate - not too low to allow adaptation
    pe_lr_scale = getattr(args, 'pe_lr_scale', 0.5)
    pe_lrate = args.lrate * pe_lr_scale if args.learnable_pe else args.lrate
    if args.learnable_pe and len(pe_params) > 0:
        optimizer = torch.optim.Adam([
            {'params': network_params, 'lr': args.lrate},
            {'params': pe_params, 'lr': pe_lrate}
        ], betas=(0.9, 0.999))
        print(f'Using separate learning rates: network={args.lrate}, PE={pe_lrate}')
    else:
        optimizer = torch.optim.Adam(params=network_params + pe_params, lr=args.lrate, betas=(0.9, 0.999))
    
    grad_vars = network_params + pe_params

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None and 'network_fine_state_dict' in ckpt and ckpt['network_fine_state_dict'] is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        
        # Load embedder parameters if they exist
        if args.learnable_pe:
            # Try to load embedder parameters from checkpoint
            if 'embed_fn_state_dict' in ckpt and isinstance(embed_fn, nn.Module):
                try:
                    embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])
                    print('Loaded learnable PE embedder parameters from checkpoint')
                except Exception as e:
                    print(f'Warning: Could not load embedder parameters: {e}')
                    print('Initializing learnable PE with default frequencies (from fixed PE checkpoint)')
            else:
                # Checkpoint was saved with fixed PE, initialize learnable PE with standard frequencies
                print('Checkpoint was saved with fixed PE. Initializing learnable PE with standard frequencies.')
                print('The learnable PE will start from the same frequencies as fixed PE and can adapt during training.')
            
            if 'embeddirs_fn_state_dict' in ckpt and isinstance(embeddirs_fn, nn.Module):
                try:
                    embeddirs_fn.load_state_dict(ckpt['embeddirs_fn_state_dict'])
                    print('Loaded learnable PE view embedder parameters from checkpoint')
                except Exception as e:
                    print(f'Warning: Could not load view embedder parameters: {e}')
                    print('Initializing learnable PE view embedder with default frequencies')

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    print("embed_fn type:", type(embed_fn))
    if hasattr(embed_fn, 'freq_bands'):
        print("freq_bands:", embed_fn.freq_bands[:5])

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, embed_fn, embeddirs_fn, pe_lrate


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--N_iters", type=int, default=200000,
                    help='number of training iterations')
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help='gradient clipping threshold (0 to disable)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--learnable_pe", action='store_true',
                        help='use learnable positional encoding (Fourier features)')
    parser.add_argument("--learnable_pe_phase", action='store_true',
                        help='make phase shifts learnable in positional encoding')
    parser.add_argument("--pe_learnable_freqs", type=lambda x: str(x).lower() in ['true', '1', 'yes', 'on'], default=True,
                        help='make frequency bands learnable (default: True, set False/0/no/off to disable)')
    parser.add_argument("--pe_init_scale", type=float, default=1.0,
                        help='initial scale for frequency initialization (default: 1.0)')
    parser.add_argument("--pe_lr_scale", type=float, default=0.5,
                        help='learning rate scale for PE parameters relative to network (default: 0.5)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--use_wandb", action='store_true',
                        help='use wandb for logging instead of tensorboard')
    parser.add_argument("--wandb_project", type=str, default='nerf',
                        help='wandb project name')
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help='wandb entity/team name (optional)')
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help='wandb run name (optional, defaults to expname)')
    
    # Positional encoding type and Gaussian Fourier settings
    parser.add_argument("--pe_type", type=str, default="baseline",
                        choices=["baseline", "learnable", "gaussian"],
                        help="positional encoding type")
    parser.add_argument("--pe_sigma", type=float, default=10.0,
                        help="sigma for Gaussian Fourier features")
    parser.add_argument("--pe_num_feats", type=int, default=30,
                        help="number of Gaussian Fourier features per input dim (gamma(x) = [x, sin, cos])")


    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, embed_fn, embeddirs_fn, pe_lrate = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Initialize logging (wandb or tensorboard)
    if args.use_wandb:
        wandb_run_name = args.wandb_run_name if args.wandb_run_name is not None else expname
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=vars(args),
            dir=basedir
        )
        writer = None
        print(f'Initialized wandb logging: project={args.wandb_project}, name={wandb_run_name}')
    else:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
        print(f'Initialized tensorboard logging: {os.path.join(basedir, "summaries", expname)}')
    #Metrics init!!
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                epoch = i_batch // rays_rgb.shape[0]
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0
                # Log data shuffle event to wandb
                if args.use_wandb:
                    wandb.log({'train/data_shuffled': 1, 'train/epoch': epoch}, step=global_step)

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        
        # Gradient clipping for stability (helps with learnable PE)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(grad_vars, args.grad_clip)
        
        # Check for NaN/Inf gradients and compute gradient norm
        grad_norm = 0.0
        has_nan = False
        for param in grad_vars:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan = True
        grad_norm = grad_norm ** (1. / 2)
        if has_nan:
            print(f"WARNING: NaN/Inf gradients detected at iteration {i}!")
            if args.use_wandb:
                wandb.log({'train/grad_nan_warning': 1}, step=global_step)
        
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        
        # Update learning rates for each parameter group
        if args.learnable_pe and len(optimizer.param_groups) > 1:
            # Separate learning rates for network and PE
            new_pe_lrate = pe_lrate * (decay_rate ** (global_step / decay_steps))
            # First group is network, second is PE (based on how we created optimizer)
            optimizer.param_groups[0]['lr'] = new_lrate
            optimizer.param_groups[1]['lr'] = new_pe_lrate
        else:
            # Single learning rate for all parameters
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Log all training metrics to wandb every step
        if args.use_wandb:
            log_dict = {
                'train/loss': loss.item(),
                'train/psnr': psnr.item(),
                'train/grad_norm': grad_norm,
                'train/learning_rate': new_lrate,
                'train/time_per_iter': dt,
            }
            if args.N_importance > 0:
                log_dict['train/psnr0'] = psnr0.item()
            if args.learnable_pe and len(optimizer.param_groups) > 1:
                log_dict['train/pe_learning_rate'] = new_pe_lrate
            
            # Add GPU memory usage if available
            if torch.cuda.is_available():
                log_dict['system/gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
                log_dict['system/gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            
            wandb.log(log_dict, step=global_step)

        # Rest is logging
        # Save checkpoints every 10000 steps
        if i % 10000 == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))

            # 先只保存一定存在的东西
            ckpt_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }

            # 如果有 fine 网络，再额外保存
            network_fine = render_kwargs_train.get('network_fine', None)
            if network_fine is not None:
                ckpt_dict['network_fine_state_dict'] = network_fine.state_dict()

            # 如果用了可学习 PE，也一并保存
            if args.learnable_pe:
                if isinstance(embed_fn, nn.Module):
                    ckpt_dict['embed_fn_state_dict'] = embed_fn.state_dict()
                if embeddirs_fn is not None and isinstance(embeddirs_fn, nn.Module):
                    ckpt_dict['embeddirs_fn_state_dict'] = embeddirs_fn.state_dict()

            # 真正写文件
            torch.save(ckpt_dict, path)
            print('Saved checkpoints at', path)

            # 同步到 wandb
            if args.use_wandb:
                wandb.save(path, base_path=basedir)
                wandb.log({'checkpoint/saved': 1, 'checkpoint/step': global_step}, step=global_step)
                print(f'Uploaded checkpoint to wandb: step {global_step}')

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            rgb_video_path = moviebase + 'rgb.mp4'
            disp_video_path = moviebase + 'disp.mp4'
            imageio.mimwrite(rgb_video_path, to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(disp_video_path, to8b(disps / np.max(disps)), fps=30, quality=8)

            if args.use_viewdirs:
                render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
                with torch.no_grad():
                    rgbs_still, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
                render_kwargs_test['c2w_staticcam'] = None
                still_video_path = moviebase + 'rgb_still.mp4'
                imageio.mimwrite(still_video_path, to8b(rgbs_still), fps=30, quality=8)
            
            # Log videos to wandb
            if args.use_wandb:
                log_dict = {
                    'video/spiral_rgb': wandb.Video(rgb_video_path),
                    'video/spiral_disp': wandb.Video(disp_video_path),
                }
                if args.use_viewdirs:
                    log_dict['video/spiral_still'] = wandb.Video(still_video_path)
                wandb.log(log_dict, step=global_step)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')
            
            # Log test set render event to wandb
            if args.use_wandb:
                wandb.log({'testset/rendered': 1, 'testset/step': global_step, 'testset/path': testsavedir}, step=global_step)


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item():.6f}  PSNR: {psnr.item():.2f}  GradNorm: {grad_norm:.4f}  LR: {new_lrate:.6f}")
            # Logging to tensorboard (wandb already logged every step above)
            if not args.use_wandb:
                log_dict = {
                    'train/loss': loss.item(),
                    'train/psnr': psnr.item(),
                    'train/grad_norm': grad_norm,
                    'train/learning_rate': new_lrate,
                }
                if args.N_importance > 0:
                    log_dict['train/psnr0'] = psnr0.item()
                writer.add_scalar('train/loss', log_dict['train/loss'], global_step)
                writer.add_scalar('train/psnr', log_dict['train/psnr'], global_step)
                writer.add_scalar('train/grad_norm', log_dict['train/grad_norm'], global_step)
                writer.add_scalar('train/learning_rate', log_dict['train/learning_rate'], global_step)
                if args.N_importance > 0:
                    writer.add_scalar('train/psnr0', log_dict['train/psnr0'], global_step)

        if i%args.i_img==0:
            # Log a rendered validation view (wandb or tensorboard)
            img_i=np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            with torch.no_grad():
                rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, c2w=pose,
                                                    **render_kwargs_test)

            psnr_holdout = mse2psnr(img2mse(rgb, target))
            rgb_images = torch.clamp(rgb.permute(2,0,1).unsqueeze(0),0,1)
            target_images = torch.clamp(target.permute(2,0,1).unsqueeze(0),0,1)
            with torch.no_grad():
                lpips_val = lpips_fn(rgb_images, target_images).item()
                ssim_val = ssim_fn(rgb_images, target_images).item()
                # --- Additional Metrics ---
                rgb_np = rgb.cpu().numpy()
                target_np = target.cpu().numpy()
                # EPI sharpness
                epi_sharp = compute_epi_sharpness(rgb_np)
                # Fourier metrics
                hf_ratio, psd = compute_fft_metrics(rgb_np)
                # Canny F1
                canny_f1 = compute_canny_f1(rgb_np, target_np)
                # FSIM / GMSD / VIF (piq expects NCHW tensors)
                pred_t = rgb.permute(2,0,1).unsqueeze(0) 
                tgt_t = target.permute(2,0,1).unsqueeze(0)
                extra_metrics = compute_additional_metrics(pred_t, tgt_t)
            # Log images and metrics (wandb or tensorboard)
            if args.use_wandb:
                log_dict = {
                    'val/psnr_holdout': psnr_holdout.item(),
                    'val/lpips': lpips_val,
                    'val/ssim': ssim_val,
                    'val/rgb': wandb.Image(to8b(rgb.cpu().numpy())),
                    'val/rgb_holdout': wandb.Image(to8b(target.cpu().numpy())),
                    'val/disp': wandb.Image(disp.cpu().numpy()),
                    'val/acc': wandb.Image(acc.cpu().numpy()),
                    'val/epi_sharpness': epi_sharp,
                    'val/hf_ratio': hf_ratio,
                    'val/canny_f1': canny_f1,
                    'val/fsim': extra_metrics['fsim'],
                    'val/gmsd': extra_metrics['gmsd'],
                    'val/vif': extra_metrics['vif'],
                    'val/psd': wandb.Image(psd/psd.max()),
                }
                if args.N_importance > 0:
                    log_dict['val/rgb0'] = wandb.Image(to8b(extras['rgb0'].cpu().numpy()))
                    log_dict['val/disp0'] = wandb.Image(extras['disp0'].cpu().numpy())
                    log_dict['val/z_std'] = wandb.Image(extras['z_std'].cpu().numpy())
                wandb.log(log_dict, step=global_step)
            else:
                # TensorBoard logging (HWC format)
                writer.add_image('val/rgb', to8b(rgb.cpu().numpy()), global_step, dataformats='HWC')
                writer.add_image('val/disp', disp.cpu().numpy(), global_step, dataformats='HW')
                writer.add_image('val/acc', acc.cpu().numpy(), global_step, dataformats='HW')
                writer.add_scalar('val/psnr_holdout', psnr_holdout.item(), global_step)
                writer.add_image('val/rgb_holdout', to8b(target.cpu().numpy()), global_step, dataformats='HWC')
                writer.add_scalar('val/lpips', lpips_val, global_step)
                writer.add_scalar('val/ssim', ssim_val, global_step)
                writer.add_scalar('val/epi_sharpness', epi_sharp, global_step)
                writer.add_scalar('val/hf_ratio', hf_ratio, global_step)
                writer.add_scalar('val/canny_f1', canny_f1, global_step)
                writer.add_scalar('val/fsim', extra_metrics['fsim'], global_step)
                writer.add_scalar('val/gmsd', extra_metrics['gmsd'], global_step)
                writer.add_scalar('val/vif', extra_metrics['vif'], global_step)
                writer.add_image('val/psd', psd/psd.max(), global_step, dataformats='HW')
                if args.N_importance > 0:
                    writer.add_image('val/rgb0', to8b(extras['rgb0'].cpu().numpy()), global_step, dataformats='HWC')
                    writer.add_image('val/disp0', extras['disp0'].cpu().numpy(), global_step, dataformats='HW')
                    writer.add_image('val/z_std', extras['z_std'].cpu().numpy(), global_step, dataformats='HW')
        

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()

