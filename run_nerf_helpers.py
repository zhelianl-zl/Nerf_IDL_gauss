import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# Learnable Fourier Feature Encoding
class LearnableFourierEmbedder(nn.Module):
    def __init__(self, input_dims=3, num_freqs=10, include_input=True, 
                 learnable_freqs=True, learnable_phase=False, init_scale=1.0):
        """
        Learnable Fourier Feature Encoding for NeRF
        
        Args:
            input_dims: Input dimension (3 for xyz coordinates)
            num_freqs: Number of frequency bands
            include_input: Whether to include raw input in output
            learnable_freqs: Make frequency bands learnable
            learnable_phase: Make phase shifts learnable
            init_scale: Initial scale for frequency initialization
        """
        super(LearnableFourierEmbedder, self).__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.learnable_freqs = learnable_freqs
        self.learnable_phase = learnable_phase
        
        # Initialize frequency bands (log-spaced like original NeRF)
        freq_bands = 2.**torch.linspace(0., num_freqs-1, steps=num_freqs) * init_scale
        
        if learnable_freqs:
            # Make frequencies learnable parameters
            self.freq_bands = nn.Parameter(freq_bands)
        else:
            # Keep frequencies fixed
            self.register_buffer('freq_bands', freq_bands)
        
        # Initialize phase shifts to zero
        if learnable_phase:
            # Separate phase for sin and cos, for each frequency and dimension
            self.phase_shifts = nn.Parameter(torch.zeros(num_freqs, input_dims, 2))
        else:
            self.register_buffer('phase_shifts', torch.zeros(num_freqs, input_dims, 2))
        
        # Calculate output dimension
        out_dim = 0
        if include_input:
            out_dim += input_dims
        out_dim += num_freqs * input_dims * 2  # *2 for sin and cos
        self.out_dim = out_dim
        
    def forward(self, inputs):
        """
        Args:
            inputs: [..., input_dims] input coordinates
        Returns:
            [..., out_dim] encoded features
        """
        outputs = []
        
        if self.include_input:
            outputs.append(inputs)
        
        # Apply learnable Fourier features
        for i, freq in enumerate(self.freq_bands):
            # Compute frequency-scaled inputs: [..., input_dims]
            scaled_inputs = inputs * freq
            
            if self.learnable_phase:
                # Add learnable phase shifts
                sin_phase = self.phase_shifts[i, :, 0]  # [input_dims]
                cos_phase = self.phase_shifts[i, :, 1]  # [input_dims]
                outputs.append(torch.sin(scaled_inputs + sin_phase))
                outputs.append(torch.cos(scaled_inputs + cos_phase))
            else:
                outputs.append(torch.sin(scaled_inputs))
                outputs.append(torch.cos(scaled_inputs))
        
        return torch.cat(outputs, -1)
    

def get_embedder(multires,
                 i=0,
                 learnable: bool = False,
                 learnable_freqs: bool = True,
                 learnable_phase: bool = False,
                 init_scale: float = 1.0,
                 pe_type: str = 'baseline',
                 gaussian_num_feats: int = 30,
                 gaussian_sigma: float = 10.0):
    """
    pe_type:
      - 'baseline' : 原始 NeRF Embedder（或者你队友的 LearnableFourierEmbedder 开关由 learnable 控）
      - 'learnable': 强制使用 LearnableFourierEmbedder
      - 'gaussian' : 使用 Gaussian Fourier Features（RFF）
    """
    # i == -1: 不做编码，直接 Identity
    if i == -1:
        return nn.Identity(), 3

    # 1) Gaussian Fourier Features 分支
    if pe_type == 'gaussian':
        embedder_obj = GaussianFourierEmbedder(
            input_dims=3,
            num_feats=gaussian_num_feats,
            sigma=gaussian_sigma,
            include_input=True
        )
        return embedder_obj, embedder_obj.out_dim

    # 2) Learnable Fourier 分支（仍然保留你队友版本）
    if learnable or pe_type == 'learnable':
        # 注意：这里用的是你队友已经写好的 LearnableFourierEmbedder
        embedder_obj = LearnableFourierEmbedder(
            input_dims=3,
            num_freqs=multires,
            include_input=True,
            learnable_freqs=learnable_freqs,
            learnable_phase=learnable_phase,
            init_scale=init_scale
        )
        return embedder_obj, embedder_obj.out_dim

    # 3) baseline 分支：这里保留你项目中原本的 Embedder 实现
    # === baseline: 你现在已有的实现 ===
    # 下面这段是“模板”，你需要用自己项目的代码替换掉里面的细节
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    #embedder_obj.create_embed_fns()
    return embedder_obj.embed, embedder_obj.out_dim
    # === baseline 部分结束 ===



import math
import torch
import torch.nn as nn

class GaussianFourierEmbedder(nn.Module):
    """
    Gaussian Fourier Features / Random Fourier Features encoding.

    gamma(x) = [x, sin(2π Bx), cos(2π Bx)]
    where B ~ N(0, sigma^2 I), sigma controls frequency bandwidth.
    """
    def __init__(self,
                 input_dims: int = 3,
                 num_feats: int = 30,
                 sigma: float = 10.0,
                 include_input: bool = True):
        super().__init__()
        self.input_dims = input_dims
        self.num_feats = num_feats
        self.sigma = sigma
        self.include_input = include_input

        # B ~ N(0, sigma^2 I)
        B = torch.randn(num_feats, input_dims) * sigma
        # 固定 B，不让它训练（经典 RFF 做法）
        self.register_buffer("B", B)

        out_dim = 0
        if include_input:
            out_dim += input_dims
        # sin + cos
        out_dim += 2 * num_feats
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., input_dims]
        return: [..., out_dim]
        """
        # [..., num_feats]
        proj = 2 * math.pi * (x @ self.B.t())
        sin = torch.sin(proj)
        cos = torch.cos(proj)

        outs = []
        if self.include_input:
            outs.append(x)
        outs.extend([sin, cos])

        return torch.cat(outs, dim=-1)


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
