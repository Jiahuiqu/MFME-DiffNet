from matplotlib import pyplot as plt
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange, repeat
# from tqdm.notebook import tqdm
from functools import partial
import math, os, copy
from tqdm import tqdm
from prettytable import PrettyTable
import scipy.io as sio
import imgvision as iv
from dataset import *
from torch.utils.data import DataLoader
from model import Fusemodel
import torchvision
from method import hyperspectral_pca, compute_direction, calculate_rotation_angle, rotate_image_parallel, ssim
import cv2
import matplotlib
# matplotlib.use('Agg')
import time

"""
    Define U-net Architecture:
    Approximate reverse diffusion process by using U-net
    U-net of SR3 : U-net backbone + Positional Encoding of time + Multihead Self-Attention
"""

import torch
import torch.nn as nn


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


def calculate_sam(target_data, reference_data):
    # 归一化目标数据和参考数据
    b, c, h, w = target_data.shape
    target_data = target_data.reshape(b, c, h * w).permute(0, 2, 1)
    reference_data = reference_data.reshape(b, c, h * w).permute(0, 2, 1)
    target_data_norm = torch.nn.functional.normalize(target_data, dim=2)
    reference_data_norm = torch.nn.functional.normalize(reference_data, dim=2)

    # 计算点积
    dot_product = torch.einsum('bnc,bnc->bn', target_data_norm, reference_data_norm)

    # 计算长度乘积
    length_product = torch.norm(target_data_norm, dim=2) * torch.norm(reference_data_norm, dim=2)

    # 计算SAM光谱角
    sam = torch.acos(dot_product / length_product)
    sam_mean = torch.mean(torch.mean(sam, dim=1))
    return sam_mean


def extract(a, t, x_shape):
    """
    从给定的张量a中检索特定的元素。t是一个包含要检索的索引的张量，
    这些索引对应于a张量中的元素。这个函数的输出是一个张量，
    包含了t张量中每个索引对应的a张量中的元素
    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        # Input : tensor of value of coefficient alpha at specific step of diffusion process e.g. torch.Tensor([0.03])
        # Transform level of noise into representation of given desired dimension
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(nn.Linear(in_channels, out_channels * (1 + self.use_affine_level)))

    def forward(self, x, noise_embed):
        noise = self.noise_func(noise_embed).view(x.shape[0], -1, 1, 1)
        if self.use_affine_level:
            gamma, beta = noise.chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + noise
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


# Linear Multi-head Self-attention
class SelfAtt(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32, att_num=0):
        super(SelfAtt, self).__init__()
        self.groupnorm = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)
        self.att = att_num

    def forward(self, x):
        x_org = x
        b, c, h, w = x.size()
        x = self.groupnorm(x)
        qkv = rearrange(self.qkv(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        keys = F.softmax(keys, dim=-1)
        att = torch.einsum('bhdn,bhen->bhde', keys, values)
        out = torch.einsum('bhde,bhdn->bhen', att, queries)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

        return x_org + self.att * self.proj(out)


class Cross_Att(nn.Module):
    def __init__(self, channel_dim, num_heads, norm_groups=32, att_num=0):
        super(Cross_Att, self).__init__()
        self.att = att_num
        self.groupnorm_1 = nn.GroupNorm(norm_groups, channel_dim)
        self.groupnorm_2 = nn.GroupNorm(norm_groups, channel_dim)
        self.num_heads = num_heads
        self.qkv_1 = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)
        self.qkv_2 = nn.Conv2d(channel_dim, channel_dim * 3, 1, bias=False)

        self.proj = nn.Conv2d(channel_dim, channel_dim, 1)

        self.downsample = nn.Sequential(nn.Conv2d(channel_dim, 2 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(2 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(2 * channel_dim, 4 * channel_dim, 3, 1, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),

                                        nn.Conv2d(4 * channel_dim, 4 * channel_dim, 3, 2, 1),
                                        nn.Upsample(scale_factor=0.5, mode='bicubic'),
                                        )

        self.upsample = nn.Sequential(nn.Conv2d(2 * channel_dim, 1 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(2 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(4 * channel_dim, 2 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),

                                      nn.Conv2d(4 * channel_dim, 4 * channel_dim, 3, 1, 1),
                                      nn.Upsample(scale_factor=2, mode='bicubic'),
                                      )

    def forward(self, x, y, mode):
        b, c, h, w = x.size()
        x_org = x
        if mode == 'spe':
            b, c, h, w = x.size()
            x = self.groupnorm_1(x)
            y = self.groupnorm_1(y)
            qkv_1 = rearrange(self.qkv_1(x), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads,
                              qkv=3)
            queries_1, keys_1, values_1 = qkv_1[0], qkv_1[1], qkv_1[2]
            qkv_2 = rearrange(self.qkv_2(y), "b (qkv heads c) h w -> (qkv) b heads c (h w)", heads=self.num_heads,
                              qkv=3)
            queries_2, keys_2, values_2 = qkv_2[0], qkv_2[1], qkv_2[2]
            keys_1 = F.softmax(keys_1, dim=-1)
            keys_2 = F.softmax(keys_2, dim=-1)
            att = torch.einsum('bhdn,bhen->bhde', keys_1, values_2)
            out = torch.einsum('bhde,bhdn->bhen', att, queries_1)
            out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.num_heads, h=h, w=w)
        else:
            x = self.groupnorm_2(x)
            y = self.groupnorm_2(y)
            if h == 512:
                times = h / 64
            else:
                times = h / 20
            n = np.log(times) / np.log(2)
            for i in range(int(n)):
                x = self.downsample[2 * i](x)
                x = self.downsample[2 * i + 1](x)

            for i in range(int(n)):
                y = self.downsample[2 * i](y)
                y = self.downsample[2 * i + 1](y)

            b, c, h, w = x.size()

            x = x.reshape(b, c, h * w).repeat(1, 1, 3)
            y = y.reshape(b, c, h * w).repeat(1, 1, 3)
            qkv_1 = rearrange(x, "b c (qkv heads h) -> (qkv) b heads h c", heads=self.num_heads, qkv=3)
            queries_1, keys_1, values_1 = qkv_1[0], qkv_1[1], qkv_1[2]
            qkv_2 = rearrange(y, "b c (qkv heads h) -> (qkv) b heads h c", heads=self.num_heads, qkv=3)
            queries_2, keys_2, values_2 = qkv_2[0], qkv_2[1], qkv_2[2]

            keys_1 = F.softmax(keys_1, dim=-1)
            keys_2 = F.softmax(keys_2, dim=-1)
            att = torch.einsum('bhdn,bhen->bhde', keys_1, values_2)
            out = torch.einsum('bhde,bhdn->bhen', att, queries_1)
            out = rearrange(out, 'b heads (h w) c -> b (heads c) h w', heads=self.num_heads, h=h, w=w)

            for i in range(int(n)):
                l = int(n) - 1 - i
                out = self.upsample[2 * l](out)
                out = self.upsample[2 * l + 1](out)

        return x_org + self.att * self.proj(out)


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=False):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        y = self.block1(x)
        y = self.noise_func(y, time_emb)
        y = self.block2(y)
        x = y + self.res_conv(x)
        return x


class ResBlock_skip(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=True):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        y = self.block1(x)

        return y + self.res_conv(x)


class SGPD(nn.Module):
    def __init__(self, in_channel=37, out_channel=34, skip_input=31, inner_channel=64, norm_groups=32,
                 channel_mults=[1, 2, 4, 8, 8], res_blocks=3, dropout=0, img_size=128):
        super().__init__()

        self_att = []
        cros_att = []
        dim_out = [inner_channel, inner_channel * 2, inner_channel * 2]
        for i in reversed(range(len(dim_out))):
            self_att.append(SelfAtt(dim_out[i], num_heads=1, norm_groups=norm_groups))

        self.self_att = nn.ModuleList(self_att)

        for j in reversed(range(len(dim_out))):
            cros_att.append(Cross_Att(dim_out[j], num_heads=1, norm_groups=norm_groups))

        self.cros_att = nn.ModuleList(cros_att)

        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        pre_channel_skip = inner_channel
        feat_channels = [pre_channel]
        feat_channels_skips = [pre_channel]

        now_res = img_size

        # Downsampling stage of SGPD
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResBlock(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout),
            ResBlock(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                     norm_groups=norm_groups, dropout=dropout, att=False)
        ])

        # Skip stage of SGPD
        skip_downs = [nn.Conv2d(skip_input, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                skip_downs.append(ResBlock_skip(
                    pre_channel_skip, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, att=False))
                pre_channel_skip = channel_mult
            if not is_last:
                feat_channels_skips.append(channel_mult)
                skip_downs.append(Downsample(pre_channel_skip))
                now_res = now_res // 2
        self.skip_downs = nn.ModuleList(skip_downs)

        # Upsampling stage of SGPD
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]

            for i in range(0, res_blocks + 1):
                ups.append(ResBlock(
                    pre_channel + feat_channels.pop() * 2, channel_mult,
                    noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult

            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

    def forward(self, x, skip_input, noise_level, mode=None):
        # Embedding of time step with noise coefficient alpha
        t = self.noise_level_mlp(noise_level)

        feats_skip = []
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        k = 0
        for i, layer in enumerate(self.skip_downs):
            # skip_input =
            skip_input = layer(skip_input)
            feats_skip.append(skip_input)

        for layer in self.mid:
            x = layer(x, t)
        z = 0
        for i, layer in enumerate(self.ups):

            if isinstance(layer, ResBlock):
                if i == 0:
                    x = layer(torch.cat([x, feats.pop(), feats_skip.pop()], dim=1), t)
                elif isinstance(self.ups[i - 1], Upsample):
                    temp_feats_skip = feats_skip.pop()
                    temp_feats = feats.pop()
                    x = layer(torch.cat([x, self.self_att[z](temp_feats_skip),
                                         self.cros_att[z](temp_feats, temp_feats_skip, mode=mode)], dim=1), t)
                    z = z + 1
                else:
                    x = layer(torch.cat([x, feats.pop(), feats_skip.pop()], dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


"""
    Define Diffusion process framework to train desired model:
    Forward Diffusion process:
        Given original image x_0, apply Gaussian noise ε_t for each time step t
        After proper length of time step, image x_T reachs to pure Gaussian noise
    Objective of model f :
        model f is trained to predict actual added noise ε_t for each time step t
"""


class Diffusion(nn.Module):
    def __init__(self, model, device, img_size, LR_size, channels=4):
        super().__init__()
        self.channels = channels
        self.model = model.to(device)
        self.img_size = img_size
        self.LR_size = LR_size
        self.device = device

        # self.upSample = nn.Upsample(scale_factor=4, mode='bicubic')
        self.downSample = nn.Upsample(scale_factor=0.25, mode='bicubic')
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')

        # complementary fusion block
        self.fuse = nn.Sequential(
            nn.Conv2d(31 * 2, 31 * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(31 * 2, 31, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(31, 31, kernel_size=3, stride=1, padding=1),
        ).to(device)

    def set_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError()

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2):
        if schedule == 'linear':
            betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
        elif schedule == 'warmup':
            warmup_frac = 0.1
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
            warmup_time = int(n_timestep * warmup_frac)
            betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
        elif schedule == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = self.make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']
        )
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        self.num_timesteps = int(len(betas))
        # Coefficient for forward diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('pred_coef1', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('pred_coef2', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Coefficient for reverse diffusion posterior q(x_{t-1} | x_t, x_0)
        variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('variance', to_torch(variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # Predict desired image x_0 from x_t with noise z_t -> Output is predicted x_0
    def predict_start(self, x_t, t, noise):
        return self.pred_coef1[t] * x_t - self.pred_coef2[t] * noise

    # Compute mean and log variance of posterior(reverse diffusion process) distribution
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, gtHS=None, lrHS=None, patch_1=None, patch_2=None, patch_3=None,
                        sim_1=None, sim_2=None, sim_3=None):
        batch_size, c = x.shape[0], gtHS.shape[1]
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)
        x_start = self.model(torch.cat([lrHS, x], dim=1), patch_1, patch_2, patch_3, noise_level, sim_1, sim_2, sim_3)

        posterior_mean = (
                self.posterior_mean_coef1[t] * x_start.clamp(-1, 1) +
                self.posterior_mean_coef2[t] * x
        )

        posterior_variance = self.posterior_log_variance_clipped[t]

        mean, posterior_log_variance = posterior_mean, posterior_variance
        return mean, posterior_log_variance

    # Progress single step of reverse diffusion process
    # Given mean and log variance of posterior, sample reverse diffusion result from the posterior
    @torch.no_grad()
    def p_sample(self, img_noise, t, clip_denoised=True, gtHS=None, lrHS=None, patch_1=None, patch_2=None, patch_3=None,
                 sim_1=None, sim_2=None, sim_3=None):

        mean, log_variance = self.p_mean_variance(x=img_noise, t=t, clip_denoised=clip_denoised, gtHS=gtHS, lrHS=lrHS,
                                                  patch_1=patch_1, patch_2=patch_2, patch_3=patch_3, sim_1=sim_1,
                                                  sim_2=sim_2, sim_3=sim_3)
        noise = torch.randn_like(img_noise) if t > 0 else torch.zeros_like(img_noise)
        return mean + noise * (0.5 * log_variance).exp()

    # Progress whole reverse diffusion process
    @torch.no_grad()
    def super_resolution(self, gtHS, lrHS, patch_1_org, patch_2_org, patch_3_org):

        patch_1, sim_1 = match_method(lrHS, patch_1_org, 0.5)
        patch_2, sim_2 = match_method(lrHS, patch_2_org, 1)
        patch_3, sim_3 = match_method(lrHS, patch_3_org, 2)

        gtHS = gtHS.type(torch.float32).to(DEVICE)
        lrHS = lrHS.type(torch.float32).to(DEVICE)
        img_noise = torch.rand_like(gtHS, device=gtHS.device)
        for i in tqdm(reversed(range(0, self.num_timesteps, 10)), desc="Processing Timesteps"):
            if (i < 1500):
                patch_1, sim_1 = match_method(torch.cat([lrHS.cpu(), img_noise.cpu()], dim=1), patch_1_org, 0.5)
                patch_2, sim_2 = match_method(torch.cat([lrHS.cpu(), img_noise.cpu()], dim=1), patch_2_org, 1)
                patch_3, sim_3 = match_method(torch.cat([lrHS.cpu(), img_noise.cpu()], dim=1), patch_3_org, 2)
            img = self.p_sample(img_noise, i, True, gtHS, lrHS, patch_1, patch_2, patch_3, sim_1, sim_2, sim_3)
        return img

    def net(self, gtHS, lrHS, patch_1, patch_2, patch_3, sim_1, sim_2, sim_3):

        gtHS = gtHS
        b, c, h, w = gtHS.shape
        t = torch.randint(1, schedule_opt['n_timestep'], size=(b,))
        sqrt_alpha_cumprod_t = extract(torch.from_numpy(self.sqrt_alphas_cumprod_prev), t, gtHS.shape)
        sqrt_alpha = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1).type(torch.float32).to(gtHS.device)
        noise = torch.randn_like(gtHS).to(gtHS.device)
        # Perturbed image obtained by forward diffusion process at random time step t
        x_noisy = sqrt_alpha * gtHS + (1 - sqrt_alpha ** 2).sqrt() * noise
        # The bilateral model predict actual x0 added at time step t
        outputs = self.model(torch.cat([lrHS, x_noisy], 1), patch_1, patch_2, patch_3, sqrt_alpha, sim_1, sim_2, sim_3)

        # complementary fusion
        Loss = self.loss_func(outputs, gtHS)
        Loss = Loss / (gtHS.shape[0] * gtHS.shape[1] * gtHS.shape[2] * gtHS.shape[3])
        return Loss

    def forward(self, gtHS, lrHS, patch_1, patch_2, patch_3, sim_1, sim_2, sim_3, *args, **kwargs):

        return self.net(gtHS, lrHS, patch_1, patch_2, patch_3, sim_1, sim_2, sim_3)


def match_method(lrHS, match_img, scale):
    lrHS = lrHS.numpy()
    match_img = match_img.numpy()
    img = torch.rand_like(torch.from_numpy(match_img[:, 0, :, :, :]))
    sim = torch.zeros(lrHS.shape[0])
    for i in range(lrHS.shape[0]):
        lrHS_i = lrHS[i, :, :, :]
        match_img_i = match_img[i, :, :, :]
        ssim_max = 0
        # print("i:{}".format(i))
        for j in range(match_img.shape[1]):
            # print("j:{}".format(j))
            matching_patch = hyperspectral_pca(match_img_i[j, :, :, :].transpose(1, 2, 0))
            MAX = np.max(matching_patch)
            MIN = np.min(matching_patch)
            matching_patch = (matching_patch - MIN) / (MAX - MIN)

            HSI_pca = hyperspectral_pca(lrHS_i.transpose(1, 2, 0))

            mode_val_matching, count_matching = compute_direction(matching_patch)
            mode_val, count = compute_direction(HSI_pca)
            angle = calculate_rotation_angle(mode_val_matching, mode_val)
            HSI_matching_ro = rotate_image_parallel(matching_patch, angle)

            matching_patch_up_2 = cv2.resize(HSI_matching_ro,
                                             (int(matching_patch.shape[0] * (1 / scale)),
                                              int(matching_patch.shape[1] * (1 / scale))),
                                             interpolation=cv2.INTER_LINEAR)
            # 高光谱图像
            HSI_pca_upsample_2 = cv2.resize(HSI_pca, (int(HSI_pca.shape[0] * scale), int(HSI_pca.shape[1] * scale)),
                                            interpolation=cv2.INTER_LINEAR)

            SSIM1 = ssim(matching_patch * 0.5, HSI_pca_upsample_2 * 0.5)
            SSIM2 = ssim(matching_patch_up_2 * 0.5, HSI_pca * 0.5)

            out = 0.5 * SSIM1 + 0.5 * SSIM2
            if (out > ssim_max):
                img[i, :, :, :] = torch.from_numpy(match_img_i[j, :, :, :])
                sim[i] = out
    return img.type(torch.float32).to(DEVICE), sim.unsqueeze(1).unsqueeze(1).unsqueeze(1).type(torch.float32).to(DEVICE)


# Class to train & test desired model
class SR3():
    def __init__(self, device, img_size, LR_size, loss_type, dataloader, testloader,
                 schedule_opt, save_path, load_path=None, load=True,
                 in_channel=62, out_channel=31, inner_channel=64, norm_groups=8,
                 channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, lr=1e-3, distributed=False):
        super(SR3, self).__init__()
        self.dataloader = dataloader
        self.testloader = testloader
        self.device = device
        self.save_path = save_path
        self.img_size = img_size
        self.LR_size = LR_size

        model = Fusemodel(in_c=102).to(device=DEVICE)

        self.sr3 = Diffusion(model, device, img_size, LR_size, out_channel)
        # Apply weight initialization & set loss & set noise schedule
        # self.sr3.apply(self.weights_init_orthogonal)
        self.sr3.set_loss(loss_type)
        self.sr3.set_new_noise_schedule(schedule_opt)

        if distributed:
            assert torch.cuda.is_available()
            self.sr3 = nn.DataParallel(self.sr3)

        self.optimizer = torch.optim.Adam(self.sr3.parameters(), lr=lr)

        params = sum(p.numel() for p in self.sr3.parameters())
        print(f"Number of model parameters : {params}")

        if load:
            self.load(load_path)

    def weights_init_orthogonal(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    def train(self, epoch, verbose):

        train = True
        for i in range(epoch):
            i = i
            train_loss = 0
            self.sr3.train()
            randn1 = np.random.randint(0, 100)
            avg_time = 0
            if train:
                for step, [gtHS, lrHS, patch_1, patch_2, patch_3] in enumerate(tqdm(self.dataloader)):
                    # 高光谱和全色图像
                    # print("step:{}".format(step))
                    start = time.time()
                    patch_1, sim_1 = match_method(lrHS, patch_1, 0.5)
                    patch_2, sim_2 = match_method(lrHS, patch_2, 1)
                    patch_3, sim_3 = match_method(lrHS, patch_3, 2)
                    end = time.time()
                    match_used_time = end - start
                    print(match_used_time)
                    avg_time += match_used_time

                    gtHS = gtHS.type(torch.float32).to(DEVICE)
                    lrHS = lrHS.type(torch.float32).to(DEVICE)

                    # patch_1 = patch_1.type(torch.float32).to(DEVICE)
                    # patch_2 = patch_2.type(torch.float32).to(DEVICE)
                    # patch_3 = patch_3.type(torch.float32).to(DEVICE)

                    self.optimizer.zero_grad()
                    loss = self.sr3(gtHS, lrHS, patch_1, patch_2, patch_3, sim_1, sim_2, sim_3)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                print('epoch: {}'.format(i))
                print('matching used average time', avg_time / float(len(self.dataloader)))
                print('损失函数:')
                x = PrettyTable()
                x.add_column("loss", ['value'])
                x.add_column("loss_all", [train_loss / float(len(self.dataloader))])
                print(x)

            if (i + 1) % verbose == 0:
                self.sr3.eval()
                test_data = copy.deepcopy(next(iter(self.testloader)))

                [gtHS, lrHS, patch_1, patch_2, patch_3] = test_data
                b, c, h, w = gtHS.shape

                randn3 = np.random.randint(0, b)
                gtHS = gtHS[randn3]
                lrHS = lrHS[randn3]
                patch_1 = patch_1[randn3]
                patch_2 = patch_2[randn3]
                patch_3 = patch_3[randn3]
                # Transform to low-resolution images
                # Save example of test images to check training

                plt.figure(figsize=(15, 10))
                plt.subplot(2, 3, 1)
                plt.axis("off")
                plt.title("gtHS")
                plt.imshow(
                    np.transpose(torchvision.utils.make_grid(gtHS, nrow=2, padding=1, normalize=True).cpu(), (1, 2, 0))[
                    :, :, [45, 30, 15]])

                plt.subplot(2, 3, 2)
                plt.axis("off")
                plt.title("lrHS")
                # A = self.test(test_img, test_lrHS_img)
                plt.imshow(
                    np.transpose(torchvision.utils.make_grid(lrHS.cpu(), nrow=2, padding=1, normalize=True), (1, 2, 0))[
                    :, :, [45, 30, 15]])

                plt.subplot(2, 3, 3)
                plt.axis("off")
                plt.title("gen_img")
                self.save(self.save_path, i)

                fuse_result = self.test(gtHS.unsqueeze(0), lrHS.unsqueeze(0), patch_1.unsqueeze(0),
                                        patch_2.unsqueeze(0), patch_3.unsqueeze(0))
                plt.imshow(np.transpose(torchvision.utils.make_grid(fuse_result.cpu(),
                                                                    nrow=2, padding=1, normalize=True), (1, 2, 0))[:, :,
                           [45, 30, 15]])

                plt.subplot(2, 3, 4)
                plt.axis("off")
                plt.title("patch_1")
                # A = self.test(test_img, test_lrHS_img)
                plt.imshow(
                    np.transpose(torchvision.utils.make_grid(patch_1.cpu(), nrow=2, padding=1, normalize=True),
                                 (1, 2, 0))[
                    :, :, :])

                plt.subplot(2, 3, 5)
                plt.axis("off")
                plt.title("patch_2")
                # A = self.test(test_img, test_lrHS_img)
                plt.imshow(
                    np.transpose(torchvision.utils.make_grid(patch_2.cpu(), nrow=2, padding=1, normalize=True),
                                 (1, 2, 0))[
                    :, :, :])

                plt.subplot(2, 3, 6)
                plt.axis("off")
                plt.title("patch_3")
                # A = self.test(test_img, test_lrHS_img)
                plt.imshow(
                    np.transpose(torchvision.utils.make_grid(patch_3.cpu(), nrow=2, padding=1, normalize=True),
                                 (1, 2, 0))[
                    :, :, :])

                sio.savemat('result/Pavia_{}.mat'.format(i), {'output': fuse_result.cpu().numpy()})
                plt.savefig('./img/' + str(i) + '.jpg')
                plt.show()
                plt.close()

                # Save model weight

                Metric = iv.spectra_metric(gtHS.permute(1, 2, 0).cpu().detach().numpy(),
                                           fuse_result[0].permute(1, 2, 0).cpu().detach().numpy(), 4)
                PSNR = Metric.PSNR()
                SAM = Metric.SAM()
                SSIM = Metric.SSIM()
                MSE = Metric.MSE()
                ERGAS = Metric.ERGAS()
                print('评价指标:')
                y = PrettyTable()
                y.add_column("Index", ['value'])
                y.add_column("PSNR", [PSNR])
                y.add_column("SAM", [SAM])
                y.add_column("SSIM", [SSIM])
                y.add_column("MSE", [MSE])
                y.add_column("ERGAS", [ERGAS])
                print(y)

    def test_(self):

        self.sr3.eval()
        # test_data = copy.deepcopy(next(iter(self.testloader)))
        # [gtHS, hrMS, lrHS, HSI_Patch, MSI_Patch2] = test_data
        for step, [gtHS, lrHS, patch_1, patch_2, patch_3] in enumerate(self.testloader):
            # if (step != 56): continue
            # gtHS = gtHS.type(torch.float32).to(DEVICE)
            # lrHS = lrHS.type(torch.float32).to(DEVICE)
            # patch_1 = patch_1.type(torch.float32).to(DEVICE)
            # patch_2 = patch_2.type(torch.float32).to(DEVICE)
            # patch_3 = patch_3.type(torch.float32).to(DEVICE)

            b, c, h, w = gtHS.shape

            randn3 = np.random.randint(0, b)
            gtHS = gtHS[randn3]
            lrHS = lrHS[randn3]
            patch_1 = patch_1[randn3]
            patch_2 = patch_2[randn3]
            patch_3 = patch_3[randn3]
            # Transform to low-resolution images
            # Save example of test images to check training
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)
            plt.axis("off")
            plt.title("gtHS")
            plt.imshow(
                np.transpose(torchvision.utils.make_grid(gtHS, nrow=2, padding=1, normalize=True).cpu(), (1, 2, 0))[:,
                :, [45, 30, 15]])

            plt.subplot(2, 3, 2)
            plt.axis("off")
            plt.title("lrHS")
            # A = self.test(test_img, test_lrHS_img)
            plt.imshow(
                np.transpose(torchvision.utils.make_grid(lrHS.cpu(), nrow=2, padding=1, normalize=True), (1, 2, 0))[:,
                :, [45, 30, 15]])

            plt.subplot(2, 3, 3)
            plt.axis("off")
            plt.title("gen_img")

            fuse_result = self.test(gtHS.unsqueeze(0), lrHS.unsqueeze(0), patch_1.unsqueeze(0), patch_2.unsqueeze(0),
                                    patch_3.unsqueeze(0))
            plt.imshow(np.transpose(torchvision.utils.make_grid(fuse_result.cpu(),
                                                                nrow=2, padding=1, normalize=True), (1, 2, 0))[:, :,
                       [45, 30, 15]])

            plt.subplot(2, 3, 4)
            plt.axis("off")
            plt.title("patch_1")
            # A = self.test(test_img, test_lrHS_img)
            plt.imshow(
                np.transpose(torchvision.utils.make_grid(patch_1.cpu(), nrow=2, padding=1, normalize=True), (1, 2, 0))[
                :, :, [2, 1, 0]])

            plt.subplot(2, 3, 5)
            plt.axis("off")
            plt.title("patch_2")
            # A = self.test(test_img, test_lrHS_img)
            plt.imshow(
                np.transpose(torchvision.utils.make_grid(patch_2.cpu(), nrow=2, padding=1, normalize=True), (1, 2, 0))[
                :, :, [2, 1, 0]])

            plt.subplot(2, 3, 6)
            plt.axis("off")
            plt.title("patch_3")
            # A = self.test(test_img, test_lrHS_img)
            plt.imshow(
                np.transpose(torchvision.utils.make_grid(patch_3.cpu(), nrow=2, padding=1, normalize=True),
                             (1, 2, 0))[
                :, :, [2, 1, 0]])

            sio.savemat('test/HSX/test_{}.mat'.format(step), {'output': fuse_result.cpu().numpy()})
            plt.savefig('./img/test_' + str(step) + '.jpg')
            plt.show()
            plt.close()

            # Save model weight

            Metric = iv.spectra_metric(gtHS.permute(1, 2, 0).cpu().detach().numpy(),
                                       fuse_result[0].permute(1, 2, 0).cpu().detach().numpy(), 4)
            PSNR = Metric.PSNR()
            SAM = Metric.SAM()
            SSIM = Metric.SSIM()
            MSE = Metric.MSE()
            ERGAS = Metric.ERGAS()
            print('评价指标:')
            y = PrettyTable()
            y.add_column("Index", ['value'])
            y.add_column("PSNR", [PSNR])
            y.add_column("SAM", [SAM])
            y.add_column("SSIM", [SSIM])
            y.add_column("MSE", [MSE])
            y.add_column("ERGAS", [ERGAS])
            print(y)

    def test(self, gtHS, lrHS, patch_1, patch_2, patch_3):
        self.sr3.eval()
        with torch.no_grad():
            if isinstance(self.sr3, nn.DataParallel):
                result_SR = self.sr3.module.super_resolution(gtHS, lrHS, patch_1, patch_2, patch_3)
            else:
                result_SR = self.sr3.super_resolution(gtHS, lrHS, patch_1, patch_2, patch_3)
        self.sr3.train()
        return result_SR

    def save(self, save_path, i):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path + 'SR3_model_epoch-{}.pt'.format(i))

    def load(self, load_path):
        network = self.sr3
        if isinstance(self.sr3, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path))
        print("Model loaded successfully")


if __name__ == "__main__":
    batch_size = 4
    LR_size = 40
    img_size = 160

    # 超参数
    EPOCH = 1000
    BATCHSIZE = 1
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    PATCH_SIZE = 16
    IN_CH_HSI = 102
    IN_CH_MSI = 4

    train_datasat = Datasat('train', 160, IN_CH_HSI=IN_CH_HSI, IN_CH_MSI=IN_CH_MSI)
    train_loader = DataLoader(train_datasat, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

    test_datasat = Datasat('test', 160, IN_CH_HSI=IN_CH_HSI, IN_CH_MSI=IN_CH_MSI)
    test_loader = DataLoader(test_datasat, batch_size=1, shuffle=False, num_workers=0)

    cuda = torch.cuda.is_available()
    schedule_opt = {'schedule': 'linear', 'n_timestep': 2000, 'linear_start': 1e-4, 'linear_end': 0.002}

    sr3 = SR3(DEVICE, img_size=img_size, LR_size=LR_size, loss_type='l1',
              dataloader=train_loader, testloader=test_loader, schedule_opt=schedule_opt,
              save_path='./models/',
              load_path='./models/SR3_model_epoch-499.pt', load=True,
              inner_channel=64,
              norm_groups=16, channel_mults=(1, 2, 2, 2), dropout=0, res_blocks=2, lr=1e-4, distributed=False)

    sr3.train(2000, 50)
    #sr3.test_()



