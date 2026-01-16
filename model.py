import torch
from torch import nn
import torch.nn.functional as F
import einops
import math


def restore_from_split(input_tensor):
    # input_tensor: shape (16 * b, c, h/4, w/4)
    b, c, h, w = input_tensor.shape[0]//16, input_tensor.shape[1], input_tensor.shape[2]*4, input_tensor.shape[3]*4
    # 确保 h 和 w 可以被 4 整除
    assert h % 4 == 0 and w % 4 == 0, "Height and width should be divisible by 4"

    # 将输入的 tensor 拆成 16 个块，并按原顺序重新组合
    top_left = input_tensor[:b, :, :, :]
    top_middle_left = input_tensor[b:2 * b, :, :, :]
    top_middle_right = input_tensor[2 * b:3 * b, :, :, :]
    top_right = input_tensor[3 * b:4 * b, :, :, :]

    middle_left = input_tensor[4 * b:5 * b, :, :, :]
    middle_middle_left = input_tensor[5 * b:6 * b, :, :, :]
    middle_middle_right = input_tensor[6 * b:7 * b, :, :, :]
    middle_right = input_tensor[7 * b:8 * b, :, :, :]

    bottom_left = input_tensor[8 * b:9 * b, :, :, :]
    bottom_middle_left = input_tensor[9 * b:10 * b, :, :, :]
    bottom_middle_right = input_tensor[10 * b:11 * b, :, :, :]
    bottom_right = input_tensor[11 * b:12 * b, :, :, :]

    bottom_left_last = input_tensor[12 * b:13 * b, :, :, :]
    bottom_middle_left_last = input_tensor[13 * b:14 * b, :, :, :]
    bottom_middle_right_last = input_tensor[14 * b:15 * b, :, :, :]
    bottom_right_last = input_tensor[15 * b:16 * b, :, :, :]

    # 重新拼接回原来的尺寸
    top = torch.cat([top_left, top_middle_left, top_middle_right, top_right], dim=2)
    middle = torch.cat([middle_left, middle_middle_left, middle_middle_right, middle_right], dim=2)
    bottom = torch.cat([bottom_left, bottom_middle_left, bottom_middle_right, bottom_right], dim=2)
    bottom_last = torch.cat([bottom_left_last, bottom_middle_left_last, bottom_middle_right_last, bottom_right_last],
                            dim=2)

    # 合并上下三个部分
    output_tensor = torch.cat([top, middle, bottom, bottom_last], dim=3)

    return output_tensor

def split_and_concat(input_tensor):
    # input_tensor: shape (b, c, h, w)
    b, c, h, w = input_tensor.shape

    # 确保 h 和 w 可以被 4 整除
    assert h % 4 == 0 and w % 4 == 0, "Height and width should be divisible by 4"

    # 划分为16个块，每个块的形状为 (h/4, w/4)
    top_left = input_tensor[:, :, :h // 4, :w // 4]
    top_middle_left = input_tensor[:, :, :h // 4, w // 4:2 * w // 4]
    top_middle_right = input_tensor[:, :, :h // 4, 2 * w // 4:3 * w // 4]
    top_right = input_tensor[:, :, :h // 4, 3 * w // 4:]

    middle_left = input_tensor[:, :, h // 4:2 * h // 4, :w // 4]
    middle_middle_left = input_tensor[:, :, h // 4:2 * h // 4, w // 4:2 * w // 4]
    middle_middle_right = input_tensor[:, :, h // 4:2 * h // 4, 2 * w // 4:3 * w // 4]
    middle_right = input_tensor[:, :, h // 4:2 * h // 4, 3 * w // 4:]

    bottom_left = input_tensor[:, :, 2 * h // 4:3 * h // 4, :w // 4]
    bottom_middle_left = input_tensor[:, :, 2 * h // 4:3 * h // 4, w // 4:2 * w // 4]
    bottom_middle_right = input_tensor[:, :, 2 * h // 4:3 * h // 4, 2 * w // 4:3 * w // 4]
    bottom_right = input_tensor[:, :, 2 * h // 4:3 * h // 4, 3 * w // 4:]

    bottom_left_last = input_tensor[:, :, 3 * h // 4:, :w // 4]
    bottom_middle_left_last = input_tensor[:, :, 3 * h // 4:, w // 4:2 * w // 4]
    bottom_middle_right_last = input_tensor[:, :, 3 * h // 4:, 2 * w // 4:3 * w // 4]
    bottom_right_last = input_tensor[:, :, 3 * h // 4:, 3 * w // 4:]

    # 合并到 batch 维度
    output_tensor = torch.cat([
        top_left, top_middle_left, top_middle_right, top_right,
        middle_left, middle_middle_left, middle_middle_right, middle_right,
        bottom_left, bottom_middle_left, bottom_middle_right, bottom_right,
        bottom_left_last, bottom_middle_left_last, bottom_middle_right_last, bottom_right_last
    ], dim=0)

    return output_tensor


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., patch_size=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.localization_linear = nn.Sequential(
            nn.Linear(in_features=dim*patch_size*patch_size, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2 * 3)
        )

    def forward(self, x, y):

        B, L, C = x.shape
        qkv_1 = self.qkv_1(x)
        qkv_2 = self.qkv_2(y)

        qkv_1 = einops.rearrange(qkv_1, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
        q_1, k_1, v_1 = qkv_1[0], qkv_1[1], qkv_1[2]  # B H L D
        qkv_2 = einops.rearrange(qkv_2, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
        q_2, k_2, v_2 = qkv_2[0], qkv_2[1], qkv_2[2]  # B H L D
        attn_1 = (q_1 @ k_2.transpose(-2, -1)) * self.scale
        attn_1 = (attn_1).softmax(dim=-1)
        attn_1 = self.attn_drop(attn_1)
        sim_features = (attn_1 @ v_2).transpose(1, 2).reshape(B, L, C)
        sim_features = sim_features.permute(0, 2, 1)

        return sim_features


class Attention_complete(nn.Module):

    def __init__(self, patch_size=40, dim=256, num_heads=4, qkv_bias=False, qk_scale=None, in_ch_msi=4, in_ch_hsi = 256):
        super(Attention_complete, self).__init__()
        patch_size = patch_size//4
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_size ** 2, dim))
        self.Embedding_HSI = nn.Conv2d(in_ch_hsi, 256, 3, 1, 1)
        self.Embedding_MSI = nn.Conv2d(in_ch_msi, 256, 3, 1, 1)
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, patch_size=patch_size)

    def forward(self, x, y):
        x_org = x
        x = self.Embedding_HSI(x).flatten(2).transpose(1, 2)
        y = self.Embedding_MSI(y).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        y = y + self.pos_embed
        sim_feature = self.attn(self.norm(x), self.norm(y))
        sim_feature = sim_feature.reshape(x_org.shape[0], x_org.shape[1], x_org.shape[2], x_org.shape[3])

        return sim_feature


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock, self).__init__()

        # 使用 Conv2d 创建卷积层
        self.conv = nn.Conv2d(
            in_channels,  # 输入通道数
            out_channels,  # 输出通道数
            kernel_size=4,  # 卷积核大小
            stride=2,  # 步长为 2，实现尺寸减小为原来的一半
            padding=1  # 填充，确保输出的尺寸是原来的一半
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, in_ch):
        super(ConvNet, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, size=(320, 320), mode='bicubic', align_corners=False)
        return x


class DeconvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvolutionBlock, self).__init__()

        # 使用 ConvTranspose2d 创建反卷积层
        self.deconv = nn.ConvTranspose2d(
            in_channels,  # 输入通道数
            out_channels,  # 输出通道数
            kernel_size=4,  # 卷积核大小
            stride=2,  # 步长为 2，实现尺寸放大为原来两倍
            padding=1,  # 填充，确保输出的尺寸是原来的两倍
            output_padding=0  # 输出填充，确保尺寸精确对齐
        )

    def forward(self, x):
        return self.deconv(x)


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual,
                      relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1,
                      dilation=2 * visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


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

class Block_norm(nn.Module):
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


class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=True):
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


class ResBlock_norm(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0,
                 num_heads=1, use_affine_level=False, norm_groups=32, att=True):
        super().__init__()
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, dim_out, use_affine_level)
        self.block1  = nn.Sequential(
            nn.Conv2d(dim, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim_out)

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim_out)

        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        y = self.block1(x)
        y = self.noise_func(y, time_emb)
        y = self.block2(y)
        x = y + self.res_conv(x)
        return x

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


class Fusemodel(nn.Module):

    def __init__(self, in_c, ):

        super(Fusemodel, self).__init__()
        noise_level_channel = 128

        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(128),
            nn.Linear(128, 128 * 4),
            Swish(),
            nn.Linear(128 * 4, 128)
        )

        self.conv = nn.Conv2d(in_c*2, 256, 3, 1, 1)
        self.ResBlock1_1 = ResBlock(dim=256, dim_out=256, noise_level_emb_dim=noise_level_channel)
        self.ResBlock1_2 = ResBlock(dim=256, dim_out=256, noise_level_emb_dim=noise_level_channel)
        self.ResBlock1_3 = ResBlock(dim=256, dim_out=256, noise_level_emb_dim=noise_level_channel)

        self.Deconv_block = DeconvolutionBlock(in_channels=4, out_channels=4)
        self.conv_block = ConvolutionBlock(in_channels = 4, out_channels=4)
        self.BasicRFB_scale_1 = BasicRFB(4, 64)
        self.BasicRFB_scale_2 = BasicRFB(4, 64)
        self.BasicRFB_scale_3 = BasicRFB(4, 64)

        self.att1 = Attention_complete(patch_size=160, dim=256, in_ch_msi=64, in_ch_hsi=256)
        self.att2 = Attention_complete(patch_size=160, dim=256, in_ch_msi=64, in_ch_hsi=256)
        self.att3 = Attention_complete(patch_size=160, dim=256, in_ch_msi=64, in_ch_hsi=256)

        self.ResBlock2 = ResBlock(dim=256+256, dim_out=256, noise_level_emb_dim=noise_level_channel)
        self.ResBlock3 = ResBlock(dim=256+256, dim_out=256, noise_level_emb_dim=noise_level_channel)
        self.ResBlock4 = ResBlock(dim=256+256, dim_out=256, noise_level_emb_dim=noise_level_channel)

        self.lastconv = nn.Conv2d(256, in_c, 3, 1, 1)

    def forward(self, lrHS, patch_1, patch_2, patch_3, noise_level, sim_1, sim_2, sim_3):
        t = self.noise_level_mlp(noise_level)

        # 高光谱分支
        lrHS_feature = self.conv(lrHS)
        lrHS_feature = self.ResBlock1_1(lrHS_feature, t)
        lrHS_feature = self.ResBlock1_2(lrHS_feature, t)
        lrHS_feature = self.ResBlock1_3(lrHS_feature, t)

        # 多光谱分支
        patch_featurn_1 = self.Deconv_block(patch_1)
        patch_featurn_1 = self.BasicRFB_scale_1(patch_featurn_1)

        patch_featurn_2 = self.BasicRFB_scale_2(patch_2)

        patch_featurn_3 = self.conv_block(patch_3)
        patch_featurn_3 = self.BasicRFB_scale_3(patch_featurn_3)
        lrHS_feature_org = lrHS_feature
        lrHS_feature = split_and_concat(lrHS_feature)
        patch_featurn_1 = split_and_concat(patch_featurn_1)
        patch_featurn_2 = split_and_concat(patch_featurn_2)
        patch_featurn_3 = split_and_concat(patch_featurn_3)

        patch_featurn_1 = self.att1(lrHS_feature, patch_featurn_1)
        patch_featurn_2 = self.att2(lrHS_feature, patch_featurn_2)
        patch_featurn_3 = self.att3(lrHS_feature, patch_featurn_3)

        patch_featurn_1 = restore_from_split(patch_featurn_1)
        patch_featurn_2 = restore_from_split(patch_featurn_2)
        patch_featurn_3 = restore_from_split(patch_featurn_3)

        lrHS_feature = torch.cat([lrHS_feature_org, sim_1*patch_featurn_1], dim=1)
        lrHS_feature = self.ResBlock2(lrHS_feature, t)

        lrHS_feature = torch.cat([lrHS_feature, sim_2*patch_featurn_2], dim=1)
        lrHS_feature = self.ResBlock3(lrHS_feature, t)

        lrHS_feature = torch.cat([lrHS_feature, sim_3*patch_featurn_3], dim=1)
        lrHS_feature = self.ResBlock4(lrHS_feature, t)
        HSI = self.lastconv(lrHS_feature)

        return HSI







