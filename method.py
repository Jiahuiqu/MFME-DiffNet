import torch
from scipy.io import savemat,loadmat
import math
import numpy as np
from sklearn.decomposition import PCA
import scipy.stats as stats
import cv2
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.ndimage import zoom
import torch.nn.functional as F
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
import numpy as np
matplotlib.use('Agg')


directions = {
    (0, -1): 1,  # 上
    (-1, 0): 2,   # 左
    (1, 0): 3,    # 右
    (0, 1): 4,    # 下
}

def rotate_band(band, angle, height, width):
    """
    旋转单个波段（2D图像）。

    参数:
    band (np.ndarray): 单个波段的2D图像，形状为 (height, width)
    angle (float): 旋转角度
    height (int): 原始图像高度
    width (int): 原始图像宽度

    返回:
    np.ndarray: 旋转后的2D图像
    """
    return rotate(band, angle, reshape=False)

def rotate_image_parallel(image, angle):
    """
    使用并行计算旋转2D图像。

    参数:
    image (np.ndarray): 2D图像，形状为 (height, width)
    angle (float): 旋转角度

    返回:
    np.ndarray: 旋转后的2D图像
    """
    height, width = image.shape

    # 直接旋转单波段图像
    rotated_image = rotate_band(image, angle, height, width)

    return rotated_image

def rotate_image(image, angle):
    """
    旋转图像到指定角度。

    参数:
    image (np.ndarray): 输入图像
    angle (float): 旋转角度

    返回:
    np.ndarray: 旋转后的图像
    """
    # 获取图像的中心点
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 旋转图像
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    return rotated_image


def calculate_rotation_angle(mode1, mode2):
    """
    通过两幅图像的众数差计算旋转角度，考虑最短旋转角度（圆形差值）。

    参数:
    mode1 (int): 第一幅图像的众数 (0-7)
    mode2 (int): 第二幅图像的众数 (0-7)

    返回:
    float: 旋转角度 (正值为左旋，负值为右旋)
    """
    # 计算原始众数差
    diff = mode1 - mode2

    # 处理差值为最短旋转角度
    if diff >= 2:
        diff -= 4  # 顺时针旋转
    elif diff <= -2:
        diff += 4  # 逆时针旋转

    # 计算旋转角度
    rotation_angle = diff * 90  # 每个众数对应45度

    return rotation_angle


def hyperspectral_pca(hyperspectral_image):
    """
    使用 PCA 将高光谱图像降到一个通道。

    参数:
    hyperspectral_image (np.ndarray): 高光谱图像，形状为 (height, width, num_channels)

    返回:
    np.ndarray: 降维后的1D图像，形状为 (height, width)
    """
    # 获取高光谱图像的形状
    height, width, num_channels = hyperspectral_image.shape

    # 将图像重塑为 (height * width, num_channels) 的二维矩阵
    reshaped_image = hyperspectral_image.reshape((-1, num_channels))

    # 使用 PCA 将高光谱图像降维到1个通道
    pca = PCA(n_components=1)
    reduced_image = pca.fit_transform(reshaped_image)

    # 将降维后的图像恢复为原始图像的大小
    reduced_image = reduced_image.reshape((height, width))

    return reduced_image


def compute_direction(img):
    """
    计算每个像素的梯度方向。

    参数:
    img (np.ndarray): 输入的灰度图像 (2D numpy array)

    返回:
    np.ndarray: 每个像素的梯度方向 (2D numpy array)
    """
    # 确保输入是灰度图像
    if len(img.shape) != 2:
        raise ValueError("输入必须是灰度图像")

    padded_img = np.pad(img, pad_width=1, mode='constant', constant_values=0)

    # 获取图像的高度和宽度
    h, w = img.shape

    # 获取中心像素周围8个邻域的差值

    diff_up = padded_img[1:-1, :-2] - padded_img[1:-1, 1:-1]
    diff_left = padded_img[:-2, 1:-1] - padded_img[1:-1, 1:-1]
    diff_right = padded_img[2:, 1:-1] - padded_img[1:-1, 1:-1]
    diff_down = padded_img[1:-1, 2:] - padded_img[1:-1, 1:-1]

    # 将所有差值合并成一个数组
    diffs = np.array([diff_up, diff_left, diff_right, diff_down])
    abs_diffs = np.abs(diffs)

    # 找到最大绝对差值的位置
    max_diff_indices = np.argmax(abs_diffs, axis=0)  # 找到每个像素最大差值的方向

    # 获取最大差值的符号
    max_diff_values = diffs[max_diff_indices, np.arange(h), np.arange(w)]

    # 计算最终方向并考虑符号
    direction_map = np.copy(max_diff_indices)

    # 如果最大值为负，方向反向
    direction_map[max_diff_values < 0] = (direction_map[max_diff_values < 0] + 4) % 4
    img_flat = direction_map.reshape(direction_map.shape[-1]*direction_map.shape[-1])
    mode_val, count = stats.mode(img_flat)
    return mode_val, count


def feature_matching(img1, img2):
    # 初始化 ORB 特征检测器
    orb = cv2.ORB_create()

    # 检测关键点和描述符
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 使用暴力匹配器（Brute Force Matcher）进行匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 计算匹配的数量
    match_count = len(matches)
    return match_count


def normalized_cross_correlation(img1, img2):
    # 计算归一化交叉相关
    ncc = np.sum((img1 - img1.mean()) * (img2 - img2.mean())) / np.sqrt(np.sum((img1 - img1.mean())**2) * np.sum((img2 - img2.mean())**2))
    return ncc


def adjust_luminance(image, alpha):
    # 调整亮度（通过缩放像素值）
    return np.clip(image * alpha, 0, 255)


def wald_downsampling_with_zoom(hyper_image, factor):
    """
    瓦尔德协议：先进行高斯模糊，再进行空间下采样（使用 zoom）
    :param hyper_image: 高光谱图像 (height x width x bands)，即3D NumPy 数组
    :param factor: 下采样因子
    :return: 下采样后的图像
    """
    # 将图像转换为 Torch tensor
    hyper_image_tensor = torch.tensor(hyper_image).float()

    # 添加 batch 和 channel 维度，形状变为 (batch, channels, height, width)
    hyper_image_tensor = hyper_image_tensor.permute(2, 0, 1).unsqueeze(0)

    # 使用 torch.nn.functional.interpolate 进行空间下采样
    # 这里使用双线性插值（bilinear）进行缩放
    downsampled_image = F.interpolate(hyper_image_tensor, scale_factor=1 / factor, mode='bicubic', align_corners=False)

    # 转换回原来的形状，去除 batch 维度
    downsampled_image = downsampled_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

    return downsampled_image


def calculate_ssim_components(image1, image2, data_range=255, K1=0.01, K2=0.03):
    """
    计算两个2D图像的SSIM值，并返回亮度、对比度、结构三部分的值。

    参数:
    image1 (np.ndarray): 第一个2D图像，形状为 (height, width)
    image2 (np.ndarray): 第二个2D图像，形状为 (height, width)
    data_range (float): 图像的动态范围（最大像素值 - 最小像素值）
    K1 (float): 常数 K1，默认值为 0.01
    K2 (float): 常数 K2，默认值为 0.03

    返回:
    dict: 包含亮度、对比度、结构和SSIM值的字典
    """
    # 确保输入图像是2D的
    if image1.ndim != 2 or image2.ndim != 2:
        raise ValueError("输入图像必须是2D的")

    # 确保两个图像的形状相同
    if image1.shape != image2.shape:
        raise ValueError("两个图像的形状必须相同")

    # 常数
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    C3 = C2 / 2

    # 计算均值
    mu_x = np.mean(image1)
    mu_y = np.mean(image2)

    # 计算方差和协方差
    sigma_x = np.std(image1)
    sigma_y = np.std(image2)
    sigma_xy = np.cov(image1.flatten(), image2.flatten())[0, 1]

    # 亮度比较
    luminance = (2 * mu_x * mu_y + C1) / (mu_x ** 2 + mu_y ** 2 + C1)

    # 对比度比较
    contrast = (2 * sigma_x * sigma_y + C2) / (sigma_x ** 2 + sigma_y ** 2 + C2)

    # 结构比较
    structure = (sigma_xy + C3) / (sigma_x * sigma_y + C3)

    # 计算SSIM
    ssim_value = luminance**0.1 * contrast * structure

    return ssim_value


# 单调栈
class MonotonicStack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.stack = []

    def push(self, value):
        # 如果栈未满，直接压入元素
        if len(self.stack) < self.capacity:
            self.stack.append(value)
            # 保持栈的单调递增性（栈顶是最小元素）
            self.stack.sort(reverse=True)  # 从大到小排序，栈顶是最小元素
        else:
            # 如果栈已满，且新元素大于栈顶元素，则替换栈顶元素
            if value > self.stack[-1]:
                self.stack.pop()  # 移除栈顶最小元素
                self.stack.append(value)
                # 重新排序以保持单调性
                self.stack.sort(reverse=True)

    def top(self):
        # 返回栈顶元素（最小元素）
        if self.stack:
            return self.stack[-1]
        else:
            return None

    def __str__(self):
        return str(self.stack)


def rotate_hyperspectral_image_parallel(hsi, angle):
    """
    使用并行计算旋转高光谱图像中的每个波段。

    参数:
    hsi (np.ndarray): 高光谱图像，形状为 (height, width, bands)
    angle (float): 旋转角度

    返回:
    np.ndarray: 旋转后的高光谱图像
    """
    height, width, bands = hsi.shape

    # 创建一个线程池来并行处理每个波段
    with ThreadPoolExecutor() as executor:
        # 使用executor.map并行处理每个波段
        rotated_bands = list(executor.map(lambda i: rotate_band(hsi[:, :, i], angle, height, width), range(bands)))

    # 将结果重新堆叠成一个 3D 数组
    rotated_hsi = np.stack(rotated_bands, axis=-1)

    return rotated_hsi


def gaussian(window_size, sigma):
    gauss = np.array([np.exp(-(x - window_size//2)**2 / (2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, sigma):
    _1D_window = gaussian(window_size, sigma).reshape(window_size, 1)
    _2D_window = np.outer(_1D_window, _1D_window)
    return _2D_window


def ssim(img1, img2, size_average=True):
    # 计算局部均值
    mu1 = gaussian_filter(img1, sigma=1.5, mode='constant', cval=0)
    mu2 = gaussian_filter(img2, sigma=1.5, mode='constant', cval=0)

    # 计算局部方差和协方差
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1**2, sigma=1.5, mode='constant', cval=0) - mu1_sq
    sigma2_sq = gaussian_filter(img2**2, sigma=1.5, mode='constant', cval=0) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=1.5, mode='constant', cval=0) - mu1_mu2

    # SSIM 计算
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)