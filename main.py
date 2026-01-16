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




