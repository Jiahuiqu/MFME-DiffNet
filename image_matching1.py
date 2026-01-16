from method import *
import argparse
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def _print(arg, args, **kargs):
    output_file = args.log_path
    with open(output_file, 'a+') as f:
        if kargs:
            # print(arg, end=kargs['end'])
            f.write(arg + kargs['end'])
        else:
            # print(arg)
            f.write(str(arg) + '\n')


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--path_HSI', default="/media/xd132/USER/houston/data/Huston/test/gtHS/", type=str, help='input files')
parser.add_argument('--path_match', default="/media/xd132/USER/houston/match/Houston2018_MSI.mat", type=str, help='result dir.')
parser.add_argument('--log_path', default="./log_1", type=str, help='alpha')
parser.add_argument('--save_path', default="./match/test/1/{}_1.mat", type=str, help='save_path')
parser.add_argument('--patch_size', default=80, type=int, help='patch_size')
parser.add_argument('--padding_size', default=40, type=int, help='padding_size')
parser.add_argument('--downsample_size', default=0.5, type=int, help='downsample_size')
parser.add_argument('--upsample_size', default=2, type=int, help='upsample_size')
parser.add_argument('--alpha', default=0.5, type=int, help='alpha')


if __name__ == "__main__":

    args = parser.parse_args()
    _print("******************************************************", args)
    args.log_path = "./log_1_test"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = args.path_HSI
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    sorted(all_files, key=lambda x: int(x.split('.')[0].split("/")[-1]))

    # ************************* 加载patch
    HSI_matching = loadmat(args.path_match)["hrMS"]
    HSI_matching_test = HSI_matching
    MAX = np.max(HSI_matching_test)
    MIN = np.min(HSI_matching_test)
    HSI_matching_test = (HSI_matching_test - MIN) / (MAX - MIN)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # HSI_matching_test[:,:,0] = scaler.fit_transform(HSI_matching_test[:,:,0])
    # HSI_matching_test[:, :, 1] = scaler.fit_transform(HSI_matching_test[:, :, 1])
    # HSI_matching_test[:, :, 2] = scaler.fit_transform(HSI_matching_test[:, :, 2])
    # 40->160   20 -> 80   60->240
    patch_size = args.patch_size
    padding_size = args.padding_size
    padded_image = np.pad(HSI_matching_test,
                          pad_width=((padding_size, padding_size),
                                     (padding_size, padding_size),
                                     (0, 0)),
                          mode='constant', constant_values=0)

    mean = np.mean(HSI_matching_test)
    variance = np.mean((HSI_matching_test - mean) ** 2)
    HSI_matching_pca = hyperspectral_to_1d_pca(HSI_matching_test)
    # ************************* 加载patch

    for step, path_i in enumerate(all_files):
        ## 32*32
        HSI = loadmat(path_i)["da"]
        HSI = wald_downsampling_with_zoom(np.squeeze(HSI).transpose(1,2,0), 4)
        # HSI = HSI.squeeze(0).transpose(1, 2, 0)
        HSI_pca = hyperspectral_to_1d_pca(HSI)


        # HSI_pca = scaler.fit_transform(HSI_pca)
        max_ssim = 0
        k = 0
        for row in range(padding_size, HSI_matching_test.shape[0]-padding_size-patch_size, padding_size):
            for col in range(padding_size, HSI_matching_test.shape[1]-40-patch_size, padding_size):
                # 筛选
                k+=1
                if(row+patch_size>HSI_matching_test.shape[0] or col+patch_size>HSI_matching_test.shape[1]): continue

                matching_patch = HSI_matching_pca[row:row+patch_size, col:col+patch_size]

                MAX = np.max(matching_patch)
                MIN = np.min(matching_patch)
                matching_patch = (matching_patch - MIN) / (MAX - MIN)

                # matching_patch = scaler.fit_transform(matching_patch)
                mean_patch = np.mean(matching_patch)
                variance_patch = np.mean((matching_patch - mean_patch) ** 2)
                # 方差小于大图直接跳过
                if(variance_patch<variance): continue

                # 计算方向 只能使用主成分计算
                mode_val_matching, count_matching = compute_direction(matching_patch)
                mode_val, count = compute_direction(HSI_pca)
                angle = calculate_rotation_angle(mode_val_matching, mode_val)
                # 上下左右扩大30旋转
                HSI_matching_rotation = rotate_hyperspectral_image_parallel(HSI_matching_test[row-padding_size:row+patch_size+padding_size, col-padding_size:col+patch_size+padding_size, :], angle[0])
                HSI_matching_pca_rotation = rotate_hyperspectral_image_parallel(
                    np.expand_dims(HSI_matching_pca, 2)[row - padding_size:row + patch_size + padding_size, col - padding_size:col + patch_size + padding_size, :], angle[0])

                HSI_matching_rotation = HSI_matching_rotation[padding_size:-padding_size, padding_size:-padding_size, :]
                HSI_matching_pca_rotation = HSI_matching_pca_rotation[padding_size:-padding_size, padding_size:-padding_size, :]
                # 上采下采两张图
                HSI_matching_rotation_downsample = zoom(HSI_matching_pca_rotation, (args.downsample_size, args.downsample_size, 1), order=1)
                HSI_upsample = zoom(np.expand_dims(HSI_pca, 2), (args.upsample_size, args.upsample_size, 1), order=1)
                # 计算SSIM
                # 调整亮度（根据需要修改 alpha）
                alpha = args.alpha
                image1_adjusted1 = adjust_luminance(HSI_matching_rotation_downsample, alpha)
                image2_adjusted1 = adjust_luminance(np.expand_dims(HSI_pca, 2)[:, :, :], alpha)
                image1_adjusted2 = adjust_luminance(HSI_matching_pca_rotation, alpha)
                image2_adjusted2 = adjust_luminance(HSI_upsample[:, :, :], alpha)
                out = 0.5*ssim(image1_adjusted1.squeeze(2), image2_adjusted1.squeeze(2),  full=True, data_range=1)[0]+0.5*ssim(image1_adjusted2.squeeze(2), image2_adjusted2.squeeze(2),  full=True, data_range=1)[0]

                if(out>max_ssim):
                    max_ssim=out
                    savemat(args.save_path.format(step), {"Match_1": HSI_matching_rotation, 'index': out})
                    # 高分旋转图像
                    plt.imshow(HSI_matching_rotation[:, :, :])
                    plt.axis('off')  # 关闭坐标轴
                    plt.savefig('./match_img/test/1/rotated_MSI_{}.jpg'.format(step), bbox_inches='tight', pad_inches=0)
                    # 高光谱图像
                    plt.imshow(HSI[:, :, [45, 30, 15]])
                    plt.axis('off')  # 关闭坐标轴
                    plt.savefig('./match_img/test/1/Match_{}.jpg'.format(step), bbox_inches='tight', pad_inches=0)

                _print("当前时间戳: {}".format(time.time()), args)
                _print("第{}张图 || 第{}patch || 相似度: {} ".format(step, k, out), args)

    print("ok")

