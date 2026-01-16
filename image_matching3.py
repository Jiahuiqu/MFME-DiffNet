import numpy

from method import *
import argparse
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import heapq
import time
import pickle
import json
from skimage.metrics import structural_similarity as ssim2

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
parser.add_argument('--path_HSI', default="/media/xidian/4b9949b2-d832-4f6e-9211-60d8e6480133/HJ/毕设3/data/PaviaC-PaviaU/test/LRHS/", type=str, help='input files')
parser.add_argument('--path_match', default="/media/xidian/4b9949b2-d832-4f6e-9211-60d8e6480133/HJ/毕设3/data/PaviaC-PaviaU/unpaired/unpair_pavia.mat", type=str, help='match file.')
parser.add_argument('--log_path', default="./log_1", type=str, help='alpha')
parser.add_argument('--save_path', default="./match/train/3/{}_3.mat", type=str, help='save_path')
parser.add_argument('--patch_size', default=160, type=int, help='patch_size')
parser.add_argument('--padding_size', default=0, type=int, help='padding_size')
parser.add_argument('--downsample_size', default=0.5, type=int, help='downsample_size')
parser.add_argument('--upsample_size', default=2, type=int, help='upsample_size')
parser.add_argument('--alpha', default=0.5, type=int, help='alpha')

capacity = 10
data_list = [{"data": 0, "image": np.random.rand(80, 80, 4)} for _ in range(capacity)]

# 将列表转换为堆（按 data 排序）
# heapq.heapify(data_list)
# 添加新数据的函数

def add_element(new_data, new_image):
    # 获取堆顶元素（data 最小的元素）
    min_element = data_list[0]

    # 如果新数据的 data 大于最小 data，则替换
    if new_data > min_element["data"]:
        heapq.heappop(data_list)  # 移除堆顶元素
        heapq.heappush(data_list, {"data": new_data, "image": new_image})  # 添加新元素
        print(f"替换了 data 最小的元素（data: {new_data}）")
    else:
        print("新数据的 data 不大于最小 data，未进行替换")


if __name__ == "__main__":


    args = parser.parse_args()
    _print("******************************************************", args)
    args.log_path = "./log_1_test"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    ## 读取高光谱图像
    path = args.path_HSI
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    all_files = sorted(all_files, key=lambda x: int(x.split('.')[0].split("/")[-1]))

    # ************************* 加载匹配图像patch****************************
    HSI_matching = loadmat(args.path_match)["hrMS"]

    # # 归一化
    # HSI_matching_test = HSI_matching
    # MAX = np.max(HSI_matching_test)
    # MIN = np.min(HSI_matching_test)
    # HSI_matching_test = (HSI_matching_test - MIN) / (MAX - MIN)

    # 40->160   20 -> 80   80->320
    # 图像均值方差
    mean = np.mean(HSI_matching)
    variance = np.mean((HSI_matching - mean) ** 2)
    HSI_matching_pca = hyperspectral_pca(HSI_matching.transpose(1, 2, 0))

    MAX = np.max(HSI_matching_pca)
    MIN = np.min(HSI_matching_pca)
    HSI_matching_pca = (HSI_matching_pca - MIN) / (MAX - MIN)
    # ************************* 加载patch ***************************
    patch_size = 320
    for step, path_i in enumerate(all_files):
        ## 32*32
        # 加载高光谱图像
        HSI = loadmat(path_i)["lrHS"]
        HSI_pca = hyperspectral_pca(HSI.transpose(1, 2, 0))
        HSI_pca_upsample = cv2.resize(HSI_pca, (HSI_pca.shape[0] * 4, HSI_pca.shape[1] * 4), interpolation=cv2.INTER_LINEAR)
        data_list=[]
        stack = MonotonicStack(10)
        stack.push(0)
        max_ssim = 0
        k = 0
        for row in range(0, HSI_matching_pca.shape[0]-patch_size, 20):
            for col in range(0, HSI_matching_pca.shape[1]-patch_size+1, 40):
                # 筛选
                k+=1
                if (row + patch_size > HSI_matching_pca.shape[0] or col + patch_size > HSI_matching_pca.shape[
                    1]): continue
                matching_patch = HSI_matching_pca[row:row+patch_size, col:col+patch_size]

                # ****************************裁块图像归一化********************************
                MAX = np.max(matching_patch)
                MIN = np.min(matching_patch)
                matching_patch = (matching_patch - MIN) / (MAX - MIN)

                # 均值-方差 筛选patch
                mean_patch = np.mean(matching_patch)
                variance_patch = np.mean((matching_patch - mean_patch) ** 2)
                # 方差小于大图直接跳过
                # if(variance_patch<variance): continue
                # 计算方向 只能使用主成分计算并旋转图像
                mode_val_matching, count_matching = compute_direction(matching_patch)
                mode_val, count = compute_direction(HSI_pca_upsample)
                angle = calculate_rotation_angle(mode_val_matching, mode_val)
                # 上下左右扩大30旋转
                HSI_matching_ro = rotate_image_parallel(matching_patch, angle)

                # 上采下采两张图
                # 匹配图像
                matching_patch_up_2 = cv2.resize(HSI_matching_ro, (int(matching_patch.shape[0] * 0.5), int(matching_patch.shape[1] * 0.5)), interpolation=cv2.INTER_LINEAR)
                # 高光谱图像
                HSI_pca_upsample_2 = cv2.resize(HSI_pca, (HSI_pca.shape[0] * 8, HSI_pca.shape[1] * 8), interpolation=cv2.INTER_LINEAR)

                # 计算SSIM
                # 调整亮度（根据需要修改 alpha）
                SSIM1 = ssim(matching_patch*5, HSI_pca_upsample_2*5)
                # SSIM1_2= ssim2(matching_patch*255, HSI_pca_upsample_2*255, range=255)
                # plt.subplot(1, 2, 1)
                # plt.imshow(matching_patch)
                # plt.subplot(1, 2, 2)
                # plt.imshow(HSI_pca_upsample_2)
                # plt.savefig('./rotated_MSI.jpg'.format(step), bbox_inches='tight', pad_inches=0)

                SSIM2 = ssim(matching_patch_up_2*5, HSI_pca_upsample*5)
                out = 0.5*SSIM1+0.5*SSIM2

                # 计算旋转图像
                image_ro = rotate_hyperspectral_image_parallel(HSI_matching.transpose(1, 2, 0)[row:row+patch_size, col:col+patch_size,:], angle)
                d = []
                d.append({'data':out, 'image':image_ro})
                if(out>stack.top()):
                    stack.push(out)
                    if(len(data_list)>=10):
                        data_list[9]=d
                    else:
                        data_list.append(d)

                    data_list.sort(key=lambda x: x[0]["data"])

                    # savemat(args.save_path.format(step), {"Match_1": HSI_matching_rotation, 'index': out})
                    # # 高分旋转图像
                    # plt.imshow(HSI_matching_rotation[:, :, :])
                    # plt.axis('off')  # 关闭坐标轴
                    # plt.savefig('./match_img/test/1/rotated_MSI_{}.jpg'.format(step), bbox_inches='tight', pad_inches=0)
                    # # 高光谱图像
                    # plt.imshow(HSI[:, :, [45, 30, 15]])
                    # plt.axis('off')  # 关闭坐标轴
                    # plt.savefig('./match_img/test/1/Match_{}.jpg'.format(step), bbox_inches='tight', pad_inches=0)
                # 保存到文件

                _print("当前时间戳: {}".format(time.time()), args)
                _print("第{}张图 || 第{}patch || 相似度: {} ".format(step, k, out), args)

        match_img = numpy.zeros([10,patch_size,patch_size,4])
        match_score = numpy.zeros([10])
        for i, data in enumerate(data_list):
            match_img[i,:,:,:]=data[0]['image']
            match_score[i]=data[0]['data']
        savemat('./match/test/3/'+str(step)+".mat", {'match_img': match_img, 'match_score': match_score})

        # 高分旋转图像
        # plt.figu
        # plt.imshow(image_ro[:, :, [0,1,2]])
        # plt.axis('off')  # 关闭坐标轴
        # plt.savefig('./match_img/test/1/rotated_MSI_{}.jpg'.format(step), bbox_inches='tight', pad_inches=0)
        # # 高光谱图像
        # plt.imshow(HSI[:, :, [15, 30, 45]])
        # plt.axis('off')  # 关闭坐标轴
        # plt.savefig('./match_img/test/1/Match_{}.jpg'.format(step), bbox_inches='tight', pad_inches=0)

    print("ok")
