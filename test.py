# %%
import os
import argparse
parser = argparse.ArgumentParser(description='Test Model')
parser.add_argument('--gpu', default='0', type=str, help='gpu to test')
parser.add_argument('--block_size', default=33, type=int, help='block size')
parser.add_argument('--cs_ratio', default=10, type=int, help='cs ratio')
parser.add_argument('--model', default="base", type=str, help='name of model')
parser.add_argument('--test_dataset', default="Set11", type=str, help='name of test dataset')
parser.add_argument('--is_cuda', default=True, type=bool, help='use cuda or not')
parser.add_argument('--is_save', default=True, type=bool, help='save reconstructed images or not')

args = parser.parse_known_args()[0]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from model import *
from torch.autograd import Variable
import numpy as np
from skimage import metrics
import cv2
import time
from scipy import io
import torch
from torch import nn
import utils
from timm.models import create_model
from matplotlib import pyplot as plt
import gc


def reconstruction_padding(model:nn.Module,
                           image_name:str,
                           imsize:int,
                           save_path:str,
                           device,
                           is_save:bool=True,):
    model.eval()
    with torch.no_grad():
        # test_img = np.array(io.loadmat(image_name)['data'], dtype=np.float32)
        test_img = plt.imread(image_name)
        org_img = test_img
        [row, col] = org_img.shape

        Ipad = utils.padding_img(org_img, imsize)
        inputs = Variable(torch.from_numpy(Ipad.astype('float32')).to(device))

        inputs = torch.unsqueeze(inputs, dim=0)
        inputs = torch.unsqueeze(inputs, dim=0)

        torch.cuda.synchronize()
        start_time = time.time()
        outputs = model(inputs)
        torch.cuda.synchronize()
        time_recon = time.time() - start_time

        outputs = torch.squeeze(outputs)
        outputs = outputs.cpu().data.numpy()

        recon_img = outputs[0:row, 0:col]
        recon_img[recon_img > 1.] = 1.
        recon_img[recon_img < 0.] = 0.

        org_img = np.array(org_img, dtype=np.float32)
        recon_img = np.array(recon_img, dtype=np.float32)
        ssim = metrics.structural_similarity(org_img, recon_img, data_range=1.)
        psnr = metrics.peak_signal_noise_ratio(org_img, recon_img, data_range=1.)

        res_info = "IMG: {}, PSNR/SSIM: {:.6f}/{:.6f}, time: {:.5f}\n".format(
            image_name, psnr, ssim, time_recon)
        print(res_info)

        save_name = image_name.split('/')[-1].split('\\')[-1]
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, 'results.csv'), 'a+') as f:
            store_info = "{},{},{},{}\n".format(
                image_name, psnr, ssim, time_recon)
            f.write(store_info)

        if is_save:
            recon_img_name = "{}_{:.4f}_{:.6f}.png".format(
                os.path.join(save_path, save_name[:-4]), psnr, ssim)
            cv2.imwrite(recon_img_name, recon_img*255)

        return psnr, ssim, time_recon


def main():
    print(args)
    is_cuda = args.is_cuda
    device = torch.device('cuda') if is_cuda else torch.device('cpu')
    if is_cuda: print("cuda: {}\n".format(torch.cuda.get_device_name()))
    ratio = args.cs_ratio
    is_save = args.is_save
    imsize = args.block_size
    test_dataset = args.test_dataset
    model = create_model(args.model, pretrained=True, ratio=ratio/100).to(device)

    save_path = "./results/" + str(ratio)
    image_path = "./data/test/" + test_dataset
    SSIM, PSNR, time_recon, count = 0, 0, 0, 0
    for _, _, files in os.walk(image_path):
        for file in files:
            print(file)
            image_name = os.path.join(image_path, file)
            psnr, ssim, time_t = reconstruction_padding(
                model, image_name, imsize, save_path, device, is_save)
            SSIM += ssim
            PSNR += psnr
            time_recon += time_t
            count += 1
            gc.collect()
            torch.cuda.empty_cache()

    avg = "AVERAGE,{},{},{}\n".format(
        PSNR/count, SSIM/count, time_recon/count)
    with open(os.path.join(save_path, 'results.csv'), 'a+') as f:
        f.write(avg)
    print("Average, PSNR, SSIM, time recon")
    print(avg)

if __name__ == '__main__':
    main()
