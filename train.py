# %%
import os
import argparse
parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--gpu', default='0', type=str, help='gpu to train')
parser.add_argument('--seed', default=42, type=int, help='random seed')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--image_size', default=132, type=int, help='image size')

parser.add_argument('--blr', default=1e-4, type=float, help='base learning rate')
parser.add_argument('--min_lr', default=1e-6, type=float, help='min learnign rate')

parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--warmup_epochs', default=20, type=int, help='epochs to warmup LR')
parser.add_argument('--restart_epochs', default=200, type=int, help='restart epochs')

parser.add_argument('--cs_ratio', default=10, type=int, help='cs ratio')
parser.add_argument('--model', default="base", type=str, help='name of model')

parser.add_argument('--train_folder', default="./data/train/", type=str, help='path of train data')
parser.add_argument('--val_folder', default="./data/val/", type=str, help='path of validte data')

parser.add_argument('--cuda', default=True, type=bool, help='use cuda or not')
parser.add_argument('--use_amp', default=True, type=bool, help='use AMP or not')
parser.add_argument('--pretrained', default=False, type=bool, help='is pretrained or not')

args = parser.parse_known_args()[0]
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


from model import *
import utils
import timm.optim
import timm.scheduler
from timm.models import create_model
import torch
from torch import nn
from engine import train_one_epoch, val_one_epoch
import torch.backends.cudnn as cudnn
from colorama import Fore, Back, Style
sr_ = Style.RESET_ALL


def main():
    print(args)
    seed = args.seed
    utils.setup_seed(seed)
    cudnn.benchmark = True

    is_cuda = args.cuda
    if is_cuda: print("cuda: {}\n".format(torch.cuda.get_device_name()))
    pretrain = args.pretrained
    use_amp = args.use_amp
    scaler = torch.cuda.amp.GradScaler()
    base_lr = args.blr
    min_lr = args.min_lr
    warmup_epochs = args.warmup_epochs
    num_epochs = args.epochs
    restart_epochs = args.restart_epochs
    model_name = args.model
    cs_ratio = args.cs_ratio
    batch_size = args.batch_size
    img_size = args.image_size
    train_folder = args.train_folder
    val_folder = args.val_folder

    train_iter, val_iter = utils.get_train_val_iter_folder(train_folder, val_folder, batch_size, img_size, use_augs=True)
    device = torch.device('cuda') if is_cuda else torch.device('cpu')
    model = create_model(model_name, pretrained=pretrain, ratio=cs_ratio/100).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    sch_lr = timm.scheduler.CosineLRScheduler(
        optimizer=optimizer,
        t_initial=restart_epochs,
        lr_min=min_lr,
        warmup_t=warmup_epochs,
        warmup_lr_init=base_lr/warmup_epochs,
    )
    loss_fun = nn.MSELoss().to(device)

    print(f"{Fore.GREEN}Net: {model_name}, CS ratio: {cs_ratio}, epoches: {num_epochs}, batch_size: {batch_size}, base lr: {base_lr}, image size: {img_size}{sr_}")

    best_epoch, psnr_max = 0, -1
    trainwriter, evalwriter = utils.load_logging("{}_{}".format(model_name, cs_ratio))
    save_path = os.path.join('.', 'saved_models', str(cs_ratio))
    save_name = model_name+'_{}_best.pkl'.format(cs_ratio)
    for epoch in range(num_epochs):
        sch_lr.step(epoch)
        # train
        train_one_epoch(cs_ratio, batch_size, epoch, model, train_iter, optimizer, loss_fun, device, trainwriter, use_amp, scaler)
        # val
        avg_psnr = val_one_epoch(cs_ratio, batch_size, epoch, model, val_iter, loss_fun, device, evalwriter, use_amp)
        # save
        if avg_psnr > psnr_max:
            print(f"{Fore.GREEN}Valid Score Improved ({psnr_max:0.6f} ---> {avg_psnr:0.6f})")
            psnr_max = avg_psnr
            best_epoch = epoch
            utils.save_model(model, save_name, save_path)
            print(f"Saved: {save_name}, from epoch: {best_epoch}{sr_}")
        else:
            print(f"{Fore.RED}Not Saved, Best epoch: {best_epoch}{sr_}")


if __name__ == '__main__':
    main()
