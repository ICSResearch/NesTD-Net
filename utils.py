import os
import time
import random
import torch
import numpy as np
import torchvision
import platform
import dataset
import time


def get_num_workers():
    if platform.system() == 'Windows':
        return 0
    elif platform.system() == 'Linux':
        return 16


def validateTitle(title):
    import re
    rstr = r"[\/\\\:\*\?\"\<\>\|]"  # '/ \ : * ? " < > |'
    new_title = re.sub(rstr, "_", title) 
    if len(new_title) > 125:
        new_title = new_title[:125]
    return new_title


def load_logging(mark: str):
    try:
        from tensorboardX import SummaryWriter
        if not os.path.exists("logs"):
            os.makedirs("logs")
        work_dir = validateTitle(time.strftime("%Y-%m-%dT%H:%M", time.localtime()))
        work_dir = os.path.join("logs", "{}_{}".format(mark, work_dir))
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        # Tensorboard initialization.
        trainwriter = SummaryWriter('{}/{}'.format(work_dir, 'Train'))
        evalwriter = SummaryWriter('{}/{}'.format(work_dir, 'Eval'))
        return trainwriter, evalwriter
    except:
        return None, None


def padding_img(Iorg: np.ndarray, block_size: int):
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row, block_size)
    col_pad = block_size-np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)

    return Ipad


def save_model(net, net_name, save_path):
    to_save = {
        'model': net.state_dict(),
    }
    torch.save(to_save, os.path.join(save_path, net_name))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_train_val_iter_folder(train_folder, val_folder, batch_size, img_size, use_augs=True):
    if use_augs:
        train_augs = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.RandomResizedCrop(
                (img_size, img_size), scale=(0.02, 1), ratio=(0.5, 2)),
            torchvision.transforms.RandomRotation(degrees=45),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            # torchvision.transforms.RandomPerspective(0.2, 0.2),
            torchvision.transforms.ToTensor()])
    else:
        train_augs = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()])
    train_dataset = dataset.CustomDataset(train_folder, transforms=train_augs)
    val_augs = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()])
    val_dataset = dataset.CustomDataset(val_folder, transforms=val_augs)

    train_iter = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=get_num_workers(),
                                            pin_memory=True)

    val_iter = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           num_workers=get_num_workers(),
                                           pin_memory=True)
    return train_iter, val_iter
