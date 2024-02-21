import numpy as np
from typing import Iterable
from skimage import metrics
import torch
from tqdm import tqdm
from torch import nn
from collections import defaultdict


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"sum_value": 0, "count": 0, "avg": 0})

    def update(self, metric_name, value):
        metric = self.metrics[metric_name]

        metric["sum_value"] += value
        metric["count"] += 1
        metric["avg"] = metric["sum_value"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def train_one_epoch(ratio: int, batch_size: int, epoch: int, model: nn.Module, train_iter: Iterable, optimizer: torch.optim.Optimizer,
                    loss_fun: nn.Module, device: torch.device, trainwriter: None, use_amp=False, scaler=None):
    model.train()
    loss_train, psnr_train = [], []
    train_stream = tqdm(train_iter)
    mm = MetricMonitor()
    for i, (feature, label) in enumerate(train_stream):  # (B, C, H, W)
        feature, label = feature.float().to(device, non_blocking=True), label.float().to(device, non_blocking=True)
        
        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                y_hat = model(feature)
                l = loss_fun(y_hat, label)
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            y_hat = model(feature)
            l = loss_fun(y_hat, label)
            l.backward()
            optimizer.step()

        loss_train.append(l.item())
        y_hat = np.array(y_hat.cpu().detach().numpy(), dtype=np.float32)
        label = np.array(label.cpu().numpy(), dtype=np.float32)
        psnr = metrics.peak_signal_noise_ratio(label, y_hat, data_range=1.)
        psnr_train.append(psnr)
        torch.cuda.synchronize()
        
        mm.update('loss', l.item())
        mm.update('PSNR', psnr)
        train_stream.set_description(f"Ratio: {ratio}. Epoch: {epoch:02}. Train. {mm}")

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        train_stream.set_postfix(
            lr=f'{current_lr:0.5f}',
            gpu_mem=f'{mem:0.2f} GB')

    if trainwriter is not None:
        trainwriter.add_scalar('Train Loss', np.mean(loss_train), epoch)
        trainwriter.add_scalar('Train PSNR', np.mean(psnr_train), epoch)


def val_one_epoch(ratio: int, batch_size: int, epoch: int, model: nn.Module, val_iter: Iterable, loss_fun: nn.Module, device: torch.device, evalwriter: None, use_amp=False):
    model.eval()
    mm = MetricMonitor()
    stream = tqdm(val_iter)
    with torch.no_grad():
        psnr_val, loss_val = [], []
        for i, (feature, label) in enumerate(stream):  # (B, C, H, W)
            feature, label = feature.float().to(device, non_blocking=True), label.float().to(device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    y_hat = model(feature)
                    l = loss_fun(y_hat, label)
            else:
                y_hat = model(feature)
                l = loss_fun(y_hat, label)

            y_hat = np.array(y_hat.cpu().detach().numpy(), dtype=np.float32)
            label = np.array(label.cpu().numpy(), dtype=np.float32)
            val_psnr = metrics.peak_signal_noise_ratio(label, y_hat, data_range=1.)

            loss_val.append(l.item())
            psnr_val.append(val_psnr)
            
            mm.update('loss', l.item())
            mm.update('PSNR', val_psnr)
            stream.set_description(f"Ratio: {ratio}. Epoch: {epoch:02}. Valid. {mm}")

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            stream.set_postfix(gpu_mem=f'{mem:0.2f} GB')

        if evalwriter is not None:
            evalwriter.add_scalar('Validate Loss', np.mean(loss_val), epoch)
            evalwriter.add_scalar('Validate PSNR', np.mean(psnr_val), epoch)
        return np.mean(psnr_val)

