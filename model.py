import torch
from torch import nn
from torch import linalg as LA
import os
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7,
                                padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Deblock(nn.Module):
    def __init__(self, dim1, dim2, depth=3, drop_path_rate=0., layer_scale_init_value=1e-6):
        super(Deblock, self).__init__()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.conv = nn.Conv2d(dim1, dim2, kernel_size=1, padding=0)
        self.stages = nn.Sequential(
            nn.Conv2d(dim2*2, dim2, kernel_size=1, padding=0),
            LayerNorm(dim2, eps=1e-6, data_format="channels_first"),
            *[Block(dim=dim2, drop_path=dp_rates[j],
                    layer_scale_init_value=layer_scale_init_value) for j in range(depth)],
        )

    def forward(self, x, x_iter):
        x_t = self.conv(x)
        x_t = torch.cat([x_t, x_iter], dim=1)
        x_t = self.stages(x_t)

        return x_t


class BasicBlock(nn.Module):
    def __init__(self, A, At, dim, k, delta, imsize, depth=1, drop_path_rate=0.):
        super(BasicBlock, self).__init__()
        self.A = A
        self.At = At
        self.k = k
        self.delta = delta
        self.imsize = imsize

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.y_seq = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            *[Block(dim=dim, drop_path=dp_rates[j]) for j in range(depth)],
        )

        self.z_seq = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, padding=0),
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            *[Block(dim=dim, drop_path=dp_rates[j]) for j in range(depth)],
        )

        self.x_seq = nn.Sequential(
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim, 1, kernel_size=1, padding=0)
        )

    def forward(self, b, mu, xplug, wk):
        xk = xplug
        Lmu = 1/mu
        Atb = torch.matmul(self.At, b)
        AtAAtb = Atb
        df = torch.true_divide(xk, torch.maximum(mu, torch.abs(xk)))

        # --- updating yk ---
        cp = xk - 1/Lmu * df
        Acp = torch.matmul(self.A, cp)
        AtAcp = torch.matmul(self.At, Acp)
        if self.delta > 0:
            lam = F.relu(Lmu * (LA.norm(b-Acp, 2)/self.delta - 1))
            gamma = lam/(lam+Lmu)
            yk = lam/Lmu*(1-gamma)*Atb + cp - gamma*AtAcp # gamma*(Atb-AtAcp) + cp
        else:  # delta == 0, gamma == 1
            yk = AtAAtb + cp - AtAcp

        # --- updating zk ---
        apk = 0.5 * (self.k+1)
        wk = apk * df + wk
        x0 = Atb
        cp = x0 - 1/Lmu*wk
        Acp = torch.matmul(self.A, cp)
        AtAcp = torch.matmul(self.At, Acp)
        if self.delta > 0:
            lam = F.relu(Lmu * (LA.norm(b-Acp, 2)/self.delta - 1))
            gamma = lam/(lam+Lmu)
            zk = lam/Lmu*(1-gamma)*Atb + cp - gamma*AtAcp
        else:  # delta == 0, gamma == 1
            zk = AtAAtb + cp - AtAcp

        # --- updating xk ---
        yk = torch.unsqueeze(torch.reshape(torch.transpose(
            yk, 0, 1), [-1, self.imsize, self.imsize]), dim=1)
        yk = self.y_seq(yk) + yk

        zk = torch.unsqueeze(torch.reshape(torch.transpose(
            zk, 0, 1), [-1, self.imsize, self.imsize]), dim=1)
        zk = self.z_seq(zk) + zk

        tauk = 2/(self.k+3)
        xk_t = tauk * zk + (1-tauk) * yk
        xk = self.x_seq(xk_t)

        return xk, wk, xk_t


class NesTDNet(torch.nn.Module):
    def __init__(self, ratio, imsize=33, dephts=[1, 1, 3, 1], dims=[32, 32, 32, 32], drop_path_rate=0.):
        super(NesTDNet, self).__init__()
        self.mu = 2e-5
        self.delta = 0
        self.imsize = imsize

        self.basicLayer = []
        self.deblockLayer = []
        self.conv = []

        A = self.load_sampling_matrix(ratio)
        self.A = nn.Parameter(torch.from_numpy(A).float(), requires_grad=True)
        self.At = nn.Parameter(torch.from_numpy(
            np.transpose(A)).float(), requires_grad=True)

        count = 0
        self.n_layer = sum(dephts)
        for i in range(len(dephts)):
            for j in range(dephts[i]):
                dim1 = dim2 if count != 0 else 1
                dim2 = dims[i]
                count += 1
                self.basicLayer.append(BasicBlock(
                    self.A, self.At, dim2, count, self.delta, self.imsize, drop_path_rate=drop_path_rate))
                self.deblockLayer.append(
                    Deblock(dim1, dim2, drop_path_rate=drop_path_rate))
                self.conv.append(nn.Conv2d(dim2, 1, kernel_size=1, padding=0))

        self.basicLayers = nn.ModuleList(self.basicLayer)
        self.deblockLayers = nn.ModuleList(self.deblockLayer)
        self.convs = nn.ModuleList(self.conv)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def load_sampling_matrix(self, ratio):
        path = "./data/sampling_matrix"
        data = np.load(os.path.join(path, str(int(ratio*100)) + '.npy'))
        return data

    def forward(self, x):
        H = int(x.shape[2] / self.imsize)
        L = int(x.shape[3] / self.imsize)
        S = x.shape[0]

        # Sampling
        b = self.sampling(x)
        # Init
        x_ref = torch.matmul(self.At, b)

        mu0 = (0.9 * torch.max(torch.abs(x_ref)))
        Gamma = (self.mu/mu0)**(1/self.n_layer)
        mu = mu0

        xk_i = x_ref
        xk_m = torch.unsqueeze(torch.reshape(torch.transpose(
            x_ref, 0, 1), [-1, self.imsize, self.imsize]), dim=1)
        xk_m = self.together(xk_m, S, H)
        wk = torch.zeros_like(x_ref)
        # Recon
        for i in range(self.n_layer):
            mu = mu * Gamma
            xk_i, wk, xk_it = self.basicLayers[i](b, mu, xk_i, wk)
            xk_i = torch.transpose(torch.reshape(
                xk_i, [-1, self.imsize * self.imsize]), 0, 1)
            xk_mt = self.together(xk_it, S, H)
            xk_m = self.deblockLayers[i](xk_m, xk_mt)
            xk_m_i = self.split(xk_m)
            xk_m_i = self.convs[i](xk_m_i)
            xk_m_i = torch.transpose(torch.reshape(
                xk_m_i, [-1, self.imsize * self.imsize]), 0, 1)
            xk_i = xk_i + xk_m_i
        xk_res = torch.unsqueeze(torch.reshape(torch.transpose(
            xk_i, 0, 1), [-1, self.imsize, self.imsize]), dim=1)
        xk_res = self.together(xk_res, S, H)
        return xk_res

    def sampling(self, X):
        inputs = self.split(X)
        inputs = torch.transpose(torch.reshape(
            inputs, [-1, self.imsize * self.imsize]), 0, 1)
        outputs = torch.matmul(self.A, inputs)
        return outputs

    def split(self, inputs):
        inputs = torch.cat(torch.split(
            inputs, split_size_or_sections=self.imsize, dim=2), dim=0)
        inputs = torch.cat(torch.split(
            inputs, split_size_or_sections=self.imsize, dim=3), dim=0)
        return inputs

    def together(self, inputs, S, H):
        inputs = torch.cat(torch.split(
            inputs, split_size_or_sections=H * S, dim=0), dim=3)
        inputs = torch.cat(torch.split(
            inputs, split_size_or_sections=S, dim=0), dim=2)
        return inputs


model_paths = {
    "tiny": "./saved_models/{}/tiny_{}_best.pkl",
    "base": "./saved_models/{}/base_{}_best.pkl",
    "xbase": "./saved_models/{}/xbase_{}_best.pkl",
    "large": "./saved_models/{}/large_{}_best.pkl",
}


@register_model
def base(ratio, pretrained=False, imsize=33, dephts=[1, 1, 3, 1], dims=[32, 64, 128, 256]):
    model = NesTDNet(ratio, imsize, dephts, dims)
    if pretrained:
        s_ratio = str(int(ratio*100))
        checkpoint = torch.load(model_paths['base'].format(
            s_ratio, s_ratio), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    return model

@register_model
def tiny(ratio, pretrained=False, imsize=33, dephts=[1, 1, 3, 1], dims=[16, 32, 64, 128]):
    model = NesTDNet(ratio, imsize, dephts, dims)
    if pretrained:
        s_ratio = str(int(ratio*100))
        checkpoint = torch.load(model_paths['tiny'].format(
            s_ratio, s_ratio), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    return model

@register_model
def xbase(ratio, pretrained=False, imsize=33, dephts=[1, 1, 3, 1], dims=[48, 96, 192, 384]):
    model = NesTDNet(ratio, imsize, dephts, dims)
    if pretrained:
        s_ratio = str(int(ratio*100))
        checkpoint = torch.load(model_paths['xbase'].format(
            s_ratio, s_ratio), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    return model

@register_model
def large(ratio, pretrained=False, imsize=33, dephts=[2, 2, 6, 2], dims=[32, 64, 128, 256]):
    model = NesTDNet(ratio, imsize, dephts, dims)
    if pretrained:
        s_ratio = str(int(ratio*100))
        checkpoint = torch.load(model_paths['large'].format(
            s_ratio, s_ratio), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    return model
