"""
@author: Zongyi Li and Daniel Zhengyu Huang
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..feedforward import FeedForward
from ..linear import WNLinear


class SpectralConv3d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, modes_z,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y, modes_z]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        x = self.forward_fourier(x)
        x = self.backcast_ff(x)

        return x

    def forward_fourier(self, x):
        x = rearrange(x, 'b s1 s2 s3 i -> b i s1 s2 s3')

        B, I, S1, S2, S3 = x.shape

        # # # Dimension Z # # #
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')

        out_ft = x_ft.new_zeros(B, I, S1, S2, S3 // 2 + 1)

        out_ft[:, :, :, :, :self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ft[:, :, :, :, :self.modes_z],
            torch.view_as_complex(self.fourier_weight[2]))

        x_out = torch.fft.irfft(out_ft, n=S3, dim=-1, norm='ortho')

        # # # Dimension Y # # #
        x_ft = torch.fft.rfft(x, dim=-2, norm='ortho')

        out_ft = x_ft.new_zeros(B, I, S1, S2 // 2 + 1, S3)

        out_ft[:, :, :, :self.modes_y, :] = torch.einsum(
            "bixyz,ioy->boxyz",
            x_ft[:, :, :, :self.modes_y, :],
            torch.view_as_complex(self.fourier_weight[1]))

        x_out += torch.fft.irfft(out_ft, n=S2, dim=-2, norm='ortho')

        # # # Dimension X # # #
        x_ft = torch.fft.rfft(x, dim=-3, norm='ortho')

        out_ft = x_ft.new_zeros(B, I, S1 // 2 + 1, S2, S3)

        out_ft[:, :, :self.modes_x, :, :] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ft[:, :, :self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight[0]))

        x_out += torch.fft.irfft(out_ft, n=S1, dim=-3, norm='ortho')

        # # Combining Dimensions # #
        #x = xx + xy + xz

        x = rearrange(x_out, 'b i s1 s2 s3 -> b s1 s2 s3 i')

        return x


class FNOFactorizedMesh3D(nn.Module):
    def __init__(self, modes_x, modes_y, modes_z, width, input_dim, output_dim,
                 n_layers, share_weight, factor, ff_weight_norm, n_ff_layers,
                 layer_norm):
        super().__init__()
        self.padding = 8  # pad the domain if input is non-periodic
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_lift = WNLinear(input_dim, self.width, wnorm=ff_weight_norm)
        self.n_layers = n_layers

        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y, modes_z]:
                weight = torch.FloatTensor(width, width, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv3d(in_dim=width,
                                                       out_dim=width,
                                                       modes_x=modes_x,
                                                       modes_y=modes_y,
                                                       modes_z=modes_z,
                                                       fourier_weight=self.fourier_weight,
                                                       factor=factor,
                                                       ff_weight_norm=ff_weight_norm,
                                                       n_ff_layers=n_ff_layers,
                                                       layer_norm=layer_norm,
                                                       dropout=0.0))

        self.out_proj = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, output_dim, wnorm=ff_weight_norm))

    def forward(self, x):
        x = rearrange(x, 'b i s1 s2 s3 -> b s1 s2 s3 i')
        x = self.in_lift(x) # [B, X, Y, Z, H]

        for i in range(self.n_layers):
            x = x + self.spectral_layers[i](x)

        x = self.out_proj(x)
        x = rearrange(x, 'b s1 s2 s3 i -> b i s1 s2 s3')

        return x
