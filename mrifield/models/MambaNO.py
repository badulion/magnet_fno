import math
from functools import partial
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass

import numpy as np

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

class LReLu(torch.nn.Module):
    def __init__(self,
        in_channels,                    
        out_channels,                   
        in_size,                        
        out_size,                       
        in_sampling_rate,               
        out_sampling_rate,             
        in_cutoff,                    
        out_cutoff,                     
        in_half_width,                  
        out_half_width,                 
        filter_size         = 6,        
        lrelu_upsampling    = 2,        
        is_critically_sampled = False,  
        use_radial_filters    = False,  
    ):
        super().__init__()
        
        self.is_critically_sampled = is_critically_sampled
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) *lrelu_upsampling
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1  else 1
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

        pad_total = (self.out_size - 1) * self.down_factor + 1 
        pad_total -= (self.in_size * self.up_factor)
        pad_total += self.up_taps + self.down_taps - 2 
                
        pad_lo = (pad_total + self.up_factor) // 2 
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]
            

    def forward(self, x, noise_mode='random', force_fp32=False, update_emas=False):
                 
        dtype = torch.float32
        gain = np.sqrt(2)
        slope = 0.2
        
        # Execute bias, filtered, and clamping.
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=None)

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert x.dtype == dtype
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False): 
        assert numtaps >= 1 

        # Identity filter.
        if numtaps == 1:
            return None

        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])

class Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'lrelu'
                 ):
        super(Block, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size  = in_size 
        self.out_size = out_size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm
        self.citically_sampled = False 

        if cutoff_den == 2.0: 
            self.citically_sampled = True 
        self.in_cutoff  = self.in_size / cutoff_den 
        self.out_cutoff = self.out_size / cutoff_den 
        
        self.in_halfwidth =  half_width_mult*self.in_size - self.in_size / cutoff_den 
        self.out_halfwidth = half_width_mult*self.out_size - self.out_size / cutoff_den 
        
        pad = (self.conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels = self.in_channels, out_channels=self.out_channels, 
                                           kernel_size=self.conv_kernel, 
                                           padding = pad)
    
        if self.batch_norm:
            self.batch_norm  = nn.BatchNorm2d(self.out_channels)
        
        if activation == "lrelu":
            self.activation  = LReLu(in_channels           = self.in_channels, 
                                     out_channels          = self.out_channels,                   
                                     in_size               = self.in_size,                       
                                     out_size              = self.out_size,                       
                                     in_sampling_rate      = self.in_size,              
                                     out_sampling_rate     = self.out_size,             
                                     in_cutoff             = self.in_cutoff,                     
                                     out_cutoff            = self.out_cutoff,                   
                                     in_half_width         = self.in_halfwidth,             
                                     out_half_width        = self.out_halfwidth,            
                                     filter_size           = filter_size,       
                                     lrelu_upsampling      = lrelu_upsampling,
                                     is_critically_sampled = self.citically_sampled,
                                     use_radial_filters    = False)
        else:
            raise ValueError("Please specify different activation function")
        
    def forward(self, x):
        x = self.convolution(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return self.activation(x)
    
class LiftProjectBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 latent_dim = 64,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'lrelu'
                 ):
        super(LiftProjectBlock, self).__init__()
    
        self.Block = Block(in_channels = in_channels,
                                    out_channels = latent_dim,
                                    in_size = in_size,
                                    out_size = out_size,
                                    cutoff_den = cutoff_den,
                                    conv_kernel = conv_kernel,
                                    filter_size = filter_size,
                                    lrelu_upsampling = lrelu_upsampling,
                                    half_width_mult  = half_width_mult,
                                    radial = radial,
                                    batch_norm = batch_norm,
                                    activation = activation)
        
        pad = (conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels = latent_dim, out_channels=out_channels, 
                                           kernel_size=conv_kernel, stride = 1, 
                                           padding = pad)
        
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm  = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.Block(x)
        x = self.convolution(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return x
            
class Up_Down_Sampling(nn.Module):
    def __init__(self, dim, out_channels, insize, outsize, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_channels = dim
        self.out_channels = out_channels
        self.in_size = insize
        self.out_size = outsize
        self.conv_kernel = 1
        pad = (self.conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels = self.in_channels, out_channels=self.out_channels, kernel_size=self.conv_kernel, padding = pad)
        
        cutoff_den = 2.0001
        half_width_mult = 0.8
        self.in_cutoff  = self.in_size / cutoff_den
        self.out_cutoff = self.out_size / cutoff_den
        self.in_halfwidth = half_width_mult*self.in_size - self.in_size / cutoff_den 
        self.out_halfwidth = half_width_mult*self.out_size - self.out_size / cutoff_den
        self.act = LReLu(in_channels           = self.in_channels, 
                         out_channels          = self.out_channels,                   
                         in_size               = self.in_size,                       
                         out_size              = self.out_size,                       
                         in_sampling_rate      = self.in_size,    
                         out_sampling_rate     = self.out_size,               
                         in_cutoff             = self.in_cutoff,                 
                         out_cutoff            = self.out_cutoff,  
                         in_half_width         = self.in_halfwidth,  
                         out_half_width        = self.out_halfwidth, 
                         filter_size           = 6,       
                         lrelu_upsampling      = 2,
                         is_critically_sampled = False,
                         use_radial_filters    = False)
        
    def forward(self, x):
        x = self.convolution(x)
        x = self.act(x)
        return x   

class Mamba_Integration(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        insize = 64,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.in_size = insize
        cutoff_den = 2.0001
        half_width_mult = 0.8
        self.in_cutoff  = self.in_size / cutoff_den
        self.in_halfwidth = half_width_mult*self.in_size - self.in_size / cutoff_den 
        self.act = LReLu(in_channels           = self.d_inner,
                         out_channels          = self.d_inner,                   
                         in_size               = self.in_size,                       
                         out_size              = self.in_size,                       
                         in_sampling_rate      = self.in_size,            
                         out_sampling_rate     = self.in_size,               
                         in_cutoff             = self.in_cutoff,                    
                         out_cutoff            = self.in_cutoff, 
                         in_half_width         = self.in_halfwidth,  
                         out_half_width        = self.in_halfwidth,  
                         filter_size           = 6,       
                         lrelu_upsampling      = 2,
                         is_critically_sampled = False,
                         use_radial_filters    = False)

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) 
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.contiguous().view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)                               # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)                # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)                            # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)                            # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)                                # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)          # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)   # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        
        z = z.permute(0,3,1,2)
        z = nn.functional.interpolate(z, size = 2*self.in_size,mode='bicubic', antialias = True)
        z = F.silu(z)
        z = nn.functional.interpolate(z, size = self.in_size,mode='bicubic', antialias = True)
        z = z.permute(0,2,3,1)
        
        y = y * z
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class MambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        insize: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.Integration = Mamba_Integration(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, insize=insize, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.Integration(self.ln_1(input)))
        return x

class MambaLayer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        insize = 64,
        outsize = 64,
        out_channels = 128,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            MambaBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                insize = insize,
            )
            for i in range(depth)])
        
        if True: 
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() 
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, out_channels=out_channels, insize=insize, outsize=outsize, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = x.permute(0,3,1,2)
            x = self.downsample(x)
            x = x.permute(0,2,3,1)
        return x
    
class MambaLayer_up(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        insize = 64,
        beforesize = 64,
        beforechannels = 128,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            MambaBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                insize = insize,
            )
            for i in range(depth)])
        
        if True: 
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() 
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=beforechannels, out_channels=dim, insize=beforesize, outsize=insize, norm_layer=norm_layer)
        else:
            self.upsample = None


    def forward(self, x):
        if self.upsample is not None:
            x = x.permute(0,3,1,2)
            x = self.upsample(x)
            x = x.permute(0,2,3,1)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

class Convolution_Integration(nn.Module):
    def __init__(self,
                 channels,
                 size,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'lrelu'
                 ):
        super(Convolution_Integration, self).__init__()

        self.channels = channels
        self.size  = size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm

        self.citically_sampled = False 

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.cutoff  = self.size / cutoff_den        
        self.halfwidth =  half_width_mult*self.size - self.size / cutoff_den
        
        
        pad = (self.conv_kernel-1)//2
        self.convolution1 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels, 
                                           kernel_size=self.conv_kernel, stride = 1, 
                                           padding = pad)
        self.convolution2 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels, 
                                           kernel_size=self.conv_kernel, stride = 1, 
                                           padding = pad)
        
        if self.batch_norm:
            self.batch_norm1  = nn.BatchNorm2d(self.channels)
            self.batch_norm2  = nn.BatchNorm2d(self.channels)
        
        if activation == "lrelu":
            self.activation  = LReLu(in_channels           = self.channels,
                                     out_channels          = self.channels,                   
                                     in_size               = self.size,                       
                                     out_size              = self.size,                       
                                     in_sampling_rate      = self.size,               
                                     out_sampling_rate     = self.size,             
                                     in_cutoff             = self.cutoff,                     
                                     out_cutoff            = self.cutoff,                  
                                     in_half_width         = self.halfwidth,                
                                     out_half_width        = self.halfwidth,              
                                     filter_size           = filter_size,       
                                     lrelu_upsampling      = lrelu_upsampling,
                                     is_critically_sampled = self.citically_sampled,
                                     use_radial_filters    = False)
        else:
            raise ValueError("Please specify different activation function")
            

    def forward(self, x):
        out = self.convolution1(x)
        if self.batch_norm:
            out = self.batch_norm1(out)
        out = self.activation(out)
        out = self.convolution2(out)
        if self.batch_norm:
            out = self.batch_norm2(out)
        
        return x + out
    
class MambaNO(nn.Module):
    def __init__(self, output_channel=1, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 2], dims=[16, 32, 64, 128], 
                 dims_decoder=[128, 64, 32, 16], encoder_sizes = [64,32,16,8], decoder_sizes = [8,16,32,64],
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, N_res = 2, N_res_neck = 6, 
                 norm_layer=nn.LayerNorm, use_checkpoint=False, **kwargs):
        super().__init__()
        
        self.output_channel = output_channel
        self.num_layers = len(depths)

        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes
        
        self.Convolution_nets = []
        self.N_res = N_res
        self.N_res_neck = N_res_neck        
        
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.dims = dims

        self.lift = LiftProjectBlock(in_channels=1,out_channels=self.embed_dim,in_size=64,out_size=64,conv_kernel = 3)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MambaLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=Up_Down_Sampling if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                insize=self.encoder_sizes[i_layer],
                outsize=self.encoder_sizes[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                out_channels=dims[i_layer+1] if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = MambaLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=Up_Down_Sampling if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
                insize=self.decoder_sizes[i_layer], 
                beforesize=self.decoder_sizes[i_layer-1] if (i_layer != 0) else None,
                beforechannels=dims_decoder[i_layer-1] if (i_layer != 0) else None,
            )
            self.layers_up.append(layer)

        self.project = LiftProjectBlock(in_channels=16,out_channels=1,in_size=self.encoder_sizes[0],out_size=self.encoder_sizes[0],conv_kernel = 3)
        

        for l in range(self.num_layers-1):
            for i in range(self.N_res):
                self.Convolution_nets.append(Convolution_Integration(channels = self.dims[l],
                                                            size     = self.encoder_sizes[l],
                                                            cutoff_den = 2.0001,
                                                            conv_kernel = 3,
                                                            filter_size = 6,
                                                            lrelu_upsampling = 2,
                                                            half_width_mult  = 0.8,
                                                            radial = False,
                                                            batch_norm = True,
                                                            activation = 'lrelu'))
        for i in range(self.N_res_neck):
            self.Convolution_nets.append(Convolution_Integration(channels = self.dims[self.num_layers-1],
                                                        size     = self.encoder_sizes[self.num_layers-1],
                                                        cutoff_den = 2.0001,
                                                        conv_kernel = 3,
                                                        filter_size = 6,
                                                        lrelu_upsampling = 2,
                                                        half_width_mult  = 0.8,
                                                        radial = False,
                                                        batch_norm = True,
                                                        activation = 'lrelu'))
        
        self.Convolution_nets = torch.nn.Sequential(*self.Convolution_nets)    
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        skip_list = []
        x = self.lift(x.contiguous())
        x = x.permute(0,2,3,1)
        x = self.pos_drop(x)

        for i in range(self.num_layers):
            y = x.permute(0,3,1,2)
            for j in range(self.N_res):
                y = self.Convolution_nets[i*self.N_res + j](y)
            y = y.permute(0,2,3,1)
            skip_list.append(y)
            x = self.layers[i](y)
        return x, skip_list
    
    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x+skip_list[-inx])
        return x
    
    def forward_final(self, x):
        x = x.permute(0,3,1,2).contiguous()
        x = self.project(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)
        
        return x




    


