"""Code is adapted from https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/util.py"""
# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!
from inspect import isfunction
import math
import torch
import torch.nn as nn
import numpy as np


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def extract_into_tensor(a, t, x_shape, batch_axis=0):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    out_shape = [1, ] * len(x_shape)
    out_shape[batch_axis] = batch_size
    return out.reshape(out_shape)


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor, batch_axis=0):
    """
    Take the mean over all non-batch dimensions.
    """
    mean_dim = list(range(len(tensor.shape)))
    mean_dim.pop(batch_axis)
    return tensor.mean(dim=mean_dim)


def noise_like(shape, device):
    return torch.randn(shape, device=device)


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################


import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import repeat


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    num_groups = min(32, channels)
    return nn.GroupNorm(num_groups, channels)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def round_to(dat, c):
    return dat + (dat - dat % c) % c


def get_activation(act, inplace=False, **kwargs):
    """

    Parameters
    ----------
    act
        Name of the activation
    inplace
        Whether to perform inplace activation

    Returns
    -------
    activation_layer
        The activation
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            negative_slope = kwargs.get("negative_slope", 0.1)
            return nn.LeakyReLU(negative_slope, inplace=inplace)
        elif act == 'identity':
            return nn.Identity()
        elif act == 'elu':
            return nn.ELU(inplace=inplace)
        elif act == 'gelu':
            return nn.GELU()
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'softrelu' or act == 'softplus':
            return nn.Softplus()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError('act="{}" is not supported. '
                                      'Try to include it if you can find that in '
                                      'https://pytorch.org/docs/stable/nn.html'.format(act))
    else:
        return act


def get_norm_layer(norm_type: str = 'layer_norm',
                   axis: int = -1,
                   epsilon: float = 1e-5,
                   in_channels: int = 0, **kwargs):
    """Get the normalization layer based on the provided type

    Parameters
    ----------
    norm_type
        The type of the layer normalization from ['layer_norm']
    axis
        The axis to normalize the
    epsilon
        The epsilon of the normalization layer
    in_channels
        Input channel

    Returns
    -------
    norm_layer
        The layer normalization layer
    """
    if isinstance(norm_type, str):
        if norm_type == 'layer_norm':
            assert in_channels > 0
            assert axis == -1
            norm_layer = nn.LayerNorm(normalized_shape=in_channels, eps=epsilon, **kwargs)
        else:
            raise NotImplementedError('norm_type={} is not supported'.format(norm_type))
        return norm_layer
    elif norm_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError('The type of normalization must be str')


def _generalize_padding(x, pad_t, pad_h, pad_w, padding_type, t_pad_left=False):
    """

    Parameters
    ----------
    x
        Shape (B, T, H, W, C)
    pad_t
    pad_h
    pad_w
    padding_type
    t_pad_left

    Returns
    -------
    out
        The result after padding the x. Shape will be (B, T + pad_t, H + pad_h, W + pad_w, C)
    """
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    assert padding_type in ['zeros', 'ignore', 'nearest']
    B, T, H, W, C = x.shape

    if padding_type == 'nearest':
        return F.interpolate(x.permute(0, 4, 1, 2, 3), size=(T + pad_t, H + pad_h, W + pad_w)).permute(0, 2, 3, 4, 1)
    else:
        if t_pad_left:
            return F.pad(x, (0, 0, 0, pad_w, 0, pad_h, pad_t, 0))
        else:
            return F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_t))


def _generalize_unpadding(x, pad_t, pad_h, pad_w, padding_type):
    assert padding_type in['zeros', 'ignore', 'nearest']
    B, T, H, W, C = x.shape
    if pad_t == 0 and pad_h == 0 and pad_w == 0:
        return x

    if padding_type == 'nearest':
        return F.interpolate(x.permute(0, 4, 1, 2, 3), size=(T - pad_t, H - pad_h, W - pad_w)).permute(0, 2, 3, 4, 1)
    else:
        return x[:, :(T - pad_t), :(H - pad_h), :(W - pad_w), :].contiguous()


def apply_initialization(m,
                         linear_mode="0",
                         conv_mode="0",
                         norm_mode="0",
                         embed_mode="0"):
    if isinstance(m, nn.Linear):
        if linear_mode in ("0", ):
            nn.init.kaiming_normal_(m.weight,
                                    mode='fan_in', nonlinearity="linear")
        elif linear_mode in ("1", ):
            nn.init.kaiming_normal_(m.weight,
                                    a=0.1,
                                    mode='fan_out',
                                    nonlinearity="leaky_relu")
        elif linear_mode in ("2", ):
            nn.init.zeros_(m.weight)
        else:
            raise NotImplementedError
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        if conv_mode in ("0", ):
            m.reset_parameters()
            # # default init of ConvNd in PyTorch 1.13, see https://github.com/pytorch/pytorch/blob/11aab72dc9da488832326a066d2e47520e4ab2b3/torch/nn/modules/conv.py#L146-L155
            # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            # if m.bias is not None:
            #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            #     if fan_in != 0:
            #         bound = 1 / math.sqrt(fan_in)
            #         nn.init.uniform_(m.bias, -bound, bound)
        elif conv_mode in ("1", ):
            nn.init.kaiming_normal_(m.weight,
                                    a=0.1,
                                    mode='fan_out',
                                    nonlinearity="leaky_relu")
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif conv_mode in ("2", ):
            nn.init.zeros_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError

    elif isinstance(m, nn.LayerNorm):
        if norm_mode in ("0", ):
            if m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError

    elif isinstance(m, nn.GroupNorm):
        if norm_mode in ("0", ):
            if m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        else:
            raise NotImplementedError
    # # pos_embed already initialized when created
    elif isinstance(m, nn.Embedding):
        if embed_mode in ("0", ):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
        else:
            raise NotImplementedError
    else:
        pass


class WrapIdentity(nn.Identity):

    def __init__(self):
        super(WrapIdentity, self).__init__()

    def reset_parameters(self):
        pass