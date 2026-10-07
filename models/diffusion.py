import math

import torch
import torch.nn as nn


def make_cosine_beta_schedule(
    timesteps,
    cosine_s=0.008
):
    steps = timesteps + 1

    x = torch.linspace(
        0,
        timesteps,
        steps,
        dtype=torch.float64
    )

    alphas_cumprod = torch.cos(
        (
            (x / timesteps) + cosine_s
        )
        /
        (1 + cosine_s)
        * math.pi
        * 0.5
    ) ** 2

    alphas_cumprod = (
        alphas_cumprod
        /
        alphas_cumprod[0]
    )

    betas = (
        1
        -
        (
            alphas_cumprod[1:]
            /
            alphas_cumprod[:-1]
        )
    )

    return betas.clamp(
        0.0001,
        0.9999
    ).float()


def extract(
    values,
    t,
    shape
):
    out = values.gather(
        0,
        t
    )

    return out.reshape(
        t.shape[0],
        *([1] * (len(shape) - 1))
    )


class DiffusionSchedule(nn.Module):

    def __init__(
        self,
        timesteps=1000,
        cosine_s=0.008
    ):
        super().__init__()

        self.timesteps = timesteps

        betas = make_cosine_beta_schedule(
            timesteps,
            cosine_s
        )

        alphas = 1.0 - betas

        alphas_cumprod = torch.cumprod(
            alphas,
            dim=0
        )

        alphas_cumprod_prev = torch.cat([
            torch.ones(1),
            alphas_cumprod[:-1]
        ])

        posterior_variance = (
            betas
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )

        posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20)
        )

        posterior_mean_coef1 = (
            betas
            * torch.sqrt(alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )

        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod)
        )

        self.register_buffer(
            "betas",
            betas
        )

        self.register_buffer(
            "alphas_cumprod",
            alphas_cumprod
        )

        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(
                alphas_cumprod
            )
        )

        self.register_buffer(
            "alphas_cumprod_prev",
            alphas_cumprod_prev
        )

        self.register_buffer(
            "posterior_variance",
            posterior_variance
        )

        self.register_buffer(
            "posterior_log_variance_clipped",
            posterior_log_variance_clipped
        )

        self.register_buffer(
            "posterior_mean_coef1",
            posterior_mean_coef1
        )

        self.register_buffer(
            "posterior_mean_coef2",
            posterior_mean_coef2
        )

        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(
                1.0
                -
                alphas_cumprod
            )
        )

    def q_sample(
        self,
        x_start,
        t,
        noise=None
    ):

        if noise is None:
            noise = torch.randn_like(
                x_start
            )

        signal = extract(
            self.sqrt_alphas_cumprod,
            t,
            x_start.shape
        )

        noise_rate = extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_start.shape
        )

        return (
            signal * x_start
            +
            noise_rate * noise
        )