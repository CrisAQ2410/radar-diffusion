import torch
import torch.nn as nn

from models.diffusion import DiffusionSchedule


class RadarLatentDiffusion(nn.Module):

    def __init__(
        self,
        vae,
        denoiser,
        timesteps=1000,
        cosine_s=0.008,
        learn_logvar=True,
        logvar_init=0.0
    ):
        super().__init__()

        self.vae = vae

        self.denoiser = denoiser

        self.diffusion = DiffusionSchedule(
            timesteps=timesteps,
            cosine_s=cosine_s
        )

        self.timesteps = timesteps

        # freeze VAE
        self.vae.eval()

        for p in self.vae.parameters():
            p.requires_grad = False

        self.learn_logvar = (
            learn_logvar
        )

        self.logvar = nn.Parameter(
            torch.full(
                (timesteps,),
                float(logvar_init)
            ),
            requires_grad=learn_logvar
        )

    @torch.no_grad()
    def encode_sequence(
        self,
        x,
        sample_posterior=True
    ):
        B, T, H, W = x.shape

        x = x.reshape(
            B * T,
            1,
            H,
            W
        )

        posterior = self.vae.encode(x)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        _, C, h, w = z.shape

        z = z.reshape(
            B,
            T,
            C,
            h,
            w
        )

        z = z.permute(
            0, 1, 3, 4, 2
        ).contiguous()

        return z

    def forward(
        self,
        past,
        future
    ):

        with torch.no_grad():

            z_past = self.encode_sequence(
                past,
                sample_posterior=False
            )

            z_future = self.encode_sequence(
                future,
                sample_posterior=True
            )

        B = z_future.shape[0]

        if self.training:

            # 50% vẫn học toàn bộ diffusion trajectory
            t = torch.randint(
                0,
                self.timesteps,
                (B,),
                device=z_future.device
            )

            # 50% batch ưu tiên vùng noise cao
            high_mask = (
                torch.rand(
                    B,
                    device=z_future.device
                ) < 0.5
            )

            high_t = torch.randint(
                900,
                self.timesteps,
                (B,),
                device=z_future.device
            )

            t = torch.where(
                high_mask,
                high_t,
                t
            )

        else:

            # validation vẫn uniform để so sánh công bằng
            t = torch.randint(
                0,
                self.timesteps,
                (B,),
                device=z_future.device
            )

        t = t.long()

        noise = torch.randn_like(
            z_future
        )

        z_noisy = self.diffusion.q_sample(
            x_start=z_future,
            t=t,
            noise=noise
        )

        # senior main config = x0 prediction
        prediction = self.denoiser(
            z_noisy,
            t,
            z_past
        )

        target = z_future

        loss_simple = (
            (prediction - target) ** 2
        ).mean(
            dim=(1, 2, 3, 4)
        )

        logvar_t = self.logvar[t]

        loss = (
            loss_simple
            /
            torch.exp(logvar_t)
            +
            logvar_t
        )

        loss = loss.mean()

        return {
            "loss": loss,
            "loss_simple":
                loss_simple.mean(),
            "logvar":
                logvar_t.mean(),
        }