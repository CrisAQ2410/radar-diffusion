import torch
from torch import nn


class RadarDiffusionModel(nn.Module):

    def __init__(
        self,
        denoiser,
        diffusion,
    ):
        super().__init__()

        self.denoiser = denoiser
        self.diffusion = diffusion

    # =====================================================
    # shape helper
    # =====================================================

    @staticmethod
    def to_channels_last(x):
        """
        [B,T,H,W]
        ->
        [B,T,H,W,1]
        """

        if x.ndim == 4:
            x = x.unsqueeze(-1)

        return x

    # =====================================================
    # forward training
    # =====================================================

    def forward(
        self,
        past,
        future,
        t=None,
        noise=None,
    ):

        # -----------------------------------------------
        # Dataset:
        # [B,6,80,80]
        #
        # CuboidTransformer:
        # [B,6,80,80,1]
        # -----------------------------------------------

        past = self.to_channels_last(
            past
        )

        future = self.to_channels_last(
            future
        )

        B = future.shape[0]

        # -----------------------------------------------
        # random diffusion timestep
        # -----------------------------------------------

        if t is None:

            t = torch.randint(
                0,
                self.diffusion.timesteps,
                (B,),
                device=future.device,
            ).long()

        # -----------------------------------------------
        # Gaussian noise
        # -----------------------------------------------

        if noise is None:

            noise = torch.randn_like(
                future
            )

        # -----------------------------------------------
        # forward diffusion:
        #
        # x_t =
        # sqrt(alpha_bar_t) * x0
        # +
        # sqrt(1-alpha_bar_t) * noise
        # -----------------------------------------------

        x_t = self.diffusion.q_sample(
            x_start=future,
            t=t,
            noise=noise,
        )

        # -----------------------------------------------
        # DIRECT x0 prediction
        # -----------------------------------------------

        x0_pred = self.denoiser(
            x_t,
            t,
            past,
        )

        return {
            "pred": x0_pred,
            "target": future,
            "x_t": x_t,
            "t": t,
            "noise": noise,
        }