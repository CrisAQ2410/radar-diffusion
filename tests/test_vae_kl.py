import torch
import yaml

from models.VAE.autoencoder_kl import AutoencoderKL


with open(
    "configs/radar_vae.yaml"
) as f:
    cfg = yaml.safe_load(f)


m = cfg["model"]


model = AutoencoderKL(
    in_channels=m["in_channels"],
    out_channels=m["out_channels"],
    down_block_types=tuple(
        m["down_block_types"]
    ),
    up_block_types=tuple(
        m["up_block_types"]
    ),
    block_out_channels=tuple(
        m["block_out_channels"]
    ),
    layers_per_block=m[
        "layers_per_block"
    ],
    latent_channels=m[
        "latent_channels"
    ],
    norm_num_groups=m[
        "norm_num_groups"
    ],
)


x = torch.randn(
    2,
    1,
    256,
    256,
)


with torch.no_grad():
    posterior = model.encode(x)

    z = posterior.mode()

    recon = model.decode(z)


print("Input :", x.shape)
print("Mean  :", posterior.mean.shape)
print("Logvar:", posterior.logvar.shape)
print("Latent:", z.shape)
print("Recon :", recon.shape)