import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

from datasets.radar_dataset import RadarDataset
from inference.sample_ldm import (
    build_vae,
    build_denoiser,
    decode_sequence,
    to_rainfall,
)
from models.ldm import RadarLatentDiffusion
from utils.ema import LitEma


DEVICE = "cuda:2"
CKPT = "checkpoints/ldm_mode/best.pt"
SAMPLE_IDX = 5690


with open("configs/radar_ldm.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device(DEVICE)

vae = build_vae(cfg).to(device)
denoiser = build_denoiser(cfg).to(device)

d = cfg["diffusion"]

model = RadarLatentDiffusion(
    vae=vae,
    denoiser=denoiser,
    timesteps=d["timesteps"],
    cosine_s=d["cosine_s"],
    learn_logvar=d["learn_logvar"],
    logvar_init=d["logvar_init"],
).to(device)

ckpt = torch.load(
    CKPT,
    map_location=device
)

model.denoiser.load_state_dict(
    ckpt["denoiser"]
)

ema = LitEma(
    model.denoiser,
    decay=cfg["training"]["ema_decay"]
).to(device)

ema.load_state_dict(
    ckpt["ema"]
)

ema.copy_to(
    model.denoiser
)

model.eval()


data_cfg = cfg["data"]

dataset = RadarDataset(
    root_dir=data_cfg["root"],
    split="test",
    input_frames=data_cfg["input_frames"],
    output_frames=data_cfg["output_frames"],
    image_size=data_cfg["image_size"],
    max_rainfall=data_cfg["max_rainfall"],
    val_ratio=data_cfg["val_ratio"],
    test_year=data_cfg["test_year"],
)

past, future = dataset[SAMPLE_IDX]

past = past.unsqueeze(0).to(device)
future = future.unsqueeze(0).to(device)


with torch.no_grad():

    # condition
    z_past = model.encode_sequence(
        past,
        sample_posterior=False
    )

    # target future
    z_future = model.encode_sequence(
        future,
        sample_posterior=True
    )


timesteps = [100, 500, 900, 950, 990, 999]

fig, axes = plt.subplots(
    len(timesteps) + 1,
    6,
    figsize=(18, 12)
)


# GT
gt = to_rainfall(
    future[0].cpu().numpy(),
    data_cfg["max_rainfall"]
)

for i in range(6):
    axes[0, i].imshow(
        gt[i],
        cmap="jet",
        vmin=0,
        vmax=max(gt.max(), 1)
    )
    axes[0, i].set_title(
        f"GT {i+1}"
    )
    axes[0, i].axis("off")


for row, timestep in enumerate(
    timesteps,
    start=1
):

    t = torch.tensor(
        [timestep],
        device=device
    ).long()

    noise = torch.randn_like(
        z_future
    )

    z_noisy = model.diffusion.q_sample(
        z_future,
        t,
        noise
    )

    with torch.no_grad():

        z_pred = model.denoiser(
            z_noisy,
            t,
            z_past
        )

        pred = decode_sequence(
            vae,
            z_pred
        )[0]

    pred = to_rainfall(
        pred.cpu().numpy(),
        data_cfg["max_rainfall"]
    )

    for i in range(6):

        axes[row, i].imshow(
            pred[i],
            cmap="jet",
            vmin=0,
            vmax=max(gt.max(), 1)
        )

        axes[row, i].set_title(
            f"t={timestep}, frame {i+1}"
        )

        axes[row, i].axis("off")


plt.tight_layout()

plt.savefig(
    "outputs/ldm_denoise_test.png",
    dpi=150
)

print(
    "Saved outputs/ldm_denoise_test.png"
)