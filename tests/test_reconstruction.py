import os
import torch
import yaml
import matplotlib.pyplot as plt

from datasets.vae_dataset import RadarVAEDataset
from models.VAE.autoencoder_kl import AutoencoderKL


device = "cuda:2"

with open("configs/radar_vae.yaml", "r") as f:
    cfg = yaml.safe_load(f)

m = cfg["model"]
d = cfg["data"]


dataset = RadarVAEDataset(
    root_dir=d["root"],
    split="val",
    image_size=d["image_size"],
    max_rainfall=d["max_rainfall"],
    train_ratio=d["train_ratio"],
    test_year=d["test_year"],
)


model = AutoencoderKL(
    in_channels=m["in_channels"],
    out_channels=m["out_channels"],
    down_block_types=tuple(m["down_block_types"]),
    up_block_types=tuple(m["up_block_types"]),
    block_out_channels=tuple(m["block_out_channels"]),
    layers_per_block=m["layers_per_block"],
    latent_channels=m["latent_channels"],
    norm_num_groups=m["norm_num_groups"],
).to(device)


model.load_state_dict(
    torch.load(
        "checkpoints/vae_kl/vae.pt",
        map_location=device
    )
)

model.eval()


os.makedirs(
    "outputs/vae_reconstruction",
    exist_ok=True
)


indices = [0, 100, 500, 1000]


for idx in indices:

    x = dataset[idx].unsqueeze(0).to(device)

    with torch.no_grad():

        posterior = model.encode(x)

        # Không random khi evaluate
        z = posterior.mode()

        recon = model.decode(z)


    original = (
        x[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    reconstructed = (
        recon[0, 0]
        .detach()
        .cpu()
        .clamp(0, 1)
        .numpy()
    )


    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4)
    )


    axes[0].imshow(
        original,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axes[0].set_title("Original")


    axes[1].imshow(
        reconstructed,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    axes[1].set_title("Reconstruction")


    diff = abs(
        original
        - reconstructed
    )

    axes[2].imshow(
        diff,
        cmap="hot"
    )

    axes[2].set_title("Absolute Error")


    for ax in axes:
        ax.axis("off")


    plt.tight_layout()

    plt.savefig(
        f"outputs/vae_reconstruction/sample_{idx}.png",
        dpi=150
    )

    plt.close()


print(
    "Saved to outputs/vae_reconstruction/"
)