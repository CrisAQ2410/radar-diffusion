import torch
import yaml

from torch.utils.data import DataLoader

from datasets.radar_dataset import RadarDataset
from models.VAE.autoencoder_kl import AutoencoderKL


device = "cuda:2"


with open("configs/radar_vae.yaml") as f:
    cfg = yaml.safe_load(f)


m = cfg["model"]


vae = AutoencoderKL(
    in_channels=m["in_channels"],
    out_channels=m["out_channels"],
    down_block_types=tuple(m["down_block_types"]),
    up_block_types=tuple(m["up_block_types"]),
    block_out_channels=tuple(m["block_out_channels"]),
    layers_per_block=m["layers_per_block"],
    latent_channels=m["latent_channels"],
    norm_num_groups=m["norm_num_groups"],
).to(device)


vae.load_state_dict(
    torch.load(
        "checkpoints/vae_kl/vae.pt",
        map_location=device
    )
)

vae.eval()

for p in vae.parameters():
    p.requires_grad = False


dataset = RadarDataset(
    "/home/student/chipk/Radar"
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)


past, future = next(iter(loader))

print("Past:", past.shape)
print("Future:", future.shape)


def encode_sequence(x):

    # [B,T,H,W]
    B, T, H, W = x.shape

    x = x.reshape(
        B * T,
        1,
        H,
        W
    ).to(device)

    with torch.no_grad():

        posterior = vae.encode(x)

        # deterministic latent
        z = posterior.mode()

    # [B*T,64,32,32]
    _, C, h, w = z.shape

    z = z.reshape(
        B,
        T,
        C,
        h,
        w
    )

    # senior LDM format:
    # [B,T,H,W,C]
    z = z.permute(
        0, 1, 3, 4, 2
    ).contiguous()

    return z


z_past = encode_sequence(past)
z_future = encode_sequence(future)


print(
    "Past latent:",
    z_past.shape
)

print(
    "Future latent:",
    z_future.shape
)