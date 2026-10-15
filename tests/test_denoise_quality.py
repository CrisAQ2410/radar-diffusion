from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from datasets.radar_dataset import RadarDataset
from models.diffusion import GaussianDiffusion
from models.LDM.cuboid_transformer_unet import CuboidTransformerUNet
from models.ldm import RadarDiffusionModel


CONFIG_PATH = "configs/radar_ldm.yaml"
DEVICE = "cuda:2"
CHECKPOINT_NAME = "best_cc.pt"
OUTPUT_DIR = "outputs/diagnostics"
SEED = 12345

TIMESTEPS_TO_TEST = [0, 100, 300, 500, 700, 900]


def build_denoiser(cfg):
    m = cfg["model"]
    patterns = [m["self_pattern"]] * len(m["depth"])

    return CuboidTransformerUNet(
        input_shape=m["input_shape"],
        target_shape=m["target_shape"],
        base_units=m["base_units"],
        scale_alpha=m["scale_alpha"],
        depth=m["depth"],
        downsample=m["downsample"],
        downsample_type=m["downsample_type"],
        upsample_type=m["upsample_type"],
        upsample_kernel_size=m["upsample_kernel_size"],
        block_attn_patterns=patterns,
        num_heads=m["num_heads"],
        attn_drop=m["attn_drop"],
        proj_drop=m["proj_drop"],
        ffn_drop=m["ffn_drop"],
        ffn_activation=m["ffn_activation"],
        gated_ffn=m["gated_ffn"],
        norm_layer=m["norm_layer"],
        padding_type=m["padding_type"],
        pos_embed_type=m["pos_embed_type"],
        checkpoint_level=m["checkpoint_level"],
        use_relative_pos=m["use_relative_pos"],
        self_attn_use_final_proj=m["self_attn_use_final_proj"],
        num_global_vectors=m["num_global_vectors"],
        use_global_vector_ffn=m["use_global_vector_ffn"],
        use_global_self_attn=m["use_global_self_attn"],
        separate_global_qkv=m["separate_global_qkv"],
        global_dim_ratio=m["global_dim_ratio"],
        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        ffn2_linear_init_mode="2",
        attn_proj_linear_init_mode="2",
        conv_init_mode="0",
        down_linear_init_mode="0",
        up_linear_init_mode="0",
        global_proj_linear_init_mode="2",
        norm_init_mode="0",
        time_embed_channels_mult=m["time_embed_channels_mult"],
        time_embed_use_scale_shift_norm=m["time_embed_use_scale_shift_norm"],
        time_embed_dropout=m["time_embed_dropout"],
        unet_res_connect=m["unet_res_connect"],
    )


def find_rainiest_sample(dataset, step=10):
    best_idx = None
    best_score = -1.0

    for idx in range(0, len(dataset), step):
        _, future = dataset[idx]
        score = future.sum().item()

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def corrcoef(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


def main():
    Path(OUTPUT_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    d = cfg["data"]

    dataset = RadarDataset(
        root_dir=d["root"],
        split="test",
        input_frames=d["input_frames"],
        output_frames=d["output_frames"],
        image_size=d["image_size"],
        max_rainfall=d["max_rainfall"],
    )

    idx = find_rainiest_sample(
        dataset,
        step=10,
    )

    print("Sample:", idx)

    past, future = dataset[idx]

    past = (
        past
        .unsqueeze(0)
        .to(device)
    )

    future = (
        future
        .unsqueeze(0)
        .to(device)
    )

    denoiser = build_denoiser(
        cfg
    ).to(device)

    diffusion = GaussianDiffusion(
        timesteps=cfg["diffusion"]["timesteps"],
    ).to(device)

    model = RadarDiffusionModel(
        denoiser=denoiser,
        diffusion=diffusion,
    ).to(device)

    checkpoint_path = (
        Path(cfg["output"]["dir"])
        / CHECKPOINT_NAME
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.denoiser.load_state_dict(
        checkpoint["denoiser"]
    )

    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Checkpoint val:",
        checkpoint.get("val_loss"),
    )

    past_cl = model.to_channels_last(
        past
    )

    future_cl = model.to_channels_last(
        future
    )

    generator = torch.Generator(
        device=device
    )
    generator.manual_seed(
        SEED
    )

    fixed_noise = torch.randn(
        future_cl.shape,
        device=device,
        generator=generator,
    )

    truth_np = (
        future[0]
        .cpu()
        .numpy()
    )

    rows = []

    with torch.no_grad():
        for t_value in TIMESTEPS_TO_TEST:
            t = torch.tensor(
                [t_value],
                device=device,
                dtype=torch.long,
            )

            x_t = diffusion.q_sample(
                x_start=future_cl,
                t=t,
                noise=fixed_noise,
            )

            with torch.cuda.amp.autocast(
                enabled=(device.type == "cuda")
            ):
                pred = model.denoiser(
                    x_t,
                    t,
                    past_cl,
                )

            pred = (
                pred
                .float()
                .clamp(0.0, 1.0)
            )

            pred_np = (
                pred[0, ..., 0]
                .cpu()
                .numpy()
            )

            mae_norm = np.mean(
                np.abs(
                    pred_np - truth_np
                )
            )

            rmse_norm = np.sqrt(
                np.mean(
                    (pred_np - truth_np) ** 2
                )
            )

            cc = corrcoef(
                pred_np,
                truth_np,
            )

            mean_val = pred_np.mean()
            max_val = pred_np.max()

            print(
                f"t={t_value:4d} | "
                f"MAE_norm={mae_norm:.6f} | "
                f"RMSE_norm={rmse_norm:.6f} | "
                f"CC={cc:.4f} | "
                f"pred_mean={mean_val:.6f} | "
                f"pred_max={max_val:.6f} | "
                f"RMSE_mm={rmse_norm * d['max_rainfall']:.3f}"
            )

            rows.append(
                (t_value, pred_np)
            )

    # Compare one representative future frame (frame 3)
    frame_idx = 2

    fig, axes = plt.subplots(
        2,
        len(rows),
        figsize=(4 * len(rows), 8),
    )

    vmax = max(
        truth_np[frame_idx].max(),
        max(
            pred_np[frame_idx].max()
            for _, pred_np in rows
        ),
        1e-6,
    )

    for col, (t_value, pred_np) in enumerate(rows):
        axes[0, col].imshow(
            truth_np[frame_idx],
            cmap="jet",
            vmin=0,
            vmax=vmax,
        )
        axes[0, col].set_title(
            f"GT frame 3\n(t={t_value})"
        )
        axes[0, col].axis("off")

        axes[1, col].imshow(
            pred_np[frame_idx],
            cmap="jet",
            vmin=0,
            vmax=vmax,
        )
        axes[1, col].set_title(
            f"Pred frame 3\nfrom q_sample t={t_value}"
        )
        axes[1, col].axis("off")

    plt.tight_layout()

    output_path = (
        Path(OUTPUT_DIR)
        / f"denoise_quality_sample_{idx}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )
    plt.close()

    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
