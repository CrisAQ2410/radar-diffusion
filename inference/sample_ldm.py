import os
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from datasets.radar_dataset import RadarDataset
from models.diffusion import GaussianDiffusion
from models.LDM.cuboid_transformer_unet import CuboidTransformerUNet
from models.ldm import RadarDiffusionModel


DEVICE = "cuda:2"
CONFIG_PATH = "configs/radar_ldm.yaml"

# Validation trước đó còn stochastic, nên với run 5 epoch hiện tại
# dùng last.pt để kiểm tra đúng model sau epoch 5.
CHECKPOINT_NAME = "best_cc.pt"

OUTPUT_DIR = "outputs/image_diffusion_mse_samples"

# Chạy 1 sample trước vì DDPM cần đủ 1000 reverse steps.
NUM_SAMPLES = 1
SEARCH_STEP = 10
SEED = 12345


# ============================================================
# Build CuboidTransformer
# ============================================================

def build_denoiser(cfg):
    m = cfg["model"]

    patterns = [
        m["self_pattern"]
    ] * len(m["depth"])

    return CuboidTransformerUNet(
        input_shape=m["input_shape"],
        target_shape=m["target_shape"],

        base_units=m["base_units"],
        scale_alpha=m["scale_alpha"],

        depth=m["depth"],

        downsample=m["downsample"],
        downsample_type=m[
            "downsample_type"
        ],

        upsample_type=m[
            "upsample_type"
        ],

        upsample_kernel_size=m[
            "upsample_kernel_size"
        ],

        block_attn_patterns=patterns,

        num_heads=m["num_heads"],

        attn_drop=m["attn_drop"],
        proj_drop=m["proj_drop"],
        ffn_drop=m["ffn_drop"],

        ffn_activation=m[
            "ffn_activation"
        ],

        gated_ffn=m["gated_ffn"],

        norm_layer=m["norm_layer"],

        padding_type=m[
            "padding_type"
        ],

        pos_embed_type=m[
            "pos_embed_type"
        ],

        checkpoint_level=m[
            "checkpoint_level"
        ],

        use_relative_pos=m[
            "use_relative_pos"
        ],

        self_attn_use_final_proj=m[
            "self_attn_use_final_proj"
        ],

        num_global_vectors=m[
            "num_global_vectors"
        ],

        use_global_vector_ffn=m[
            "use_global_vector_ffn"
        ],

        use_global_self_attn=m[
            "use_global_self_attn"
        ],

        separate_global_qkv=m[
            "separate_global_qkv"
        ],

        global_dim_ratio=m[
            "global_dim_ratio"
        ],

        attn_linear_init_mode="0",
        ffn_linear_init_mode="0",
        ffn2_linear_init_mode="2",
        attn_proj_linear_init_mode="2",

        conv_init_mode="0",

        down_linear_init_mode="0",
        up_linear_init_mode="0",

        global_proj_linear_init_mode="2",
        norm_init_mode="0",

        time_embed_channels_mult=m[
            "time_embed_channels_mult"
        ],

        time_embed_use_scale_shift_norm=m[
            "time_embed_use_scale_shift_norm"
        ],

        time_embed_dropout=m[
            "time_embed_dropout"
        ],

        unet_res_connect=m[
            "unet_res_connect"
        ],
    )


# ============================================================
# Reverse DDPM
# x_t -> x_{t-1}, model predicts x0 directly
# ============================================================

@torch.no_grad()
def sample_ddpm(
    model,
    past,
    seed=SEED,
):
    schedule = model.diffusion

    # Dataset: [B, 6, 80, 80]
    # Denoiser: [B, 6, 80, 80, 1]
    past = model.to_channels_last(
        past
    )

    B = past.shape[0]

    generator = torch.Generator(
        device=past.device
    )
    generator.manual_seed(seed)

    # x_T ~ N(0, I)
    x = torch.randn(
        past.shape,
        device=past.device,
        dtype=past.dtype,
        generator=generator,
    )

    timesteps = schedule.timesteps

    for i in reversed(
        range(timesteps)
    ):
        t = torch.full(
            (B,),
            i,
            device=x.device,
            dtype=torch.long,
        )

        # Không cần gradient khi inference.
        # AMP giảm VRAM và tăng tốc.
        with torch.cuda.amp.autocast(
            enabled=(x.device.type == "cuda")
        ):
            x0_pred = model.denoiser(
                x,
                t,
                past,
            )

        # Không clamp x0 ở giữa reverse process để không tự ý
        # thay đổi dynamics so với công thức DDPM.
        coef1 = (
            schedule
            .posterior_mean_coef1[t]
            .reshape(
                B, 1, 1, 1, 1
            )
        )

        coef2 = (
            schedule
            .posterior_mean_coef2[t]
            .reshape(
                B, 1, 1, 1, 1
            )
        )

        model_mean = (
            coef1 * x0_pred.float()
            + coef2 * x.float()
        )

        if i > 0:
            log_variance = (
                schedule
                .posterior_log_variance_clipped[t]
                .reshape(
                    B, 1, 1, 1, 1
                )
            )

            noise = torch.randn(
                x.shape,
                device=x.device,
                dtype=torch.float32,
                generator=generator,
            )

            x = (
                model_mean
                + torch.exp(
                    0.5 * log_variance
                )
                * noise
            )
        else:
            x = model_mean

        if (
            i % 100 == 0
            or i == timesteps - 1
        ):
            print(
                f"Sampling timestep {i}"
            )

    # Final prediction ở image-space normalized [0,1].
    return x.clamp(
        0.0,
        1.0,
    )


# ============================================================
# Inverse Min-Max normalization
# Thesis preprocessing:
# x_norm = rain / 260
# ============================================================

def to_rainfall(
    x,
    max_rainfall,
):
    x = np.clip(
        x,
        0.0,
        1.0,
    )

    return (
        x
        * max_rainfall
    )


# ============================================================
# Basic sample metrics
# ============================================================

def sample_metrics(
    pred,
    truth,
):
    pred = pred.astype(
        np.float64
    )
    truth = truth.astype(
        np.float64
    )

    mae = np.mean(
        np.abs(
            pred - truth
        )
    )

    rmse = np.sqrt(
        np.mean(
            (pred - truth) ** 2
        )
    )

    pred_flat = pred.reshape(-1)
    truth_flat = truth.reshape(-1)

    if (
        np.std(pred_flat) > 0
        and np.std(truth_flat) > 0
    ):
        cc = np.corrcoef(
            pred_flat,
            truth_flat,
        )[0, 1]
    else:
        cc = np.nan

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "cc": float(cc),
    }


# ============================================================
# Plot
# ============================================================

def plot_sequence(
    past,
    truth,
    pred,
    path,
):
    fig, axes = plt.subplots(
        3,
        6,
        figsize=(18, 9),
    )

    vmax = max(
        float(truth.max()),
        float(pred.max()),
        float(past.max()),
        1.0,
    )

    for t in range(6):
        axes[0, t].imshow(
            past[t],
            cmap="jet",
            vmin=0,
            vmax=vmax,
        )

        axes[1, t].imshow(
            truth[t],
            cmap="jet",
            vmin=0,
            vmax=vmax,
        )

        axes[2, t].imshow(
            pred[t],
            cmap="jet",
            vmin=0,
            vmax=vmax,
        )

        axes[0, t].set_title(
            f"Past {t + 1}"
        )

        axes[1, t].set_title(
            f"GT {t + 1}"
        )

        axes[2, t].set_title(
            f"Pred {t + 1}"
        )

        for row in range(3):
            axes[row, t].axis(
                "off"
            )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
    )

    plt.close()


# ============================================================
# Pick rainy samples
# ============================================================

def find_rainy_samples(
    dataset,
    top_k=1,
    step=10,
):
    scores = []

    for idx in range(
        0,
        len(dataset),
        step,
    ):
        _, future = dataset[idx]

        # Dữ liệu hiện tại đã Min-Max [0,1].
        # Tổng future càng lớn => sample có nhiều mưa hơn.
        score = future.sum().item()

        scores.append(
            (score, idx)
        )

    scores.sort(
        reverse=True
    )

    selected = [
        idx
        for _, idx
        in scores[:top_k]
    ]

    print(
        "Rainy samples:",
        selected,
    )

    return selected


# ============================================================
# Main
# ============================================================

def main():
    Path(
        OUTPUT_DIR
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CONFIG_PATH,
        "r",
    ) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    denoiser = build_denoiser(
        cfg
    ).to(device)

    dcfg = cfg["diffusion"]

    diffusion = GaussianDiffusion(
        timesteps=dcfg["timesteps"],
    ).to(device)

    model = RadarDiffusionModel(
        denoiser=denoiser,
        diffusion=diffusion,
    ).to(device)

    checkpoint_path = (
        Path(
            cfg["output"]["dir"]
        )
        / CHECKPOINT_NAME
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.denoiser.load_state_dict(
        checkpoint["denoiser"]
    )

    # EMA đang tắt ở run này nên KHÔNG load/copy EMA.
    model.eval()

    print(
        "Loaded epoch:",
        checkpoint["epoch"],
    )

    print(
        "Train loss:",
        checkpoint.get(
            "train_loss"
        ),
    )

    print(
        "Val loss:",
        checkpoint.get(
            "val_loss"
        ),
    )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    d = cfg["data"]

    dataset = RadarDataset(
        root_dir=d["root"],
        split="test",

        input_frames=d[
            "input_frames"
        ],

        output_frames=d[
            "output_frames"
        ],

        image_size=d[
            "image_size"
        ],

        max_rainfall=d[
            "max_rainfall"
        ],
    )

    indices = find_rainy_samples(
        dataset,
        top_k=NUM_SAMPLES,
        step=SEARCH_STEP,
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    for idx in indices:
        print(
            "\nSample:",
            idx,
        )

        past, future = dataset[
            idx
        ]

        past_batch = (
            past
            .unsqueeze(0)
            .to(device)
        )

        print(
            "condition:",
            past_batch.shape,
        )

        pred = sample_ddpm(
            model,
            past_batch,
            seed=SEED + idx,
        )

        print(
            "prediction:",
            pred.shape,
        )

        # [1,6,80,80,1]
        # -> [6,80,80]
        pred_np = (
            pred[0, ..., 0]
            .float()
            .cpu()
            .numpy()
        )

        past_np = (
            past
            .float()
            .cpu()
            .numpy()
        )

        future_np = (
            future
            .float()
            .cpu()
            .numpy()
        )

        max_rain = d[
            "max_rainfall"
        ]

        past_mm = to_rainfall(
            past_np,
            max_rain,
        )

        future_mm = to_rainfall(
            future_np,
            max_rain,
        )

        pred_mm = to_rainfall(
            pred_np,
            max_rain,
        )

        metrics = sample_metrics(
            pred_mm,
            future_mm,
        )

        print(
            "MAE :",
            f"{metrics['mae']:.4f}",
        )

        print(
            "RMSE:",
            f"{metrics['rmse']:.4f}",
        )

        print(
            "CC  :",
            f"{metrics['cc']:.4f}",
        )

        output_path = (
            Path(OUTPUT_DIR)
            / f"sample_{idx}.png"
        )

        plot_sequence(
            past_mm,
            future_mm,
            pred_mm,
            output_path,
        )

        print(
            "Saved:",
            output_path,
        )

        # Giải phóng sample trước sample tiếp theo
        del past_batch
        del pred

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
