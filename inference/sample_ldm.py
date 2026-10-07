import os
import math

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from datasets.radar_dataset import RadarDataset

from models.VAE.autoencoder_kl import AutoencoderKL
from models.LDM.cuboid_transformer_unet import CuboidTransformerUNet
from models.ldm import RadarLatentDiffusion

from utils.ema import LitEma


DEVICE = "cuda:2"

CONFIG_PATH = "configs/radar_ldm.yaml"
CHECKPOINT_PATH = "checkpoints/ldm_mode/best.pt"

OUTPUT_DIR = "outputs/ldm_samples"


# ============================================================
# Build VAE
# ============================================================

def build_vae(cfg):

    with open(
        cfg["vae"]["config"],
        "r"
    ) as f:

        vae_cfg = yaml.safe_load(f)

    m = vae_cfg["model"]

    vae = AutoencoderKL(

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

    vae.load_state_dict(
        torch.load(
            cfg["vae"]["checkpoint"],
            map_location="cpu"
        )
    )

    vae.eval()

    for p in vae.parameters():
        p.requires_grad = False

    return vae


# ============================================================
# Build CuboidTransformer
# ============================================================

def build_denoiser(cfg):

    m = cfg["model"]

    patterns = [
        m["self_pattern"]
    ] * len(m["depth"])

    model = CuboidTransformerUNet(

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

    return model


# ============================================================
# Decode latent sequence
# ============================================================

@torch.no_grad()
def decode_sequence(
    vae,
    z
):
    """
    z:
    [B,T,H,W,C]

    output:
    [B,T,256,256]
    """

    B, T, H, W, C = z.shape

    z = z.permute(
        0,
        1,
        4,
        2,
        3
    ).contiguous()

    z = z.reshape(
        B * T,
        C,
        H,
        W
    )

    x = vae.decode(z)

    x = x.reshape(
        B,
        T,
        x.shape[-2],
        x.shape[-1]
    )

    return x


# ============================================================
# Reverse DDPM
# ============================================================

@torch.no_grad()
def sample_ddpm(
    model,
    z_past
):

    schedule = model.diffusion

    B = z_past.shape[0]

    shape = z_past.shape

    # x_T
    z = torch.randn(
        shape,
        device=z_past.device
    )

    for i in reversed(
        range(model.timesteps)
    ):

        t = torch.full(
            (B,),
            i,
            device=z.device,
            dtype=torch.long
        )

        # x0 prediction
        z0_pred = model.denoiser(
            z,
            t,
            z_past
        )

        coef1 = schedule.posterior_mean_coef1[
            t
        ].reshape(
            B, 1, 1, 1, 1
        )

        coef2 = schedule.posterior_mean_coef2[
            t
        ].reshape(
            B, 1, 1, 1, 1
        )

        model_mean = (
            coef1 * z0_pred
            +
            coef2 * z
        )

        if i > 0:

            log_variance = (
                schedule
                .posterior_log_variance_clipped[
                    t
                ]
                .reshape(
                    B, 1, 1, 1, 1
                )
            )

            noise = torch.randn_like(
                z
            )

            z = (
                model_mean
                +
                torch.exp(
                    0.5
                    * log_variance
                )
                * noise
            )

        else:

            z = model_mean

        if (
            i % 100 == 0
            or i == 999
        ):
            print(
                f"Sampling timestep {i}"
            )

    return z


# ============================================================
# Inverse normalization
# ============================================================

def to_rainfall(
    x,
    max_rainfall
):

    x = np.clip(
        x,
        0.0,
        1.0
    )

    return np.expm1(
        x
        * np.log1p(
            max_rainfall
        )
    )


# ============================================================
# Plot
# ============================================================

def plot_sequence(
    past,
    truth,
    pred,
    path
):

    fig, axes = plt.subplots(
        3,
        6,
        figsize=(18, 9)
    )

    vmax = max(
        truth.max(),
        pred.max(),
        past.max(),
        1.0
    )

    for t in range(6):

        axes[0, t].imshow(
            past[t],
            cmap="jet",
            vmin=0,
            vmax=vmax
        )

        axes[1, t].imshow(
            truth[t],
            cmap="jet",
            vmin=0,
            vmax=vmax
        )

        axes[2, t].imshow(
            pred[t],
            cmap="jet",
            vmin=0,
            vmax=vmax
        )

        axes[0, t].set_title(
            f"Past {t+1}"
        )

        axes[1, t].set_title(
            f"GT {t+1}"
        )

        axes[2, t].set_title(
            f"Pred {t+1}"
        )

        for row in range(3):
            axes[row, t].axis(
                "off"
            )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        CONFIG_PATH,
        "r"
    ) as f:

        cfg = yaml.safe_load(f)

    device = torch.device(
        DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # --------------------------
    # Build models
    # --------------------------

    vae = build_vae(
        cfg
    ).to(device)

    denoiser = build_denoiser(
        cfg
    ).to(device)

    dcfg = cfg[
        "diffusion"
    ]

    model = RadarLatentDiffusion(

        vae=vae,

        denoiser=denoiser,

        timesteps=dcfg[
            "timesteps"
        ],

        cosine_s=dcfg[
            "cosine_s"
        ],

        learn_logvar=dcfg[
            "learn_logvar"
        ],

        logvar_init=dcfg[
            "logvar_init"
        ],

    ).to(device)

    # --------------------------
    # Load checkpoint
    # --------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.denoiser.load_state_dict(
        checkpoint["denoiser"]
    )

    if "logvar" in checkpoint:

        model.logvar.data.copy_(
            checkpoint["logvar"]
            .to(device)
        )

    # --------------------------
    # EMA
    # --------------------------

    ema = LitEma(
        model.denoiser,
        decay=cfg[
            "training"
        ]["ema_decay"]
    ).to(device)

    ema.load_state_dict(
        checkpoint["ema"]
    )

    # dùng EMA weights để inference
    ema.copy_to(
        model.denoiser
    )

    model.eval()

    print(
        "Loaded epoch:",
        checkpoint["epoch"]
    )

    print(
        "Val loss:",
        checkpoint["val_loss"]
    )

    # --------------------------
    # Test dataset
    # --------------------------

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

        val_ratio=d[
            "val_ratio"
        ],

        test_year=d[
            "test_year"
        ],
    )

    # vài sample trước
    def find_rainy_samples(
        dataset,
        max_rainfall,
        top_k=3,
        step=10,
    ):
        scores = []

        for idx in range(
            0,
            len(dataset),
            step
        ):
            _, future = dataset[idx]

            # future đang ở log-normalized [0,1]
            # score càng lớn -> lượng mưa tổng càng nhiều
            score = future.sum().item()

            scores.append(
                (score, idx)
            )

        scores.sort(
            reverse=True
        )

        selected = [
            idx
            for score, idx
            in scores[:top_k]
        ]

        print(
            "Rainy samples:",
            selected
        )

        return selected

    indices = find_rainy_samples(
        dataset,
        d["max_rainfall"],
        top_k=3,
        step=10
    )

    for idx in indices:

        print(
            "\nSample:",
            idx
        )

        past, future = dataset[
            idx
        ]

        past_batch = (
            past
            .unsqueeze(0)
            .to(device)
        )

        # ----------------------
        # Encode condition
        # ----------------------

        with torch.no_grad():

            z_past = (
                model
                .encode_sequence(
                    past_batch,
                    sample_posterior=False
                )
            )

        print(
            "condition:",
            z_past.shape
        )

        # ----------------------
        # reverse diffusion
        # ----------------------

        z_pred = sample_ddpm(
            model,
            z_past
        )

        print(
            "pred latent:",
            z_pred.shape
        )

        # ----------------------
        # VAE decode
        # ----------------------

        pred = decode_sequence(
            vae,
            z_pred
        )

        pred = (
            pred[0]
            .cpu()
            .numpy()
        )

        past_np = (
            past
            .cpu()
            .numpy()
        )

        future_np = (
            future
            .cpu()
            .numpy()
        )

        # ----------------------
        # inverse log norm
        # ----------------------

        max_rain = d[
            "max_rainfall"
        ]

        past_mm = to_rainfall(
            past_np,
            max_rain
        )

        future_mm = to_rainfall(
            future_np,
            max_rain
        )

        pred_mm = to_rainfall(
            pred,
            max_rain
        )

        # ----------------------
        # save
        # ----------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            f"sample_{idx}.png"
        )

        plot_sequence(
            past_mm,
            future_mm,
            pred_mm,
            output_path
        )

        print(
            "Saved:",
            output_path
        )


if __name__ == "__main__":
    main()