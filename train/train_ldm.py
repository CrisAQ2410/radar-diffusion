import os
from pathlib import Path

import torch
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.radar_dataset import RadarDataset

from models.VAE.autoencoder_kl import AutoencoderKL

from models.LDM.cuboid_transformer_unet import (
    CuboidTransformerUNet
)

from models.ldm import RadarLatentDiffusion

from utils.ema import LitEma


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

    state = torch.load(
        cfg["vae"]["checkpoint"],
        map_location="cpu"
    )

    vae.load_state_dict(state)

    return vae


def build_denoiser(cfg):

    m = cfg["model"]

    num_blocks = len(
        m["depth"]
    )

    patterns = [
        m["self_pattern"]
    ] * num_blocks

    return CuboidTransformerUNet(

        input_shape=m[
            "input_shape"
        ],

        target_shape=m[
            "target_shape"
        ],

        base_units=m[
            "base_units"
        ],

        scale_alpha=m[
            "scale_alpha"
        ],

        depth=m["depth"],

        downsample=m[
            "downsample"
        ],

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

        num_heads=m[
            "num_heads"
        ],

        attn_drop=m[
            "attn_drop"
        ],

        proj_drop=m[
            "proj_drop"
        ],

        ffn_drop=m[
            "ffn_drop"
        ],

        ffn_activation=m[
            "ffn_activation"
        ],

        gated_ffn=m[
            "gated_ffn"
        ],

        norm_layer=m[
            "norm_layer"
        ],

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


def make_dataset(
    cfg,
    split
):

    d = cfg["data"]

    return RadarDataset(
        root_dir=d["root"],

        split=split,

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


@torch.no_grad()
def validate(
    model,
    loader,
    ema,
    device,
    max_batches=None
):

    model.eval()

    ema.store(
        model.denoiser.parameters()
    )

    ema.copy_to(
        model.denoiser
    )

    total = 0.0
    count = 0

    for batch_idx, (
        past,
        future
    ) in enumerate(loader):

        if (
            max_batches is not None
            and
            batch_idx >= max_batches
        ):
            break

        past = past.to(
            device,
            non_blocking=True
        )

        future = future.to(
            device,
            non_blocking=True
        )

        result = model(
            past,
            future
        )

        total += result[
            "loss"
        ].item()

        count += 1


    ema.restore(
        model.denoiser.parameters()
    )

    model.train()

    return total / max(
        count,
        1
    )


def main():

    with open(
        "configs/radar_ldm.yaml",
        "r"
    ) as f:

        cfg = yaml.safe_load(f)

    train_cfg = cfg[
        "training"
    ]

    device = torch.device(
        train_cfg["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # -----------------------------
    # Dataset
    # -----------------------------

    train_dataset = make_dataset(
        cfg,
        "train"
    )

    val_dataset = make_dataset(
        cfg,
        "val"
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=train_cfg[
            "batch_size"
        ],

        shuffle=True,

        num_workers=train_cfg[
            "num_workers"
        ],

        pin_memory=True,

        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=train_cfg[
            "batch_size"
        ],

        shuffle=False,

        num_workers=train_cfg[
            "num_workers"
        ],

        pin_memory=True
    )

    # -----------------------------
    # Models
    # -----------------------------

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

    # -----------------------------
    # optimizer
    # -----------------------------

    params = list(
        model.denoiser.parameters()
    )

    if model.learn_logvar:
        params.append(
            model.logvar
        )

    optimizer = torch.optim.AdamW(

        params,

        lr=train_cfg[
            "lr"
        ],

        weight_decay=train_cfg[
            "weight_decay"
        ],
    )

    ema = LitEma(
        model.denoiser,
        decay=train_cfg[
            "ema_decay"
        ]
    )

    start_epoch = 1

    resume_path = train_cfg.get("resume")

    if resume_path:

        checkpoint = torch.load(
            resume_path,
            map_location=device
        )

        model.denoiser.load_state_dict(
            checkpoint["denoiser"]
        )

        model.logvar.data.copy_(
            checkpoint["logvar"].to(device)
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        ema.load_state_dict(
            checkpoint["ema"]
        )

        start_epoch = checkpoint["epoch"] + 1

        print(
            f"Resume from epoch "
            f"{checkpoint['epoch']}"
        )

    output_dir = Path(
        cfg["output"]["dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    best_val = float(
        "inf"
    )

    accumulation = train_cfg[
        "accumulation_steps"
    ]

    max_train_batches = train_cfg.get(
        "max_train_batches"
    )

    max_val_batches = train_cfg.get(
        "max_val_batches"
    )

    # -----------------------------
    # Train
    # -----------------------------

    for epoch in range(
        start_epoch,
        train_cfg["epochs"] + 1
    ):

        model.train()

        # VAE luôn frozen + eval
        model.vae.eval()

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss = 0.0
        count = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}"
        )

        for batch_idx, (
            past,
            future
        ) in enumerate(pbar):

            if (
                max_train_batches
                is not None
                and
                batch_idx
                >= max_train_batches
            ):
                break

            past = past.to(
                device,
                non_blocking=True
            )

            future = future.to(
                device,
                non_blocking=True
            )

            result = model(
                past,
                future
            )

            loss = (
                result["loss"]
                /
                accumulation
            )

            loss.backward()

            if (
                (batch_idx + 1)
                %
                accumulation
                == 0
            ):

                torch.nn.utils.clip_grad_norm_(
                    params,
                    train_cfg[
                        "gradient_clip"
                    ]
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                ema(
                    model.denoiser
                )

            total_loss += (
                result["loss"].item()
            )

            count += 1

            pbar.set_postfix(
                loss=(
                    f"{result['loss'].item():.5f}"
                ),
                simple=(
                    f"{result['loss_simple'].item():.5f}"
                )
            )

        if count % accumulation != 0:
            torch.nn.utils.clip_grad_norm_(
                params,
                train_cfg["gradient_clip"]
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            ema(
                model.denoiser
            )

        train_loss = (
            total_loss
            /
            max(count, 1)
        )

        torch.save(
            {
                "epoch": epoch,
                "denoiser": model.denoiser.state_dict(),
                "logvar": model.logvar.detach().cpu(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "train_loss": train_loss,
                "config": cfg,
            },
            output_dir / "after_train.pt"
        )

        val_loss = validate(
            model,
            val_loader,
            ema,
            device,
            max_val_batches
        )

        print(
            f"\nEpoch {epoch:03d} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f}"
        )

        checkpoint = {
            "epoch": epoch,

            "denoiser":
                model.denoiser.state_dict(),

            "logvar":
                model.logvar.detach().cpu(),

            "optimizer":
                optimizer.state_dict(),

            "ema":
                ema.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "config":
                cfg,
        }

        torch.save(
            checkpoint,
            output_dir / "last.pt"
        )

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                checkpoint,
                output_dir / "best.pt"
            )

            print(
                "Best LDM:",
                best_val
            )


if __name__ == "__main__":
    main()