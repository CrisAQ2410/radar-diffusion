import os

# Phải set trước khi CUDA được khởi tạo
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

from pathlib import Path

import torch
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.radar_dataset import RadarDataset
from losses.factory import build_loss
from models.diffusion import GaussianDiffusion
from models.LDM.cuboid_transformer_unet import CuboidTransformerUNet
from models.ldm import RadarDiffusionModel
from utils.ema import LitEma


# ============================================================
# Helpers
# ============================================================

def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_cuda_memory(prefix=""):
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print(
        f"{prefix} CUDA | "
        f"allocated={allocated:.2f} GB | "
        f"reserved={reserved:.2f} GB | "
        f"peak={max_allocated:.2f} GB"
    )


# ============================================================
# Build denoiser
# ============================================================

def build_denoiser(cfg):
    m = cfg["model"]

    num_blocks = len(m["depth"])
    patterns = [m["self_pattern"]] * num_blocks

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
# Dataset
# ============================================================

def make_dataset(cfg, split):
    d = cfg["data"]

    return RadarDataset(
        root_dir=d["root"],
        split=split,

        input_frames=d["input_frames"],
        output_frames=d["output_frames"],

        image_size=d["image_size"],
        max_rainfall=d["max_rainfall"],
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    loss_fn,
    ema,
    device,
    max_batches=None,
):
    model.eval()

    # Nếu dùng EMA thì validation bằng EMA weights
    if ema is not None:
        ema.store(
            model.denoiser.parameters()
        )
        ema.copy_to(
            model.denoiser
        )

    clear_cuda()

    total = 0.0
    count = 0

    for batch_idx, (past, future) in enumerate(loader):
        if (
            max_batches is not None
            and batch_idx >= max_batches
        ):
            break

        past = past.to(
            device,
            non_blocking=True,
        )

        future = future.to(
            device,
            non_blocking=True,
        )

        # Validation cũng dùng AMP để giảm VRAM
        with torch.cuda.amp.autocast(
            enabled=(device.type == "cuda")
        ):
            result = model(
                past,
                future,
            )

            val_loss = loss_fn(
                result["pred"],
                result["target"],
                past,
            )

        total += val_loss.item()
        count += 1

        # Xóa reference càng sớm càng tốt
        del result
        del val_loss
        del past
        del future

    if ema is not None:
        ema.restore(
            model.denoiser.parameters()
        )

    clear_cuda()
    model.train()

    return total / max(count, 1)


# ============================================================
# Main
# ============================================================

def main():
    with open(
        "configs/radar_ldm.yaml",
        "r",
    ) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]

    device = torch.device(
        train_cfg["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ========================================================
    # Dataset
    # ========================================================

    train_dataset = make_dataset(
        cfg,
        "train",
    )

    val_dataset = make_dataset(
        cfg,
        "val",
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=train_cfg["batch_size"],
        shuffle=True,

        num_workers=train_cfg["num_workers"],

        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=train_cfg["batch_size"],
        shuffle=False,

        num_workers=train_cfg["num_workers"],

        pin_memory=True,
        drop_last=False,
    )

    # ========================================================
    # Model
    # ========================================================

    denoiser = build_denoiser(
        cfg
    ).to(device)

    dcfg = cfg["diffusion"]

    # Nếu GaussianDiffusion của bạn chỉ nhận timesteps
    # thì đoạn này đúng với test_image_diffusion.py đã chạy trước đó.
    diffusion = GaussianDiffusion(
        timesteps=dcfg["timesteps"],
    ).to(device)

    model = RadarDiffusionModel(
        denoiser=denoiser,
        diffusion=diffusion,
    ).to(device)

    # ========================================================
    # Loss
    # ========================================================

    loss_fn = build_loss(
        cfg["loss"]
    )

    print(
        "Loss:",
        cfg["loss"]["type"],
    )

    # ========================================================
    # Optimizer
    # ========================================================

    params = list(
        model.denoiser.parameters()
    )

    # foreach=False giảm peak VRAM của AdamW trên model lớn.
    optimizer = torch.optim.AdamW(
        params,

        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],

        foreach=False,
    )

    # ========================================================
    # AMP
    # ========================================================

    use_amp = (
        train_cfg.get("use_amp", True)
        and device.type == "cuda"
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    print("AMP:", use_amp)

    # ========================================================
    # EMA - mặc định OFF để tiết kiệm VRAM
    # ========================================================

    use_ema = train_cfg.get(
        "use_ema",
        False,
    )

    ema = None

    if use_ema:
        ema = LitEma(
            model.denoiser,
            decay=train_cfg["ema_decay"],
        )

    print("EMA:", use_ema)

    # ========================================================
    # Resume
    # ========================================================

    start_epoch = 1
    best_val = float("inf")

    resume_path = train_cfg.get(
        "resume"
    )

    if resume_path:
        checkpoint = torch.load(
            resume_path,
            map_location=device,
        )

        model.denoiser.load_state_dict(
            checkpoint["denoiser"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        if (
            ema is not None
            and checkpoint.get("ema") is not None
        ):
            ema.load_state_dict(
                checkpoint["ema"]
            )

        if (
            use_amp
            and checkpoint.get("scaler") is not None
        ):
            scaler.load_state_dict(
                checkpoint["scaler"]
            )

        start_epoch = (
            checkpoint["epoch"]
            + 1
        )

        best_val = checkpoint.get(
            "best_val",
            checkpoint.get(
                "val_loss",
                float("inf"),
            ),
        )

        print(
            f"Resume from epoch "
            f"{checkpoint['epoch']}"
        )

    # ========================================================
    # Output
    # ========================================================

    output_dir = Path(
        cfg["output"]["dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    clear_cuda()
    print_cuda_memory(
        "Before training"
    )

    # ========================================================
    # Train
    # ========================================================

    for epoch in range(
        start_epoch,
        train_cfg["epochs"] + 1,
    ):
        model.train()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss = 0.0
        count = 0
        num_batches_since_step = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
        )

        for batch_idx, (past, future) in enumerate(pbar):
            if (
                max_train_batches is not None
                and batch_idx >= max_train_batches
            ):
                break

            past = past.to(
                device,
                non_blocking=True,
            )

            future = future.to(
                device,
                non_blocking=True,
            )

            # ------------------------------------------------
            # Forward + loss in mixed precision
            # ------------------------------------------------

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):
                result = model(
                    past,
                    future,
                )

                raw_loss = loss_fn(
                    result["pred"],
                    result["target"],
                    past,
                )

                loss = (
                    raw_loss
                    / accumulation
                )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            scaler.scale(
                loss
            ).backward()

            total_loss += (
                raw_loss.detach().item()
            )

            count += 1
            num_batches_since_step += 1

            # ------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------

            if (
                num_batches_since_step
                == accumulation
            ):
                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    params,
                    train_cfg[
                        "gradient_clip"
                    ],
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

                if ema is not None:
                    ema(
                        model.denoiser
                    )

                num_batches_since_step = 0

            pbar.set_postfix(
                loss=f"{raw_loss.item():.6f}",
            )

            # Xóa graph/reference của batch hiện tại
            del result
            del raw_loss
            del loss
            del past
            del future

        # ----------------------------------------------------
        # Gradient accumulation còn dư
        # ----------------------------------------------------

        if num_batches_since_step > 0:
            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                params,
                train_cfg[
                    "gradient_clip"
                ],
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

            if ema is not None:
                ema(
                    model.denoiser
                )

        train_loss = (
            total_loss
            / max(count, 1)
        )

        clear_cuda()
        print_cuda_memory(
            f"Epoch {epoch} after train"
        )

        # ====================================================
        # Emergency checkpoint sau training phase
        # ====================================================

        emergency_checkpoint = {
            "epoch": epoch,

            "denoiser":
                model.denoiser.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "ema":
                (
                    ema.state_dict()
                    if ema is not None
                    else None
                ),

            "scaler":
                (
                    scaler.state_dict()
                    if use_amp
                    else None
                ),

            "train_loss":
                train_loss,

            "best_val":
                best_val,

            "config":
                cfg,
        }

        torch.save(
            emergency_checkpoint,
            output_dir
            / "after_train.pt",
        )

        # Sau torch.save() bỏ reference checkpoint tạm
        del emergency_checkpoint
        clear_cuda()

        # ====================================================
        # Validation
        # ====================================================

        val_loss = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            ema=ema,
            device=device,
            max_batches=max_val_batches,
        )

        print_cuda_memory(
            f"Epoch {epoch} after val"
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

            "optimizer":
                optimizer.state_dict(),

            "ema":
                (
                    ema.state_dict()
                    if ema is not None
                    else None
                ),

            "scaler":
                (
                    scaler.state_dict()
                    if use_amp
                    else None
                ),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "best_val":
                min(
                    best_val,
                    val_loss,
                ),

            "config":
                cfg,
        }

        torch.save(
            checkpoint,
            output_dir / "last.pt",
        )

        if val_loss < best_val:
            best_val = val_loss

            checkpoint[
                "best_val"
            ] = best_val

            torch.save(
                checkpoint,
                output_dir / "best.pt",
            )

            print(
                "Best image diffusion:",
                best_val,
            )

        del checkpoint
        clear_cuda()


if __name__ == "__main__":
    main()
