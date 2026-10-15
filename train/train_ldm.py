import os

# Must be set before CUDA is initialized.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import csv
import math
import random
from pathlib import Path

import numpy as np
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

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_cuda_memory(device, prefix=""):
    if (
        not torch.cuda.is_available()
        or device.type != "cuda"
    ):
        return

    allocated = (
        torch.cuda.memory_allocated(device)
        / (1024 ** 3)
    )
    reserved = (
        torch.cuda.memory_reserved(device)
        / (1024 ** 3)
    )
    max_allocated = (
        torch.cuda.max_memory_allocated(device)
        / (1024 ** 3)
    )

    print(
        f"{prefix} CUDA | "
        f"allocated={allocated:.2f} GB | "
        f"reserved={reserved:.2f} GB | "
        f"peak={max_allocated:.2f} GB"
    )


def optimizer_to(optimizer, device):
    """
    Move optimizer states to the same device as model parameters after
    loading a checkpoint on CPU.
    """
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


# ============================================================
# Build denoiser
# ============================================================

def build_denoiser(cfg):
    m = cfg["model"]

    num_blocks = len(m["depth"])
    patterns = [
        m["self_pattern"]
    ] * num_blocks

    return CuboidTransformerUNet(
        input_shape=m["input_shape"],
        target_shape=m["target_shape"],

        base_units=m["base_units"],
        scale_alpha=m["scale_alpha"],
        depth=m["depth"],

        downsample=m["downsample"],
        downsample_type=m["downsample_type"],

        upsample_type=m["upsample_type"],
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
# Dataset
# ============================================================

def make_dataset(cfg, split):
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
    )


# ============================================================
# Deterministic validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    loss_fn,
    ema,
    device,
    use_amp,
    seed=12345,
    max_batches=None,
):
    """
    Uses the same timestep/noise sequence every epoch.

    Besides val MSE/loss, compute a global Pearson CC. This is useful
    because an all-zero prediction can obtain a deceptively small MSE
    on sparse radar fields while having no spatial correlation.
    """
    model.eval()

    generator = torch.Generator(
        device=device
    )
    generator.manual_seed(seed)

    if ema is not None:
        ema.store(
            model.denoiser.parameters()
        )
        ema.copy_to(
            model.denoiser
        )

    clear_cuda()

    total_loss = 0.0
    count = 0

    # Running sums for global Pearson correlation.
    n = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_y2 = 0.0
    sum_xy = 0.0

    pred_sum = 0.0
    pred_count = 0
    pred_max = -float("inf")

    for batch_idx, (
        past,
        future,
    ) in enumerate(loader):
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

        future_cl = (
            model.to_channels_last(
                future
            )
        )

        batch_size = future.shape[0]

        t = torch.randint(
            0,
            model.diffusion.timesteps,
            (batch_size,),
            device=device,
            generator=generator,
        ).long()

        noise = torch.randn(
            future_cl.shape,
            device=device,
            dtype=future_cl.dtype,
            generator=generator,
        )

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):
            result = model(
                past,
                future,
                t=t,
                noise=noise,
            )

            val_loss = loss_fn(
                result["pred"],
                result["target"],
                past,
            )

        total_loss += (
            val_loss.detach().item()
        )
        count += 1

        pred = (
            result["pred"]
            .detach()
            .float()
        )
        target = (
            result["target"]
            .detach()
            .float()
        )

        flat_x = pred.reshape(-1)
        flat_y = target.reshape(-1)

        batch_n = flat_x.numel()

        n += batch_n
        sum_x += flat_x.sum().item()
        sum_y += flat_y.sum().item()
        sum_x2 += (
            flat_x.square()
            .sum()
            .item()
        )
        sum_y2 += (
            flat_y.square()
            .sum()
            .item()
        )
        sum_xy += (
            (flat_x * flat_y)
            .sum()
            .item()
        )

        pred_sum += (
            flat_x.sum().item()
        )
        pred_count += batch_n
        pred_max = max(
            pred_max,
            flat_x.max().item(),
        )

        del flat_x
        del flat_y
        del pred
        del target

        del result
        del val_loss
        del t
        del noise
        del future_cl
        del past
        del future

    if ema is not None:
        ema.restore(
            model.denoiser.parameters()
        )

    val_loss = (
        total_loss
        / max(count, 1)
    )

    if n > 0:
        numerator = (
            n * sum_xy
            - sum_x * sum_y
        )

        denom_x = (
            n * sum_x2
            - sum_x * sum_x
        )
        denom_y = (
            n * sum_y2
            - sum_y * sum_y
        )

        denominator = math.sqrt(
            max(
                denom_x * denom_y,
                0.0,
            )
        )

        if denominator > 1e-12:
            val_cc = (
                numerator
                / denominator
            )
        else:
            val_cc = float("nan")
    else:
        val_cc = float("nan")

    if pred_count > 0:
        val_pred_mean = (
            pred_sum
            / pred_count
        )
    else:
        val_pred_mean = float("nan")

    if pred_max == -float("inf"):
        val_pred_max = float("nan")
    else:
        val_pred_max = pred_max

    clear_cuda()
    model.train()

    return {
        "loss": val_loss,
        "cc": val_cc,
        "pred_mean": val_pred_mean,
        "pred_max": val_pred_max,
    }


# ============================================================
# Checkpoint
# ============================================================

def make_checkpoint(
    epoch,
    model,
    optimizer,
    ema,
    scaler,
    use_amp,
    train_loss,
    val_metrics,
    best_val,
    best_cc,
    cfg,
):
    return {
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
            val_metrics["loss"],

        "val_cc":
            val_metrics["cc"],

        "val_pred_mean":
            val_metrics["pred_mean"],

        "val_pred_max":
            val_metrics["pred_max"],

        "best_val":
            best_val,

        "best_cc":
            best_cc,

        "config":
            cfg,
    }


def append_history(
    output_dir,
    epoch,
    train_loss,
    val_metrics,
    lr,
):
    history_path = (
        output_dir
        / "history.csv"
    )

    exists = history_path.exists()

    with history_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "val_cc",
                "val_pred_mean",
                "val_pred_max",
                "lr",
            ])

        writer.writerow([
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["cc"],
            val_metrics["pred_mean"],
            val_metrics["pred_max"],
            lr,
        ])


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

    seed = train_cfg.get(
        "seed",
        12345,
    )

    seed_everything(seed)

    device = torch.device(
        train_cfg["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Seed:", seed)

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

    loader_generator = (
        torch.Generator()
    )
    loader_generator.manual_seed(
        seed
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
        drop_last=True,

        generator=loader_generator,
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

    diffusion = GaussianDiffusion(
        timesteps=dcfg[
            "timesteps"
        ],
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

    optimizer = torch.optim.AdamW(
        params,

        lr=train_cfg["lr"],
        weight_decay=train_cfg[
            "weight_decay"
        ],

        # Reduces peak VRAM on this large model.
        foreach=False,
    )

    # ========================================================
    # AMP
    # ========================================================

    use_amp = (
        train_cfg.get(
            "use_amp",
            True,
        )
        and device.type == "cuda"
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=use_amp
        )
    )

    print("AMP:", use_amp)

    # ========================================================
    # EMA
    # ========================================================

    use_ema = train_cfg.get(
        "use_ema",
        False,
    )

    ema = None

    if use_ema:
        ema = LitEma(
            model.denoiser,
            decay=train_cfg[
                "ema_decay"
            ],
        )

    print("EMA:", use_ema)

    # ========================================================
    # Resume
    # ========================================================

    start_epoch = 1
    best_val = float("inf")
    best_cc = -float("inf")

    resume_path = train_cfg.get(
        "resume"
    )

    if resume_path:
        # Load on CPU first so optimizer/checkpoint tensors do not
        # temporarily consume several extra GB of VRAM.
        checkpoint = torch.load(
            resume_path,
            map_location="cpu",
        )

        model.denoiser.load_state_dict(
            checkpoint["denoiser"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        # optimizer.load_state_dict() may restore the OLD learning rate.
        # The YAML value is authoritative for the resumed run.
        for param_group in (
            optimizer.param_groups
        ):
            param_group["lr"] = (
                train_cfg["lr"]
            )

        optimizer_to(
            optimizer,
            device,
        )

        if (
            ema is not None
            and checkpoint.get(
                "ema"
            ) is not None
        ):
            ema.load_state_dict(
                checkpoint["ema"]
            )

        if (
            use_amp
            and checkpoint.get(
                "scaler"
            ) is not None
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

        best_cc = checkpoint.get(
            "best_cc",
            -float("inf"),
        )

        resumed_epoch = (
            checkpoint["epoch"]
        )

        del checkpoint
        clear_cuda()

        print(
            f"Resume from epoch "
            f"{resumed_epoch}"
        )

        print(
            "Resume LR:",
            optimizer.param_groups[0][
                "lr"
            ],
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

    max_train_batches = (
        train_cfg.get(
            "max_train_batches"
        )
    )

    max_val_batches = (
        train_cfg.get(
            "max_val_batches"
        )
    )

    clear_cuda()
    print_cuda_memory(
        device,
        "Before training",
    )

    # ========================================================
    # Train
    # ========================================================

    for epoch in range(
        start_epoch,
        train_cfg["epochs"] + 1,
    ):
        model.train()

        if (
            torch.cuda.is_available()
            and device.type == "cuda"
        ):
            torch.cuda.reset_peak_memory_stats(
                device
            )

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

        for batch_idx, (
            past,
            future,
        ) in enumerate(pbar):
            if (
                max_train_batches
                is not None
                and batch_idx
                >= max_train_batches
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

                if not torch.isfinite(
                    raw_loss
                ):
                    raise RuntimeError(
                        "Non-finite training "
                        f"loss at epoch={epoch}, "
                        f"batch={batch_idx}: "
                        f"{raw_loss.item()}"
                    )

                loss = (
                    raw_loss
                    / accumulation
                )

            scaler.scale(
                loss
            ).backward()

            total_loss += (
                raw_loss
                .detach()
                .item()
            )

            count += 1
            num_batches_since_step += 1

            if (
                num_batches_since_step
                == accumulation
            ):
                scaler.unscale_(
                    optimizer
                )

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        params,
                        train_cfg[
                            "gradient_clip"
                        ],
                    )
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
                loss=(
                    f"{raw_loss.item():.6f}"
                )
            )

            del result
            del raw_loss
            del loss
            del past
            del future

        # ----------------------------------------------------
        # Remaining accumulated gradients
        # ----------------------------------------------------

        if num_batches_since_step > 0:
            # Correct scaling for a partial final accumulation window.
            correction = (
                accumulation
                / num_batches_since_step
            )

            for param in params:
                if param.grad is not None:
                    param.grad.mul_(
                        correction
                    )

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
            device,
            f"Epoch {epoch} after train",
        )

        # ====================================================
        # Emergency checkpoint before validation
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

            "best_cc":
                best_cc,

            "config":
                cfg,
        }

        torch.save(
            emergency_checkpoint,
            output_dir
            / "after_train.pt",
        )

        del emergency_checkpoint
        clear_cuda()

        # ====================================================
        # Validation
        # ====================================================

        val_metrics = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            ema=ema,
            device=device,
            use_amp=use_amp,
            seed=seed,
            max_batches=max_val_batches,
        )

        val_loss = val_metrics[
            "loss"
        ]
        val_cc = val_metrics[
            "cc"
        ]

        print_cuda_memory(
            device,
            f"Epoch {epoch} after val",
        )

        print(
            f"\nEpoch {epoch:03d} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"val_cc={val_cc:.4f} | "
            f"pred_mean="
            f"{val_metrics['pred_mean']:.6f} | "
            f"pred_max="
            f"{val_metrics['pred_max']:.6f}"
        )

        append_history(
            output_dir=output_dir,
            epoch=epoch,
            train_loss=train_loss,
            val_metrics=val_metrics,
            lr=optimizer.param_groups[
                0
            ]["lr"],
        )

        current_best_val = min(
            best_val,
            val_loss,
        )

        current_best_cc = best_cc

        if (
            math.isfinite(val_cc)
            and val_cc > best_cc
        ):
            current_best_cc = val_cc

        checkpoint = make_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            ema=ema,
            scaler=scaler,
            use_amp=use_amp,
            train_loss=train_loss,
            val_metrics=val_metrics,
            best_val=current_best_val,
            best_cc=current_best_cc,
            cfg=cfg,
        )

        # Always save the latest full training state.
        torch.save(
            checkpoint,
            output_dir / "last.pt",
        )

        # Legacy/standard best checkpoint by validation loss.
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
                "Best val loss:",
                best_val,
            )

        # Additional checkpoint selected by structural correlation.
        # This prevents an all-zero field from looking "best" only
        # because radar data are sparse.
        if (
            math.isfinite(val_cc)
            and val_cc > best_cc
        ):
            best_cc = val_cc

            checkpoint[
                "best_cc"
            ] = best_cc

            torch.save(
                checkpoint,
                output_dir / "best_cc.pt",
            )

            print(
                "Best val CC:",
                best_cc,
            )

        del checkpoint
        clear_cuda()


if __name__ == "__main__":
    main()
