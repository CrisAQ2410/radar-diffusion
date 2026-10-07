import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.vae_dataset import RadarVAEDataset
from models.VAE.autoencoder_kl import AutoencoderKL


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg):
    m = cfg["model"]

    return AutoencoderKL(
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        down_block_types=tuple(m["down_block_types"]),
        up_block_types=tuple(m["up_block_types"]),
        block_out_channels=tuple(m["block_out_channels"]),
        layers_per_block=m["layers_per_block"],
        latent_channels=m["latent_channels"],
        norm_num_groups=m["norm_num_groups"],
        sample_size=cfg["data"]["image_size"],
    )


def run_epoch(
    model,
    loader,
    device,
    kl_weight,
    optimizer=None,
    scaler=None,
    use_amp=False,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_rec = 0.0
    total_kl = 0.0
    total_samples = 0

    pbar = tqdm(
        loader,
        desc="Train" if training else "Val",
        leave=False,
    )

    for x in pbar:
        x = x.to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):
                recon, posterior = model(
                    sample=x,
                    sample_posterior=True,
                    return_posterior=True,
                )

                # pixel có mưa được quan tâm nhiều hơn
                rain_mask = (x > 0.01).float()

                weight = 1.0 + 10.0 * rain_mask

                weighted_mse = (
                    weight * (recon - x) ** 2
                ).mean()

                l1 = F.l1_loss(
                    recon,
                    x
                )

                rec_loss = (
                    weighted_mse
                    +
                    0.5 * l1
                )

                kl_loss = (
                    posterior
                    .kl()
                    .mean()
                )

                loss = (
                    rec_loss
                    +
                    kl_weight * kl_loss
                )

            if training:
                if use_amp:
                    scaler.scale(
                        loss
                    ).backward()

                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                    )

                    optimizer.step()

        bs = x.size(0)

        total_loss += (
            loss.item() * bs
        )

        total_rec += (
            rec_loss.item() * bs
        )

        total_kl += (
            kl_loss.item() * bs
        )

        total_samples += bs

        pbar.set_postfix(
            loss=f"{loss.item():.6f}",
            rec=f"{rec_loss.item():.6f}",
            kl=f"{kl_loss.item():.2f}",
        )

    n = max(total_samples, 1)

    return {
        "loss": total_loss / n,
        "rec": total_rec / n,
        "kl": total_kl / n,
    }


def main():
    with open(
        "configs/radar_vae.yaml",
        "r",
    ) as f:
        cfg = yaml.safe_load(f)

    set_seed(42)

    tcfg = cfg["training"]
    dcfg = cfg["data"]

    device = torch.device(
        tcfg["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(
        cfg["output"]["dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset = RadarVAEDataset(
        root_dir=dcfg["root"],
        split="train",
        image_size=dcfg["image_size"],
        max_rainfall=dcfg["max_rainfall"],
        train_ratio=dcfg["train_ratio"],
        test_year=dcfg["test_year"],
    )

    val_dataset = RadarVAEDataset(
        root_dir=dcfg["root"],
        split="val",
        image_size=dcfg["image_size"],
        max_rainfall=dcfg["max_rainfall"],
        train_ratio=dcfg["train_ratio"],
        test_year=dcfg["test_year"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=tcfg["num_workers"],
        pin_memory=True,
    )

    model = build_model(
        cfg
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
    )

    use_amp = bool(
        tcfg["use_amp"]
        and device.type == "cuda"
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    best_val = float("inf")

    print("Device:", device)
    print(
        "Train:",
        len(train_dataset),
        "Val:",
        len(val_dataset),
    )

    for epoch in range(
        1,
        tcfg["epochs"] + 1,
    ):
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            kl_weight=tcfg["kl_weight"],
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )

        with torch.no_grad():
            val_result = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                kl_weight=tcfg["kl_weight"],
                optimizer=None,
                scaler=None,
                use_amp=use_amp,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_result['loss']:.6f} | "
            f"val={val_result['loss']:.6f} | "
            f"rec={val_result['rec']:.6f} | "
            f"kl={val_result['kl']:.2f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "val_loss":
                val_result["loss"],
            "config": cfg,
        }

        torch.save(
            checkpoint,
            output_dir / "last.pt",
        )

        if (
            val_result["loss"]
            < best_val
        ):
            best_val = (
                val_result["loss"]
            )

            # Đây là format mà LDM load tiện nhất
            torch.save(
                model.state_dict(),
                output_dir / "vae.pt",
            )

            torch.save(
                checkpoint,
                output_dir / "best_checkpoint.pt",
            )

            print(
                f"  Best VAE: "
                f"{best_val:.6f}"
            )


if __name__ == "__main__":
    main()