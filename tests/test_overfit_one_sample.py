from pathlib import Path

import numpy as np
import torch
import yaml

from datasets.radar_dataset import RadarDataset
from models.diffusion import GaussianDiffusion
from models.LDM.cuboid_transformer_unet import CuboidTransformerUNet
from models.ldm import RadarDiffusionModel


CONFIG_PATH = "configs/radar_ldm.yaml"
CHECKPOINT_NAME = "last.pt"
DEVICE = "cuda:2"

STEPS = 100
LR = 1e-3
T_VALUE = 0
SEED = 12345


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


def find_rainy_sample(dataset, step=10):
    scores = []

    for idx in range(0, len(dataset), step):
        _, future = dataset[idx]
        scores.append((future.sum().item(), idx))

    scores.sort(reverse=True)
    return scores[0][1]


def cc(pred, target):
    a = pred.detach().float().cpu().numpy().reshape(-1)
    b = target.detach().float().cpu().numpy().reshape(-1)

    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def evaluate(model, past, future, t, noise):
    model.eval()

    future_cl = model.to_channels_last(future)

    x_t = model.diffusion.q_sample(
        x_start=future_cl,
        t=t,
        noise=noise,
    )

    with torch.cuda.amp.autocast(
        enabled=(future.device.type == "cuda")
    ):
        pred = model.denoiser(
            x_t,
            t,
            model.to_channels_last(past),
        )

    pred = pred.float()
    mse = torch.mean((pred - future_cl) ** 2).item()
    corr = cc(pred, future_cl)

    return mse, corr, pred.mean().item(), pred.max().item()


def main():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        DEVICE if torch.cuda.is_available() else "cpu"
    )

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    dataset = RadarDataset(
        root_dir=cfg["data"]["root"],
        split="train",
        input_frames=cfg["data"]["input_frames"],
        output_frames=cfg["data"]["output_frames"],
        image_size=cfg["data"]["image_size"],
        max_rainfall=cfg["data"]["max_rainfall"],
    )

    idx = find_rainy_sample(dataset, step=10)
    print("Train sample:", idx)

    past, future = dataset[idx]

    past = past.unsqueeze(0).to(device)
    future = future.unsqueeze(0).to(device)

    denoiser = build_denoiser(cfg).to(device)
    diffusion = GaussianDiffusion(
        timesteps=cfg["diffusion"]["timesteps"]
    ).to(device)

    model = RadarDiffusionModel(
        denoiser=denoiser,
        diffusion=diffusion,
    ).to(device)

    checkpoint_path = (
        Path(cfg["output"]["dir"])
        / CHECKPOINT_NAME
    )

    # IMPORTANT:
    # Load checkpoint on CPU. The checkpoint also contains AdamW optimizer
    # states; loading the whole file directly to CUDA wastes several GB VRAM.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.denoiser.load_state_dict(
        checkpoint["denoiser"]
    )

    loaded_epoch = checkpoint["epoch"]
    print("Loaded epoch:", loaded_epoch)

    # We only need denoiser weights for this diagnostic.
    del checkpoint

    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t = torch.tensor(
        [T_VALUE],
        device=device,
        dtype=torch.long,
    )

    future_cl = model.to_channels_last(future)

    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)

    noise = torch.randn(
        future_cl.shape,
        device=device,
        dtype=future_cl.dtype,
        generator=generator,
    )

    before = evaluate(
        model,
        past,
        future,
        t,
        noise,
    )

    print(
        f"BEFORE | "
        f"MSE={before[0]:.8f} | "
        f"CC={before[1]:.4f} | "
        f"mean={before[2]:.6f} | "
        f"max={before[3]:.6f}"
    )

    # Diagnostic only: SGD avoids AdamW's extra first/second-moment buffers,
    # so the test measures "can the network memorize one sample?" without
    # spending VRAM on optimizer state.
    optimizer = torch.optim.SGD(
        model.denoiser.parameters(),
        lr=LR,
        momentum=0.0,
        weight_decay=0.0,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda")
    )

    model.train()

    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)

        x_t = model.diffusion.q_sample(
            x_start=future_cl,
            t=t,
            noise=noise,
        )

        with torch.cuda.amp.autocast(
            enabled=(device.type == "cuda")
        ):
            pred = model.denoiser(
                x_t,
                t,
                model.to_channels_last(past),
            )

            loss = torch.mean(
                (pred - future_cl) ** 2
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.denoiser.parameters(),
            1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 10 == 0:
            mse, corr, mean_val, max_val = evaluate(
                model,
                past,
                future,
                t,
                noise,
            )

            print(
                f"step={step:03d} | "
                f"MSE={mse:.8f} | "
                f"CC={corr:.4f} | "
                f"mean={mean_val:.6f} | "
                f"max={max_val:.6f}"
            )

            model.train()

    after = evaluate(
        model,
        past,
        future,
        t,
        noise,
    )

    print(
        f"AFTER  | "
        f"MSE={after[0]:.8f} | "
        f"CC={after[1]:.4f} | "
        f"mean={after[2]:.6f} | "
        f"max={after[3]:.6f}"
    )


if __name__ == "__main__":
    main()
