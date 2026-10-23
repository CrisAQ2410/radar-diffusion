import torch
import torch.nn.functional as F


WEIGHT_PRESETS = {
    "balanced": [
        1 / 6,
        1 / 6,
        1 / 6,
        1 / 6,
        1 / 6,
        1 / 6,
    ],
    "gradual": [
        0.35,
        0.25,
        0.15,
        0.10,
        0.10,
        0.05,
    ],
    "steep": [
        0.50,
        0.15,
        0.15,
        0.10,
        0.05,
        0.05,
    ],
}


def _to_channels_last(x):
    """
    Supported shapes:
      [B,T,H,W]
      [B,T,H,W,1]
    Returns:
      [B,T,H,W,1]
    """
    if x.dim() == 4:
        return x.unsqueeze(-1)

    if x.dim() == 5:
        return x

    raise ValueError(
        f"Expected [B,T,H,W] or [B,T,H,W,C], "
        f"got {tuple(x.shape)}"
    )


def _get_weights(
    name,
    num_steps,
    device,
    dtype,
):
    name = name.lower()

    if name not in WEIGHT_PRESETS:
        raise ValueError(
            f"Unknown temporal weighting: {name}"
        )

    values = WEIGHT_PRESETS[name]

    if len(values) != num_steps:
        raise ValueError(
            f"Preset {name} has {len(values)} weights, "
            f"but prediction has {num_steps} steps."
        )

    weights = torch.tensor(
        values,
        device=device,
        dtype=dtype,
    )

    return weights / weights.sum()


def _weighted_step_reduce(
    step_losses,
    weights,
):
    """
    step_losses: [T]
    weights:     [T]
    """
    return (
        step_losses * weights
    ).sum()


def weighted_mse_loss(
    pred,
    target,
    weighting="balanced",
):
    """
    Thesis Eq. 3.3 + Eq. 3.6:
      calculate MSE independently for each forecast step,
      then combine using temporal weights.
    """
    pred = _to_channels_last(pred)
    target = _to_channels_last(target)

    if pred.shape != target.shape:
        raise ValueError(
            f"pred {tuple(pred.shape)} != "
            f"target {tuple(target.shape)}"
        )

    # [T]
    step_losses = (
        (pred - target)
        .pow(2)
        .mean(
            dim=(0, 2, 3, 4)
        )
    )

    weights = _get_weights(
        weighting,
        pred.shape[1],
        pred.device,
        pred.dtype,
    )

    return _weighted_step_reduce(
        step_losses,
        weights,
    )


def _pool_spatial(
    x,
    pool_type,
    kernel_size,
):
    """
    x: [B,T,H,W,1]

    Pool only H,W independently for every forecast frame.
    """
    x = _to_channels_last(x)

    b, t, h, w, c = x.shape

    if c != 1:
        raise ValueError(
            "Current radar setup expects one channel."
        )

    # [B*T,1,H,W]
    x = (
        x.permute(
            0, 1, 4, 2, 3
        )
        .reshape(
            b * t,
            c,
            h,
            w,
        )
    )

    if pool_type == "max":
        x = F.max_pool2d(
            x,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    elif pool_type == "mean":
        x = F.avg_pool2d(
            x,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    else:
        raise ValueError(
            f"Unknown pool_type: {pool_type}"
        )

    hp, wp = x.shape[-2:]

    # back to [B,T,H',W',1]
    return (
        x.reshape(
            b,
            t,
            c,
            hp,
            wp,
        )
        .permute(
            0, 1, 3, 4, 2
        )
        .contiguous()
    )


def weighted_pool_mse_loss(
    pred,
    target,
    pool_type="max",
    kernel_size=4,
    weighting="balanced",
):
    """
    Thesis Eq. 3.4 + Eq. 3.6:
      Pool(pred), Pool(target), MSE per forecast step,
      then temporal weighting.
    """
    pred_pool = _pool_spatial(
        pred,
        pool_type=pool_type,
        kernel_size=kernel_size,
    )

    target_pool = _pool_spatial(
        target,
        pool_type=pool_type,
        kernel_size=kernel_size,
    )

    return weighted_mse_loss(
        pred_pool,
        target_pool,
        weighting=weighting,
    )


def weighted_different_loss(
    pred,
    target,
    past,
    weighting="balanced",
):
    """
    Thesis Eq. 3.5 + Eq. 3.6.

    Anchor = final observed frame X_t0.

    For each future step k:
      MSE(
          pred_k   - X_t0,
          target_k - X_t0
      )

    NOTE:
    As written mathematically in the thesis, subtracting the same
    anchor from both terms cancels algebraically, so this expression
    is numerically equivalent to frame-wise MSE when the same weighting
    is used. This function intentionally follows the thesis equation
    literally so the implementation is traceable to the document.
    """
    pred = _to_channels_last(pred)
    target = _to_channels_last(target)
    past = _to_channels_last(past)

    if pred.shape != target.shape:
        raise ValueError(
            f"pred {tuple(pred.shape)} != "
            f"target {tuple(target.shape)}"
        )

    if past.shape[0] != pred.shape[0]:
        raise ValueError(
            "past and prediction batch sizes differ"
        )

    # [B,H,W,1]
    anchor = past[:, -1]

    # [B,T,H,W,1]
    pred_delta = (
        pred
        - anchor.unsqueeze(1)
    )

    target_delta = (
        target
        - anchor.unsqueeze(1)
    )

    step_losses = (
        (pred_delta - target_delta)
        .pow(2)
        .mean(
            dim=(0, 2, 3, 4)
        )
    )

    weights = _get_weights(
        weighting,
        pred.shape[1],
        pred.device,
        pred.dtype,
    )

    return _weighted_step_reduce(
        step_losses,
        weights,
    )


def hybrid_mse_different_loss(
    pred,
    target,
    past,
    mse_ratio=0.5,
    different_ratio=0.5,
    weighting="steep",
):
    """
    Thesis Eq. 3.7.

    Best reported setting in the thesis:
      0.5 * MSE
      + 0.5 * Different
      with Steep Decay temporal weighting.
    """
    total_ratio = (
        mse_ratio
        + different_ratio
    )

    if total_ratio <= 0:
        raise ValueError(
            "Loss ratios must sum to > 0"
        )

    mse_ratio = (
        mse_ratio
        / total_ratio
    )

    different_ratio = (
        different_ratio
        / total_ratio
    )

    mse = weighted_mse_loss(
        pred,
        target,
        weighting=weighting,
    )

    different = weighted_different_loss(
        pred,
        target,
        past,
        weighting=weighting,
    )

    return (
        mse_ratio * mse
        + different_ratio * different
    )


def build_loss(cfg):
    """
    Returns a callable:
        loss_fn(pred, target, past)

    YAML examples:

    loss:
      type: mse
      weighting: balanced

    loss:
      type: pool_mse
      pool_type: max
      kernel_size: 4
      weighting: steep

    loss:
      type: different
      weighting: steep

    loss:
      type: combine
      weighting: steep
      mse_ratio: 0.5
      different_ratio: 0.5
    """
    loss_type = (
        cfg.get(
            "type",
            "mse",
        )
        .lower()
    )

    weighting = (
        cfg.get(
            "weighting",
            "balanced",
        )
        .lower()
    )

    if loss_type == "mse":

        def loss_fn(
            pred,
            target,
            past=None,
        ):
            return weighted_mse_loss(
                pred,
                target,
                weighting=weighting,
            )

        return loss_fn

    if loss_type in {
        "pool_mse",
        "pool",
        "max_pool",
        "mean_pool",
    }:
        if loss_type == "max_pool":
            pool_type = "max"
        elif loss_type == "mean_pool":
            pool_type = "mean"
        else:
            pool_type = (
                cfg.get(
                    "pool_type",
                    "max",
                )
                .lower()
            )

        kernel_size = int(
            cfg.get(
                "kernel_size",
                4,
            )
        )

        if kernel_size not in {
            4,
            8,
        }:
            raise ValueError(
                "Thesis Pool MSE experiments use "
                "kernel_size 4 or 8."
            )

        def loss_fn(
            pred,
            target,
            past=None,
        ):
            return weighted_pool_mse_loss(
                pred,
                target,
                pool_type=pool_type,
                kernel_size=kernel_size,
                weighting=weighting,
            )

        return loss_fn

    if loss_type in {
        "different",
        "diff",
    }:

        def loss_fn(
            pred,
            target,
            past=None,
        ):
            if past is None:
                raise ValueError(
                    "Different Loss requires past frames "
                    "to obtain the final input anchor t0."
                )

            return weighted_different_loss(
                pred,
                target,
                past,
                weighting=weighting,
            )

        return loss_fn

    if loss_type in {
        "combine",
        "hybrid",
        "mse_different",
    }:
        mse_ratio = float(
            cfg.get(
                "mse_ratio",
                0.5,
            )
        )

        different_ratio = float(
            cfg.get(
                "different_ratio",
                0.5,
            )
        )

        def loss_fn(
            pred,
            target,
            past=None,
        ):
            if past is None:
                raise ValueError(
                    "Combine Loss requires past frames "
                    "for Different Loss."
                )

            return hybrid_mse_different_loss(
                pred,
                target,
                past,
                mse_ratio=mse_ratio,
                different_ratio=different_ratio,
                weighting=weighting,
            )

        return loss_fn

    raise ValueError(
        f"Unknown loss type: {loss_type}"
    )
