import torch

from losses.factory import (
    build_loss,
    weighted_mse_loss,
    weighted_different_loss,
)


torch.manual_seed(123)

B = 2
T = 6
H = 80
W = 80

past = torch.rand(
    B,
    T,
    H,
    W,
)

target = torch.rand(
    B,
    T,
    H,
    W,
)

pred = torch.rand(
    B,
    T,
    H,
    W,
    1,
)


configs = [
    {
        "type": "mse",
        "weighting": "balanced",
    },
    {
        "type": "mse",
        "weighting": "gradual",
    },
    {
        "type": "mse",
        "weighting": "steep",
    },
    {
        "type": "pool_mse",
        "pool_type": "max",
        "kernel_size": 4,
        "weighting": "steep",
    },
    {
        "type": "pool_mse",
        "pool_type": "mean",
        "kernel_size": 4,
        "weighting": "gradual",
    },
    {
        "type": "different",
        "weighting": "steep",
    },
    {
        "type": "combine",
        "weighting": "steep",
        "mse_ratio": 0.5,
        "different_ratio": 0.5,
    },
]


print("=== LOSS SMOKE TEST ===")

for cfg in configs:
    loss_fn = build_loss(
        cfg
    )

    loss = loss_fn(
        pred,
        target,
        past,
    )

    assert torch.isfinite(
        loss
    )

    print(
        cfg,
        "->",
        float(loss),
    )


# ------------------------------------------------------------
# Verify the important mathematical consequence of Eq. 3.5:
# same anchor on both sides cancels.
# ------------------------------------------------------------

mse_steep = weighted_mse_loss(
    pred,
    target.unsqueeze(-1),
    weighting="steep",
)

diff_steep = weighted_different_loss(
    pred,
    target.unsqueeze(-1),
    past,
    weighting="steep",
)

print()
print(
    "MSE steep:",
    float(mse_steep),
)

print(
    "Different steep (literal thesis Eq. 3.5):",
    float(diff_steep),
)

print(
    "Absolute difference:",
    float(
        torch.abs(
            mse_steep
            - diff_steep
        )
    ),
)

assert torch.allclose(
    mse_steep,
    diff_steep,
    atol=1e-7,
    rtol=1e-6,
)

print(
    "\nPASS: literal Eq. 3.5 is numerically "
    "equivalent to MSE when the same temporal "
    "weights are used."
)
