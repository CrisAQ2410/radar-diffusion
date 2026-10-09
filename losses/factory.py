import torch.nn.functional as F


def build_loss(cfg):
    loss_type = cfg["type"].lower()

    if loss_type == "mse":

        def loss_fn(
            pred,
            target,
            past=None,
        ):
            return F.mse_loss(
                pred,
                target,
            )

        return loss_fn

    raise ValueError(
        f"Unknown loss type: {loss_type}"
    )
