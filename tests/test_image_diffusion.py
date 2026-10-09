import torch

from models.diffusion import DiffusionSchedule


device = "cuda:2"

diffusion = DiffusionSchedule(
    timesteps=1000,
).to(device)


x0 = torch.rand(
    2,
    6,
    80,
    80,
    1,
    device=device,
)

t = torch.tensor(
    [100, 900],
    device=device,
).long()

noise = torch.randn_like(
    x0
)

x_t = diffusion.q_sample(
    x_start=x0,
    t=t,
    noise=noise,
)


print(
    "x0 :",
    x0.shape
)

print(
    "x_t:",
    x_t.shape
)

print(
    "range x0:",
    x0.min().item(),
    x0.max().item()
)

print(
    "range x_t:",
    x_t.min().item(),
    x_t.max().item()
)