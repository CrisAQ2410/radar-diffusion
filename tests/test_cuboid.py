import torch
import yaml

from models.LDM.cuboid_transformer_unet import CuboidTransformerUNet


with open(
    "configs/radar_ldm.yaml",
    "r",
) as f:
    cfg = yaml.safe_load(f)


device = torch.device(
    cfg["training"]["device"]
    if torch.cuda.is_available()
    else "cpu"
)

m = cfg["model"]

model = CuboidTransformerUNet(
    input_shape=m["input_shape"],
    target_shape=m["target_shape"],

    base_units=m["base_units"],
    scale_alpha=m["scale_alpha"],
    num_heads=m["num_heads"],

    depth=m["depth"],

    block_attn_patterns=m["self_pattern"],

    downsample=m["downsample"],
    downsample_type=m["downsample_type"],

    upsample_type=m["upsample_type"],
    upsample_kernel_size=m["upsample_kernel_size"],

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
).to(device)


x = torch.rand(
    1,
    6,
    80,
    80,
    1,
    device=device,
)

cond = torch.rand(
    1,
    6,
    80,
    80,
    1,
    device=device,
)

t = torch.randint(
    0,
    1000,
    (1,),
    device=device,
).long()


with torch.no_grad():

    out = model(
        x,
        t,
        cond,
    )


print(
    "x    :",
    x.shape
)

print(
    "cond :",
    cond.shape
)

print(
    "t    :",
    t.shape
)

print(
    "out  :",
    out.shape
)