import torch
from models.LDM.cuboid_transformer_unet import CuboidTransformerUNet

device = "cuda:2"

model = CuboidTransformerUNet(
    input_shape=[6, 32, 32, 64],
    target_shape=[6, 32, 32, 64],

    base_units=128,
    scale_alpha=1.0,

    depth=[4, 4],
    downsample=2,
    downsample_type="patch_merge",

    upsample_type="upsample",
    upsample_kernel_size=3,

    block_attn_patterns=["axial", "axial"],

    num_heads=4,

    attn_drop=0.1,
    proj_drop=0.1,
    ffn_drop=0.1,

    ffn_activation="gelu",
    gated_ffn=False,

    norm_layer="layer_norm",
    padding_type="zeros",
    pos_embed_type="t+h+w",

    checkpoint_level=0,
    use_relative_pos=True,
    self_attn_use_final_proj=True,

    num_global_vectors=0,
    use_global_vector_ffn=False,
    use_global_self_attn=True,
    separate_global_qkv=True,
    global_dim_ratio=1,

    attn_linear_init_mode="0",
    ffn_linear_init_mode="0",
    ffn2_linear_init_mode="2",
    attn_proj_linear_init_mode="2",

    conv_init_mode="0",
    down_linear_init_mode="0",
    up_linear_init_mode="0",

    global_proj_linear_init_mode="2",
    norm_init_mode="0",

    time_embed_channels_mult=4,
    time_embed_use_scale_shift_norm=False,
    time_embed_dropout=0.0,

    unet_res_connect=True,
).to(device)


x = torch.randn(
    1, 6, 32, 32, 64,
    device=device
)

cond = torch.randn(
    1, 6, 32, 32, 64,
    device=device
)

t = torch.randint(
    0,
    1000,
    (1,),
    device=device
)


with torch.no_grad():
    out = model(
        x,
        t,
        cond
    )


print("x    :", x.shape)
print("cond :", cond.shape)
print("t    :", t.shape)
print("out  :", out.shape)