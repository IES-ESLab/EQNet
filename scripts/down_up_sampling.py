# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn as nn

# %%
nt = 2**4
t = torch.arange(nt)
x = torch.arange(nt).float().unsqueeze(0).unsqueeze(0)

# %% Downsampling
kernel_size = 7
stride = 4
padding = kernel_size // 2

x_down = F.interpolate(x, scale_factor=(1 / stride), mode="linear", align_corners=False)

x_conv_down = F.conv1d(x, weight=torch.ones(1, 1, kernel_size) / kernel_size, stride=stride, padding=padding)

# x_patch_down = Rearrange("b c (t s) -> b (c s) t", s=stride)(x)
x_patch_down = rearrange(x, "b c (t s) -> b (c s) t", s=stride)
x_patch_down = F.conv1d(
    x_patch_down,
    weight=torch.ones(
        1,
        stride,
        1,
    )
    / stride,
)

plt.figure()
plt.scatter(t, x.squeeze(), marker="o", s=100, alpha=1.0, c="k", label="Original")
t_ = t[::stride] + stride / 2 - 0.5
plt.scatter(t_, x_down.squeeze(), marker="^", s=80, alpha=0.7, label="Interpolate")
t_ = t[::stride]
plt.scatter(t_, x_conv_down.squeeze(), marker="s", s=80, alpha=0.7, label="Conv+Stride")
t_ = t[::stride] + stride / 2 - 0.5
plt.scatter(t_, x_patch_down.squeeze(), marker="D", s=80, alpha=0.7, label="Patch")
plt.legend()
# plt.xlim(t[0], t[-1])
# plt.ylim(t[0], t[-1])
plt.title("Comparison of downsampling methods")
plt.xlabel("index")
plt.ylabel("value")


# %% Upsampling
x_up = F.interpolate(x_down, scale_factor=(stride), mode="linear", align_corners=False)
x_conv_up_7 = F.conv_transpose1d(
    x_down,
    weight=torch.ones(1, 1, kernel_size),
    stride=stride,
    padding=padding,
    output_padding=padding,
)
x_conv_up_4 = F.conv_transpose1d(
    x_down,
    weight=torch.ones(1, 1, 4),
    stride=stride,
)
x_conv_up_8 = F.conv_transpose1d(
    x_down,
    weight=torch.ones(1, 1, 8) / 2,
    stride=stride,
    padding=stride // 2,
)

plt.figure()
plt.scatter(t, x.squeeze(), marker="o", s=100, alpha=1.0, c="k", label="Original")
t_ = t[::stride] + stride / 2 - 0.5
plt.scatter(t_, x_down.squeeze(), marker="^", s=80, alpha=0.7, label="Downsampled (Interp)")
plt.scatter(t, x_up.squeeze(), marker="s", s=80, alpha=0.7, label="Interpolate")
plt.scatter(t, x_conv_up_7.squeeze(), marker="D", s=80, alpha=0.7, label="ConvTranspose (7)")
plt.scatter(t, x_conv_up_4.squeeze(), marker="*", s=120, alpha=0.7, label="ConvTranspose (4)")
plt.scatter(t, x_conv_up_8.squeeze(), marker="P", s=80, alpha=0.7, label="ConvTranspose (8)")
# plt.xlim(t[0], t[-1])
# plt.ylim(t[0], t[-1])
plt.legend(loc="upper left")
plt.title("Comparison of upsampling methods")
plt.xlabel("index")
plt.ylabel("value")


# %%
x_up = F.interpolate(x_conv_down, scale_factor=(stride), mode="linear", align_corners=False)
x_conv_up_7 = F.conv_transpose1d(
    x_conv_down,
    weight=torch.ones(1, 1, kernel_size),
    stride=stride,
    padding=padding,
    output_padding=padding,
)
x_conv_up_4 = F.conv_transpose1d(
    x_conv_down,
    weight=torch.ones(1, 1, 4),
    stride=stride,
)
x_conv_up_8 = F.conv_transpose1d(
    x_conv_down,
    weight=torch.ones(1, 1, 8) / 2.0,
    stride=stride,
    padding=stride // 2,
)

plt.figure()
plt.scatter(t, x.squeeze(), marker="o", s=100, alpha=1.0, c="k", label="Original")
t_ = t[::stride]
plt.scatter(t_, x_conv_down.squeeze(), marker="^", s=80, alpha=0.7, label="Downsampled (Stride)")
plt.scatter(t, x_up.squeeze(), marker="s", s=80, alpha=0.7, label="Interpolate")
plt.scatter(t, x_conv_up_7.squeeze(), marker="D", s=80, alpha=0.7, label="ConvTranspose (7)")
plt.scatter(t, x_conv_up_4.squeeze(), marker="*", s=120, alpha=0.7, label="ConvTranspose (4)")
plt.scatter(t, x_conv_up_8.squeeze(), marker="P", s=80, alpha=0.7, label="ConvTranspose (8)")
# plt.xlim(t[0], t[-1])
# plt.ylim(t[0], t[-1])
plt.legend(loc="upper left")
plt.title("Comparison of upsampling methods")
plt.xlabel("index")
plt.ylabel("value")

# %%
x_up = F.interpolate(x_patch_down, scale_factor=(stride), mode="linear", align_corners=False)
x_conv_up_7 = F.conv_transpose1d(
    x_patch_down,
    weight=torch.ones(1, 1, kernel_size),
    stride=stride,
    padding=padding,
    output_padding=padding,
)
x_conv_up_4 = F.conv_transpose1d(
    x_patch_down,
    weight=torch.ones(1, 1, 4),
    stride=stride,
)
x_conv_up_8 = F.conv_transpose1d(
    x_patch_down,
    weight=torch.ones(1, 1, 8) / 2.0,
    stride=stride,
    padding=stride // 2,
)

plt.figure()
plt.scatter(t, x.squeeze(), marker="o", s=100, alpha=1.0, c="k", label="Original")
t_ = t[::stride] + stride / 2 - 0.5
plt.scatter(t_, x_patch_down.squeeze(), marker="^", s=80, alpha=0.7, label="Downsampled (Patch)")
plt.scatter(t, x_up.squeeze(), marker="s", s=80, alpha=0.7, label="Interpolate")
plt.scatter(t, x_conv_up_7.squeeze(), marker="D", s=80, alpha=0.7, label="ConvTranspose (7)")
plt.scatter(t, x_conv_up_4.squeeze(), marker="*", s=120, alpha=0.7, label="ConvTranspose (4)")
plt.scatter(t, x_conv_up_8.squeeze(), marker="P", s=80, alpha=0.7, label="ConvTranspose (8)")
# plt.xlim(t[0], t[-1])
# plt.ylim(t[0], t[-1])
plt.legend(loc="upper left")
plt.title("Comparison of upsampling methods")
plt.xlabel("index")
plt.ylabel("value")

# %%
