import math
from functools import partial, wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

def log_transform(x):
    x = torch.sgn(x) * torch.log1p(torch.abs(x))
    return x

def moving_normalize(data, filter=1024, stride=256):
    if len(data.shape) == 5:
        freq_dim = True
        nb, nch, nx, nf, nt = data.shape
        data = data.view(nb, nch, nx * nf, nt)
    else:
        freq_dim = False
        nb, nch, nx, nt = data.shape

    # Adjust filter and stride to be smaller than input time dimension
    filter = min(filter, nt)
    stride = min(stride, filter)

    if nt % stride == 0:
        pad = max(filter - stride, 0)
    else:
        pad = max(filter - (nt % stride), 0)
    pad1 = pad // 2
    pad2 = pad - pad1

    with torch.no_grad():
        data_ = F.pad(data, (pad1, pad2, 0, 0), mode="reflect")
        mean = F.avg_pool2d(data_, kernel_size=(1, filter), stride=(1, stride), count_include_pad=False)
        mean = F.interpolate(mean, scale_factor=(1, stride), mode="bilinear", align_corners=False)[:, :, :nx, :nt]
        data -= mean

        data_ = F.pad(data, (pad1, pad2, 0, 0), mode="reflect")
        std = F.avg_pool2d(torch.abs(data_), kernel_size=(1, filter), stride=(1, stride), count_include_pad=False)
        std = torch.mean(std, dim=(1,), keepdim=True)  # keep relative amplitude between channels
        std = F.interpolate(std, scale_factor=(1, stride), mode="bilinear", align_corners=False)[:, :, :nx, :nt]
        std[std == 0.0] = 1.0
        data = data / std

    if freq_dim:
        data = data.view(nb, nch, nx, nf, nt)

    return data

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

# helper classes

class Identity(nn.Identity):
    """Identity that accepts and ignores extra args/kwargs."""
    def forward(self, x, *args, **kwargs):
        return x

def l2norm(t):
    return F.normalize(t, dim=-1)

def get_transformer_block(layer_attn, layer_use_linear_attn):
    """Select appropriate transformer block based on attention flags."""
    if layer_attn:
        return TransformerBlock
    if layer_use_linear_attn:
        return LinearAttentionTransformerBlock
    return Identity

class STFT(nn.Module):
    def __init__(
        self,
        n_fft=64 + 1,
        hop_length=4,
        window_fn=torch.hann_window,
        magnitude=True,
        normalize_freq=False,
        discard_zero_freq=True,
        **kwargs,
    ):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.magnitude = magnitude
        self.discard_zero_freq = discard_zero_freq
        self.normalize_freq = normalize_freq
        self.register_buffer("window", window_fn(n_fft))

    def forward(self, x):
        """
        x: bt, ch, nt
        """
        nb, nc, nt = x.shape
        x = x.view(-1, nt)  # nb*nc, nt
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            window=self.window,
            hop_length=self.hop_length,
            center=True,
            return_complex=True,
        )
        stft = torch.view_as_real(stft)
        if self.discard_zero_freq:
            stft = stft.narrow(dim=-3, start=1, length=stft.shape[-3] - 1)
        nf, nt_out, _ = stft.shape[-3:]
        if self.magnitude:
            stft = torch.norm(stft, dim=-1, keepdim=False).view(nb, nc, nf, nt_out)  # nb, nc, nf, nt
        else:
            stft = stft.view(nb, nc, nf, nt_out, 2)  # nb, nc, nf, nt, 2
            stft = rearrange(stft, "b c nf nt d -> b (c d) nf nt")  # nb, nc*2, nf, nt

        if self.normalize_freq:
            vmax = torch.max(torch.abs(stft), dim=-2, keepdim=True)[0]
            vmax[vmax == 0.0] = 1.0
            stft = stft / vmax

        return stft

class MergeFrequency(nn.Module):
    """
    Merge frequency dimension to 1 using a linear layer.
    """

    def __init__(self, dim_in):
        super().__init__()
        self.linear = nn.Linear(dim_in, 1)

    def forward(self, x):
        # x: nb, nc, nf, nt
        x = x.permute(0, 1, 3, 2)  # nb, nc, nt, nf
        x = self.linear(x).squeeze(-1)  # nb, nc, nt
        return x

class MergeBranch(nn.Module):
    """
    Merge two branches of the same dimension.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, 1)

    def forward(self, x1, x2):
        return self.conv(torch.cat((x1, x2), dim=1))

def resize_image_to(
    image,
    target_image_size,
    clamp_range = None,
    mode = 'nearest'
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode = mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out

# norms and residuals

class ChanRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.scale * self.gamma

class LayerNorm(nn.Module):
    def __init__(self, feats, stable = False, dim = -1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)

ChanLayerNorm = partial(LayerNorm, dim = -3)

class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        context_dim = None,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context = None, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# decoder

def Upsample(dim, dim_out = None, stride = (1, 2)):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor = stride, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None, stride = (1, 2)):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.stride = stride
        scale_factor = stride[0] * stride[1]
        conv = nn.Conv2d(dim, dim_out * scale_factor, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c s1 s2) h w -> b c (h s1) (w s2)', s1=stride[0], s2=stride[1])
        )

        self.init_conv_(conv, scale_factor)

    def init_conv_(self, conv, scale_factor):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // scale_factor, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, f'o ... -> (o {scale_factor}) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(dim, dim_out = None, stride = (1, 2)):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    dim_out = default(dim_out, dim)
    scale_factor = stride[0] * stride[1]
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = stride[0], s2 = stride[1]),
        nn.Conv2d(dim * scale_factor, dim_out, 1)
    )

class Block(nn.Module):
    def __init__(self, dim, dim_out, norm = True, kernel_size = (1, 7)):
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.norm = ChanRMSNorm(dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, kernel_size, padding = padding)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        return self.project(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, kernel_size = (1, 7)):
        super().__init__()
        self.block1 = Block(dim, dim_out, kernel_size = kernel_size)
        self.block2 = Block(dim_out, dim_out, kernel_size = kernel_size)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8,
        dropout = 0.05,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias = False)
    )

def ChanFeedForward(dim, mult = 2):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, hidden_dim, 1, bias = False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv2d(hidden_dim, dim, 1, bias = False)
    )

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')
        return x

class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearAttention(dim = dim, heads = heads, dim_head = dim_head),
                ChanFeedForward(dim = dim, mult = ff_mult)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = (1, 2)
    ):
        super().__init__()
        stride = (stride, stride) if isinstance(stride, int) else stride
        # Scale kernel sizes proportionally if stride > 2 (original design assumes stride=2)
        scale_ratio = max(1, stride[1] // 2)
        kernel_sizes = tuple(k * scale_ratio for k in kernel_sizes)
        # kernel_sizes are for temporal dim only (width), height kernel is always 1
        assert all([*map(lambda t: (t % 2) == (stride[1] % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            # kernel is temporal width, use (1, kernel) for 2D conv
            kernel_2d = (1, kernel)
            padding = (0, (kernel - stride[1]) // 2)
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel_2d, stride=stride, padding=padding))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)

class UpsampleCombiner(nn.Module):
    def __init__(
        self,
        dim,
        *,
        enabled = False,
        dim_ins = tuple(),
        dim_outs = tuple()
    ):
        super().__init__()
        dim_outs = cast_tuple(dim_outs, len(dim_ins))
        assert len(dim_ins) == len(dim_outs)

        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)

    def forward(self, x, fmaps = None):
        target_size = x.shape[-2:]  # (height, width)

        fmaps = default(fmaps, tuple())

        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        fmaps = [F.interpolate(fmap, size=target_size, mode='nearest') for fmap in fmaps]
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        return torch.cat((x, *outs), dim = 1)

class Unet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_resnet_blocks = 1,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        phase_channels = None,
        polarity_channels = 1,
        event_channels = 1,
        attn_dim_head = 64,
        attn_heads = 8,
        ff_mult = 2,
        layer_attns = True,
        layer_attns_depth = 1,
        layer_mid_attns_depth = 1,
        attend_at_middle = True,
        use_linear_attn = False,
        init_dim = None,
        init_conv_kernel_size = 7,
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        dropout = 0.,
        memory_efficient = False,
        scale_skip_connection = True,
        final_resnet_block = True,
        final_conv_kernel_size = 3,
        combine_upsample_fmaps = False,
        pixel_shuffle_upsample = True,
        # stride and kernel per dimension (space=station/height, time=samples/width, freq=STFT frequency)
        space_stride = 1,
        time_stride = 4,
        freq_stride = 2,
        space_kernel = 1,
        time_kernel = 7,
        freq_kernel = 3,
        # domain-specific features
        moving_norm = (1024, 256),
        log_scale = False,
        add_stft = False,
        stft_n_fft = 33,  # STFT window size (65 -> 32 freq bins, 33 -> 16, 17 -> 8)
        stft_dim_divisor = 2,  # reduce STFT encoder channels by this factor (1, 2, or 4)
        add_polarity = False,
        add_event = False,
        add_prompt = False,
    ):
        super().__init__()

        # derive composite tuples from per-dimension settings
        stride = (space_stride, time_stride)
        kernel_size = (space_kernel, time_kernel)

        # domain-specific settings
        self.moving_norm = moving_norm
        self.log_scale = log_scale
        self.add_stft = add_stft
        self.add_polarity = add_polarity
        self.add_event = add_event
        self.add_prompt = add_prompt
        self.stride = stride
        self.kernel_size = kernel_size

        # validate attention heads
        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'

        # determine dimensions

        self.channels = channels
        self.phase_channels = default(phase_channels, channels)
        self.polarity_channels = polarity_channels
        self.event_channels = event_channels

        init_dim = default(init_dim, dim)

        # initial convolution

        self.init_conv = CrossEmbedLayer(channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1) if init_cross_embed else nn.Conv2d(channels, init_dim, init_conv_kernel_size, padding = init_conv_kernel_size // 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # attention related params

        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head)

        num_layers = len(in_out)

        # resnet block klass

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)

        resnet_klass = partial(ResnetBlock, kernel_size = kernel_size)

        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_attns_depth = cast_tuple(layer_attns_depth, num_layers)

        use_linear_attn = cast_tuple(use_linear_attn, num_layers)

        assert all([layers == num_layers for layers in list(map(len, (layer_attns,)))])

        # downsample klass

        downsample_klass = partial(Downsample, stride = stride)

        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes = cross_embed_downsample_kernel_sizes, stride = stride)

        # initial resnet block (for memory efficient unet)

        self.init_resnet_block = resnet_klass(init_dim, init_dim) if memory_efficient else None

        # scale for resnet skip connections

        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [num_resnet_blocks, layer_attns, layer_attns_depth, use_linear_attn]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers

        skip_connect_dims = [] # keep track of skip connection dimensions

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, layer_attn, layer_attn_depth, layer_use_linear_attn) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)
            transformer_block_klass = get_transformer_block(layer_attn, layer_use_linear_attn)
            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet

            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out=dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet

            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(current_dim, dim_out=dim_out) if not is_last else Parallel(nn.Conv2d(dim_in, dim_out, 3, padding = 1), nn.Conv2d(dim_in, dim_out, 1))

            self.downs.append(nn.ModuleList([
                pre_downsample,
                resnet_klass(current_dim, current_dim),
                nn.ModuleList([resnet_klass(current_dim, current_dim) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = current_dim, depth = layer_attn_depth, ff_mult = ff_mult, **attn_kwargs),
                post_downsample
            ]))

        # middle layers

        mid_dim = dims[-1]

        self.mid_block1 = resnet_klass(mid_dim, mid_dim)
        self.mid_attn = TransformerBlock(mid_dim, depth = layer_mid_attns_depth, **attn_kwargs) if attend_at_middle else None
        self.mid_block2 = resnet_klass(mid_dim, mid_dim)

        # upsample klass

        upsample_klass = partial(Upsample, stride = stride) if not pixel_shuffle_upsample else partial(PixelShuffleUpsample, stride = stride)

        # upsampling layers

        upsample_fmap_dims = []

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, layer_attn, layer_attn_depth, layer_use_linear_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (len(in_out) - 1)
            transformer_block_klass = get_transformer_block(layer_attn, layer_use_linear_attn)
            skip_connect_dim = skip_connect_dims.pop()

            upsample_fmap_dims.append(dim_out)

            self.ups.append(nn.ModuleList([
                resnet_klass(dim_out + skip_connect_dim, dim_out),
                nn.ModuleList([resnet_klass(dim_out + skip_connect_dim, dim_out) for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = dim_out, depth = layer_attn_depth, ff_mult = ff_mult, **attn_kwargs),
                upsample_klass(dim_out, dim_in) if not is_last or memory_efficient else Identity()
            ]))

        # whether to combine feature maps from all upsample blocks before final resnet block out
        # output dim after all upsamples depends on memory_efficient mode
        upsample_out_dim = init_dim if memory_efficient else dim

        self.upsample_combiner = UpsampleCombiner(
            dim = upsample_out_dim,
            enabled = combine_upsample_fmaps,
            dim_ins = upsample_fmap_dims,
            dim_outs = upsample_out_dim
        )

        # final optional resnet block and convolution out

        final_conv_dim = self.upsample_combiner.dim_out

        self.final_res_block = resnet_klass(final_conv_dim, init_dim) if final_resnet_block else None

        final_conv_dim_in = init_dim if final_resnet_block else final_conv_dim

        self.final_conv = nn.Conv2d(final_conv_dim_in, self.phase_channels, final_conv_kernel_size, padding = final_conv_kernel_size // 2)

        zero_init_(self.final_conv)

        # store for domain-specific layers
        self.dims = dims
        self.in_out = in_out
        self.memory_efficient = memory_efficient

        # Polarity encoder - works at original scale without downsampling
        # Encodes raw waveform and concatenates with phase features for prediction
        if self.add_polarity:
            # Use first stage params for polarity encoder (works at full resolution)
            layer_num_resnet_blocks = num_resnet_blocks[0]
            layer_use_linear_attn = use_linear_attn[0]

            self.polarity_init = nn.Conv2d(1, init_dim, init_conv_kernel_size, padding=init_conv_kernel_size // 2)
            self.polarity_init_resnet = resnet_klass(init_dim, init_dim) if memory_efficient else None
            self.polarity_encoder = nn.ModuleList([
                resnet_klass(init_dim, init_dim),
                nn.ModuleList([resnet_klass(init_dim, init_dim) for _ in range(layer_num_resnet_blocks)]),
                LinearAttentionTransformerBlock(dim=init_dim, depth=1, ff_mult=ff_mult, **attn_kwargs) if layer_use_linear_attn else Identity(),
            ])
            # Merge with phase features (from decoder output, same dim as dims[0])
            self.polarity_merge = MergeBranch(dims[0] * 2, dims[0])
            self.polarity_final = nn.Sequential(
                resnet_klass(init_dim, init_dim),
                nn.Conv2d(init_dim, self.polarity_channels, final_conv_kernel_size, padding=final_conv_kernel_size // 2),
            )

        # Event/Prompt - always outputs at /16 scale (upsample if deeper)
        if self.add_event or self.add_prompt:
            num_downsamples = num_resolutions if memory_efficient else (num_resolutions - 1)
            current_scale = time_stride ** num_downsamples
            target_scale = 16
            num_upsamples = int(math.log(current_scale / target_scale, time_stride)) if current_scale > target_scale else 0
            self.event_scale_layers = nn.ModuleList([upsample_klass(mid_dim, mid_dim) for _ in range(num_upsamples)])

        if self.add_event:
            self.event_block = resnet_klass(mid_dim, mid_dim)
            self.event_final = nn.Sequential(
                resnet_klass(mid_dim, mid_dim),
                nn.Conv2d(mid_dim, self.event_channels, final_conv_kernel_size, padding=final_conv_kernel_size // 2),
            )

        # STFT encoder - mirrors main encoder structure but starts from stage 1
        # STFT hop_length=time_stride means output already has 1/time_stride temporal resolution
        # So STFT output matches main encoder stage 1 resolution, we skip stage 0
        if self.add_stft:
            n_fft = stft_n_fft
            self.n_freq = n_fft // 2  # frequency bins after discarding zero freq
            stft_kernel_size = (freq_kernel, time_kernel)

            # Scaled dimensions for STFT encoder (reduce compute by stft_dim_divisor)
            stft_dims = [max(d // stft_dim_divisor, 16) for d in dims]  # min 16 channels
            stft_mid_dim = stft_dims[-1]
            stft_in_out = list(zip(stft_dims[:-1], stft_dims[1:]))

            # Downsample both frequency and time dimensions
            stft_stride = (freq_stride, time_stride)
            stft_resnet_klass = partial(ResnetBlock, kernel_size=stft_kernel_size)
            stft_downsample_klass = partial(Downsample, stride=stft_stride)

            self.stft = STFT(n_fft=n_fft, hop_length=time_stride)
            # Start with stft_dims[1] since we skip stage 0 (STFT already at that resolution)
            self.spec_init = nn.Conv2d(channels, stft_dims[1], stft_kernel_size, padding=(freq_kernel // 2, time_kernel // 2))
            self.spec_init_resnet = stft_resnet_klass(stft_dims[1], stft_dims[1]) if memory_efficient else None
            self.spec_downs = nn.ModuleList([])

            # Build encoder stages starting from stage 1 (skip stage 0)
            # Track frequency dimension as it gets downsampled
            # Forward flow: pre_downsample -> resnets -> merge_freq -> post_downsample
            current_n_freq = self.n_freq
            for ind, ((stft_dim_in, stft_dim_out), (main_dim_in, main_dim_out), layer_num_resnet_blocks, layer_attn, layer_attn_depth, layer_use_linear_attn) in enumerate(zip(stft_in_out, in_out, *layer_params)):
                if ind == 0:
                    continue  # skip stage 0, STFT already at stage 1 resolution
                is_last = ind >= (num_resolutions - 1)
                transformer_block_klass = get_transformer_block(layer_attn, layer_use_linear_attn)
                current_stft_dim = stft_dim_in
                current_main_dim = main_dim_in if not memory_efficient else main_dim_out
                pre_downsample = None
                if memory_efficient:
                    pre_downsample = stft_downsample_klass(stft_dim_in, stft_dim_out)
                    current_stft_dim = stft_dim_out
                    # Freq is reduced by pre_downsample BEFORE merge_freq
                    current_n_freq = max(1, current_n_freq // freq_stride)

                post_downsample = None
                if not memory_efficient:
                    post_downsample = stft_downsample_klass(current_stft_dim, stft_dim_out) if not is_last else nn.Conv2d(stft_dim_in, stft_dim_out, 1)

                # MergeFrequency uses current freq (after pre_downsample if any, before post_downsample)
                stage_n_freq = current_n_freq

                self.spec_downs.append(nn.ModuleList([
                    pre_downsample,
                    stft_resnet_klass(current_stft_dim, current_stft_dim),
                    nn.ModuleList([stft_resnet_klass(current_stft_dim, current_stft_dim) for _ in range(layer_num_resnet_blocks)]),
                    transformer_block_klass(dim=current_stft_dim, depth=layer_attn_depth, ff_mult=ff_mult, **attn_kwargs),
                    post_downsample,
                    MergeFrequency(stage_n_freq),
                    MergeBranch(current_stft_dim + current_main_dim, current_main_dim),
                ]))

                # Update freq for next stage (post_downsample reduces freq AFTER merge_freq)
                if not memory_efficient and post_downsample is not None and not is_last:
                    current_n_freq = max(1, current_n_freq // freq_stride)

            # Final frequency dimension at bottleneck
            self.spec_mid_n_freq = current_n_freq
            self.spec_mid_block1 = stft_resnet_klass(stft_mid_dim, stft_mid_dim)
            self.spec_mid_merge_freq = MergeFrequency(current_n_freq)
            self.mid_merge = MergeBranch(stft_mid_dim + mid_dim, mid_dim)

    def _forward_stft(self, x_norm, hiddens, stage_indices):
        """STFT encoder - processes spectrogram and merges into hiddens for skip connections."""
        nb, nc, nx, nt = x_norm.shape
        x_stft = x_norm.permute(0, 2, 1, 3).reshape(nb * nx, nc, nt)
        x_stft = self.stft(x_stft)
        sgram = x_stft
        x_stft = self.spec_init(sgram)

        if self.spec_init_resnet is not None:
            x_stft = self.spec_init_resnet(x_stft)

        for i, (pre_downsample, init_block, resnet_blocks, attn_block, post_downsample, merge_freq, merge_branch) in enumerate(self.spec_downs):
            if exists(pre_downsample):
                x_stft = pre_downsample(x_stft)

            x_stft = init_block(x_stft)
            for resnet_block in resnet_blocks:
                x_stft = resnet_block(x_stft)
            x_stft = attn_block(x_stft)

            # merge with main encoder hidden (STFT stage i → main encoder stage i+1)
            idx = stage_indices[i + 1]
            x_stft_m = merge_freq(x_stft).view(nb, nx, -1, x_stft.shape[-1]).permute(0, 2, 1, 3)
            hiddens[idx] = merge_branch(hiddens[idx], x_stft_m)

            if exists(post_downsample):
                x_stft = post_downsample(x_stft)

        # STFT mid processing
        x_stft = self.spec_mid_block1(x_stft)
        x_stft = self.spec_mid_merge_freq(x_stft).view(nb, nx, -1, x_stft.shape[-1]).permute(0, 2, 1, 3)
        return x_stft, sgram

    def _forward_polarity(self, x_polarity, x_phase):
        """Polarity prediction - encodes raw waveform and merges with phase features."""
        if self.polarity_init_resnet is not None:
            x_polarity = self.polarity_init_resnet(x_polarity)
        init_block, resnet_blocks, attn_block = self.polarity_encoder
        x_polarity = init_block(x_polarity)
        for resnet_block in resnet_blocks:
            x_polarity = resnet_block(x_polarity)
        x_polarity = attn_block(x_polarity)
        x_polarity = self.polarity_merge(x_polarity, x_phase)
        return self.polarity_final(x_polarity)

    def _forward_event(self, x_scaled):
        """Event detection from scaled feature."""
        x_event = self.event_block(x_scaled)
        return self.event_final(x_event)

    def forward(self, x):
        # apply moving normalization
        x = moving_normalize(x, filter=self.moving_norm[0], stride=self.moving_norm[1])
        if self.log_scale:
            x = log_transform(x)

        # store normalized input for polarity/STFT (no clone needed)
        x_norm = x

        # initial convolution
        x = self.init_conv(x)

        # polarity initialization
        x_polarity = None
        if self.add_polarity:
            x_polarity = self.polarity_init(x_norm[:, -1:, :, :])

        # initial resnet block (for memory efficient unet)
        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x)

        # go through the layers of the unet, down and up
        hiddens = []
        stage_indices = []  # indices into hiddens for each stage's final output

        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x)

            for resnet_block in resnet_blocks:
                x = resnet_block(x)
                hiddens.append(x)

            x = attn_block(x)
            hiddens.append(x)
            stage_indices.append(len(hiddens) - 1)

            if exists(post_downsample):
                x = post_downsample(x)

        # STFT processing (updates hiddens in-place for skip connections)
        sgram = None
        x = self.mid_block1(x)
        if self.add_stft:
            x_stft, sgram = self._forward_stft(x_norm, hiddens, stage_indices)
            x = self.mid_merge(x, x_stft)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x)

        # scale mid feature to /16 for event/prompt (compute once, reuse for both)
        out_prompt = None
        x_scaled = None
        if self.add_event or self.add_prompt:
            x_scaled = x
            for scale_layer in self.event_scale_layers:
                x_scaled = scale_layer(x_scaled)
            if self.add_prompt:
                out_prompt = x_scaled

        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim = 1)

        up_hiddens = []

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = add_skip_connection(x)
            x = init_block(x)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x)

            x = attn_block(x)
            up_hiddens.append(x.contiguous())
            x = upsample(x)

        # whether to combine all feature maps from upsample blocks
        x = self.upsample_combiner(x, up_hiddens)

        if exists(self.final_res_block):
            x = self.final_res_block(x)

        out_phase = self.final_conv(x)

        # build output dict
        out = {"phase": out_phase}
        if self.add_polarity:
            out["polarity"] = self._forward_polarity(x_polarity, x)
        if self.add_event:
            out["event"] = self._forward_event(x_scaled)
        if self.add_prompt:
            out["prompt"] = out_prompt
        if sgram is not None:
            out["spectrogram"] = sgram.squeeze(2)
        return out

# null unet

class NullUnet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    def forward(self, x, *args, **kwargs):
        return x

# predefined unets, with configs lining up with hyperparameters in appendix of paper

class BaseUnet64(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 32,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 2, 2, 2),
            layer_attns = False,
            attn_heads = 8,
            ff_mult = 2,
            memory_efficient = False
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class SRUnet256(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 32,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 2, 2, 2),
            layer_attns = False,
            attn_heads = 8,
            ff_mult = 2,
            memory_efficient = True,
            final_resnet_block=False,
            combine_upsample_fmaps = True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class SRUnet1024(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 32,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 2, 2, 2),
            layer_attns = False,
            attn_heads = 8,
            ff_mult = 2,
            memory_efficient = True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})
