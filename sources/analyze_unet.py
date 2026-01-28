"""
UNet Model Analysis Script

Analyzes model parameters, FLOPs, memory usage, and layer-wise statistics.
Supports seismic (3, 1, T) and DAS (1, H, T) input formats.
"""

import torch
from torch import nn
from collections import OrderedDict

from unet import Unet


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_module(model):
    """Count parameters grouped by top-level module."""
    param_counts = OrderedDict()
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if count > 0:
            param_counts[name] = count
    return param_counts


def estimate_flops(model, x):
    """Estimate FLOPs using forward hooks."""
    total = [0]

    def hook(module, inp, out):
        if isinstance(out, dict):
            return
        if isinstance(module, nn.Conv2d):
            b, c, h, w = out.shape
            k = module.kernel_size
            kh, kw = (k, k) if isinstance(k, int) else k
            total[0] += 2 * kh * kw * (module.in_channels // module.groups) * c * h * w
        elif isinstance(module, nn.Linear):
            total[0] += 2 * module.in_features * module.out_features * inp[0].numel() // module.in_features

    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return total[0]


def estimate_flops_by_module(model, x):
    """Estimate FLOPs grouped by top-level module."""
    flops_by_module = OrderedDict()

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, dict):
                return
            f = 0
            if isinstance(module, nn.Conv2d):
                b, c, h, w = out.shape
                k = module.kernel_size
                kh, kw = (k, k) if isinstance(k, int) else k
                f = 2 * kh * kw * (module.in_channels // module.groups) * c * h * w
            elif isinstance(module, nn.Linear):
                f = 2 * module.in_features * module.out_features * inp[0].numel() // module.in_features
            if f > 0:
                top = name.split('.')[0]
                flops_by_module[top] = flops_by_module.get(top, 0) + f
        return hook

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return flops_by_module


def measure_activation_memory(model, x):
    """Measure total activation memory during forward pass."""
    sizes = []

    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            sizes.append(out.numel() * out.element_size())

    hooks = [m.register_forward_hook(hook) for m in model.modules()]
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return sum(sizes)


def fmt(n):
    """Format large numbers with K/M/G suffix."""
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(int(n))


def analyze_model(config, input_shape, name="Model"):
    """Analyze a single model configuration."""
    try:
        model = Unet(**config)
    except Exception as e:
        return {'name': name, 'error': str(e)}

    x = torch.randn(1, *input_shape)
    params = count_parameters(model)
    flops = estimate_flops(model, x)
    memory = measure_activation_memory(model, x)

    model.eval()
    with torch.no_grad():
        out = model(x)
    output_shapes = {k: tuple(v.shape) for k, v in out.items()}

    return {
        'name': name,
        'params': params,
        'flops': flops,
        'memory': memory,
        'output_shapes': output_shapes,
        'module_params': count_parameters_by_module(model),
    }


# Model size configurations
MODEL_CONFIGS = {
    'tiny': {
        'dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'num_resnet_blocks': (1, 1, 1, 1),
        'layer_attns': (False, False, False, False),
        'attn_heads': 4,
        'attn_dim_head': 32,
        'ff_mult': 2,
    },
    'small': {
        'dim': 64,
        'dim_mults': (1, 2, 4, 8),
        'num_resnet_blocks': (1, 2, 2, 2),
        'layer_attns': (False, False, False, True),
        'attn_heads': 8,
        'attn_dim_head': 64,
        'ff_mult': 2,
    },
    'base': {
        'dim': 128,
        'dim_mults': (1, 2, 4, 8),
        'num_resnet_blocks': (2, 2, 2, 2),
        'layer_attns': (False, False, True, True),
        'attn_heads': 8,
        'attn_dim_head': 64,
        'ff_mult': 2,
    },
}

# Data type configurations
DATA_CONFIGS = {
    'seismic': {
        'input_shape': (3, 1, 4096),
        'channels': 3,
        'phase_channels': 3,
        'space_stride': 1,
        'time_stride': 4,
        'space_kernel': 1,
        'time_kernel': 7,
        'freq_stride': 2,
        'freq_kernel': 3,
    },
    'das': {
        'input_shape': (1, 256, 256),  # minimal for 4-stage stride-4: need 256 each dim
        'channels': 1,
        'phase_channels': 1,
        'space_stride': 4,
        'time_stride': 4,
        'space_kernel': 7,
        'time_kernel': 7,
        'freq_stride': 2,
        'freq_kernel': 3,
        'moving_norm': (64, 16),  # smaller filter for short time series
    },
}

# STFT configurations to test (n_fft, dim_divisor, freq_stride)
STFT_CONFIGS = [
    # (name, n_fft, dim_divisor, freq_stride)
    ('default', 33, 1, 1),
    ('light', 33, 2, 2),
    ('lighter', 17, 2, 2),
    ('lightest', 17, 4, 2),
]


def analyze_stft_overhead(model_size='tiny', data_type='seismic'):
    """Analyze STFT overhead with different configurations."""
    data_config = DATA_CONFIGS[data_type]
    if data_config['space_stride'] != 1:
        print(f"\nSkipping STFT analysis for {data_type} (space_stride != 1)")
        return
    input_shape = data_config['input_shape']
    base_config = {
        'channels': data_config['channels'],
        'phase_channels': data_config['phase_channels'],
        'space_stride': data_config['space_stride'],
        'time_stride': data_config['time_stride'],
        'space_kernel': data_config['space_kernel'],
        'time_kernel': data_config['time_kernel'],
        **({'moving_norm': data_config['moving_norm']} if 'moving_norm' in data_config else {}),
        **MODEL_CONFIGS[model_size],
    }

    # Baseline without STFT
    baseline = analyze_model(base_config, input_shape, 'baseline')
    if 'error' in baseline:
        print(f"Error creating baseline: {baseline['error']}")
        return

    print(f"\n{'='*90}")
    print(f"STFT Overhead Analysis: {model_size.upper()} model, {data_type.upper()} data")
    print(f"{'='*90}")
    print(f"Input: {input_shape}")
    print(f"Baseline: {fmt(baseline['params'])} params, {fmt(baseline['flops'])} FLOPs, {fmt(baseline['memory'])} memory")
    print()
    print(f"{'Config':<12} {'n_fft':>5} {'div':>4} {'f_ds':>4} | {'Params':>8} {'FLOPs':>10} {'Memory':>10} | {'P%':>5} {'F%':>5} {'M%':>5} | OK")
    print('-' * 90)

    for name, n_fft, div, freq_ds in STFT_CONFIGS:
        config = {
            **base_config,
            'add_stft': True,
            'stft_n_fft': n_fft,
            'stft_dim_divisor': div,
            'freq_stride': freq_ds,
        }
        r = analyze_model(config, input_shape, name)
        if 'error' in r:
            print(f"{name:<12} {n_fft:>5} {div:>4} {freq_ds:>4} | ERROR: {r['error']}")
            continue

        p_pct = 100 * (r['params'] - baseline['params']) / baseline['params']
        f_pct = 100 * (r['flops'] - baseline['flops']) / baseline['flops']
        m_pct = 100 * (r['memory'] - baseline['memory']) / baseline['memory']
        ok = '✓' if p_pct < 100 and f_pct < 100 and m_pct < 100 else ''

        print(f"{name:<12} {n_fft:>5} {div:>4} {freq_ds:>4} | {fmt(r['params']):>8} {fmt(r['flops']):>10} {fmt(r['memory']):>10} | {p_pct:>4.0f}% {f_pct:>4.0f}% {m_pct:>4.0f}% | {ok}")


def analyze_memory_efficient(model_size='tiny', data_type='seismic'):
    """Compare default vs memory_efficient mode."""
    data_config = DATA_CONFIGS[data_type]
    input_shape = data_config['input_shape']
    base_config = {
        'channels': data_config['channels'],
        'phase_channels': data_config['phase_channels'],
        'space_stride': data_config['space_stride'],
        'time_stride': data_config['time_stride'],
        'space_kernel': data_config['space_kernel'],
        'time_kernel': data_config['time_kernel'],
        **({'moving_norm': data_config['moving_norm']} if 'moving_norm' in data_config else {}),
        **MODEL_CONFIGS[model_size],
    }

    print(f"\n{'='*80}")
    print(f"Memory Efficient Mode: {model_size.upper()} model, {data_type.upper()} data")
    print(f"{'='*80}")

    for mode_name, memory_efficient in [('default', False), ('memory_efficient', True)]:
        config = {**base_config, 'memory_efficient': memory_efficient}
        r = analyze_model(config, input_shape, mode_name)
        if 'error' in r:
            print(f"{mode_name}: ERROR - {r['error']}")
            continue

        print(f"\n{mode_name}:")
        print(f"  Params: {fmt(r['params'])}")
        print(f"  FLOPs:  {fmt(r['flops'])}")
        print(f"  Memory: {fmt(r['memory'])}")


def analyze_all_features(model_size='tiny', data_type='seismic'):
    """Analyze overhead of all features."""
    data_config = DATA_CONFIGS[data_type]
    input_shape = data_config['input_shape']
    base_config = {
        'channels': data_config['channels'],
        'phase_channels': data_config['phase_channels'],
        'space_stride': data_config['space_stride'],
        'time_stride': data_config['time_stride'],
        'space_kernel': data_config['space_kernel'],
        'time_kernel': data_config['time_kernel'],
        **({'moving_norm': data_config['moving_norm']} if 'moving_norm' in data_config else {}),
        **MODEL_CONFIGS[model_size],
    }

    print(f"\n{'='*90}")
    print(f"Feature Overhead: {model_size.upper()} model, {data_type.upper()} data")
    print(f"{'='*90}")

    # Baseline
    baseline = analyze_model(base_config, input_shape, 'baseline')
    if 'error' in baseline:
        print(f"Error: {baseline['error']}")
        return

    print(f"Baseline: {fmt(baseline['params'])} params, {fmt(baseline['flops'])} FLOPs, {fmt(baseline['memory'])} memory")
    print()
    print(f"{'Feature':<20} | {'Params':>8} {'FLOPs':>10} {'Memory':>10} | {'P%':>5} {'F%':>5} {'M%':>5} | Outputs")
    print('-' * 90)

    # STFT only works with space_stride=1 (seismic), skip for DAS
    has_stft = data_config['space_stride'] == 1
    features = [
        ('+ polarity', {'add_polarity': True}),
        ('+ event', {'add_event': True}),
    ]
    if has_stft:
        features = [
            ('+ stft (light)', {'add_stft': True, 'stft_n_fft': 33, 'stft_dim_divisor': 2, 'freq_stride': 2}),
        ] + features + [
            ('+ all', {'add_stft': True, 'stft_n_fft': 33, 'stft_dim_divisor': 2, 'freq_stride': 2, 'add_polarity': True, 'add_event': True}),
        ]
    else:
        features.append(('+ all (no stft)', {'add_polarity': True, 'add_event': True}))

    for name, feature_config in features:
        config = {**base_config, **feature_config}
        r = analyze_model(config, input_shape, name)
        if 'error' in r:
            print(f"{name:<20} | ERROR: {r['error']}")
            continue

        p_pct = 100 * (r['params'] - baseline['params']) / baseline['params']
        f_pct = 100 * (r['flops'] - baseline['flops']) / baseline['flops']
        m_pct = 100 * (r['memory'] - baseline['memory']) / baseline['memory']
        outputs = ', '.join(r['output_shapes'].keys())

        print(f"{name:<20} | {fmt(r['params']):>8} {fmt(r['flops']):>10} {fmt(r['memory']):>10} | {p_pct:>4.0f}% {f_pct:>4.0f}% {m_pct:>4.0f}% | {outputs}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze UNet model configurations")
    parser.add_argument('--size', choices=['tiny', 'small', 'base', 'all'], default='tiny')
    parser.add_argument('--data', choices=['seismic', 'das', 'both'], default='seismic')
    parser.add_argument('--analysis', choices=['stft', 'memory', 'features', 'all'], default='all')

    args = parser.parse_args()

    sizes = [args.size] if args.size != 'all' else ['tiny', 'small', 'base']
    data_types = [args.data] if args.data != 'both' else ['seismic', 'das']

    for size in sizes:
        for data_type in data_types:
            if args.analysis in ['stft', 'all']:
                analyze_stft_overhead(size, data_type)
            if args.analysis in ['memory', 'all']:
                analyze_memory_efficient(size, data_type)
            if args.analysis in ['features', 'all']:
                analyze_all_features(size, data_type)


if __name__ == "__main__":
    main()
