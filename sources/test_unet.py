"""Test suite for UNet with domain-specific features."""

import torch
from unet import Unet

# Seismic config: 3-component, 1D temporal processing
SEISMIC_CONFIG = dict(
    dim=32,
    channels=3,
    phase_channels=3,
    dim_mults=(1, 2, 4, 8),
    num_resnet_blocks=(2, 2, 2, 2),
    layer_attns=False,
    attn_heads=8,
    ff_mult=2,
    memory_efficient=True,
    space_stride=1,
    time_stride=4,
    space_kernel=1,
    time_kernel=7,
)

# DAS config: 1-channel, 2D spatial-temporal processing
DAS_CONFIG = dict(
    dim=32,
    channels=1,
    phase_channels=1,
    dim_mults=(1, 2, 4, 8),
    num_resnet_blocks=(2, 2, 2, 2),
    layer_attns=False,
    attn_heads=8,
    ff_mult=2,
    memory_efficient=True,
    space_stride=4,
    time_stride=4,
    space_kernel=7,
    time_kernel=7,
)

# Default uses seismic config for backward compatibility
DEFAULT_CONFIG = SEISMIC_CONFIG


def create_model(data_type='seismic', **overrides):
    """Create a Unet model with appropriate config for data type."""
    base_config = SEISMIC_CONFIG if data_type == 'seismic' else DAS_CONFIG
    config = {**base_config, **overrides}
    return Unet(**config)


def check_output(out, expected, name="Test"):
    """Check output shapes and print results."""
    for key, shape in expected.items():
        if shape is None:
            assert key not in out, f"{name}: {key} should not be in output"
        else:
            assert key in out, f"{name}: missing {key}"
            assert out[key].shape == shape, f"{name}: {key} expected {shape}, got {out[key].shape}"
    print(f"  {name}: PASSED")


def run_test(name, x, expected, data_type='seismic', **model_kwargs):
    """Create model, run forward pass, check outputs."""
    model = create_model(data_type=data_type, **model_kwargs)
    out = model(x)
    check_output(out, expected, name)
    return out


# =============================================================================
# Test Data
# =============================================================================

def get_seismic_data(batch=2, nt=512):
    """Seismic data: (batch, 3, 1, nt) - 3 components, 1 station"""
    return torch.randn(batch, 3, 1, nt)


def get_das_data(batch=2, nx=256, nt=512):
    """DAS data: (batch, 1, nx, nt) - 1 channel, nx stations

    Note: nx=256 is minimum for 4 stages with stride 4.
    With 4 downsamples at stride 4: 256 -> 64 -> 16 -> 4 -> 1
    """
    return torch.randn(batch, 1, nx, nt)


# =============================================================================
# Core Architecture Tests
# =============================================================================

def test_basic():
    """Test basic model without domain features."""
    print("\n" + "=" * 60)
    print("Basic Architecture Tests")
    print("=" * 60)

    # Seismic data
    x = get_seismic_data()
    run_test("Seismic basic", x, {"phase": (2, 3, 1, 512)})

    # DAS data
    x = get_das_data()
    run_test("DAS basic", x, {"phase": (2, 1, 256, 512)}, data_type='das')

    # memory_efficient=False (post-downsample mode)
    x = get_seismic_data()
    run_test("Post-downsample mode", x, {"phase": (2, 3, 1, 512)}, memory_efficient=False)

    print("  All basic tests PASSED")


def test_attention_options():
    """Test attention-related options."""
    print("\n" + "=" * 60)
    print("Attention Options Tests")
    print("=" * 60)

    x = get_seismic_data()

    # Linear attention
    run_test("use_linear_attn=True", x, {"phase": (2, 3, 1, 512)}, use_linear_attn=True)

    # Full attention at last layer
    run_test("layer_attns=(F,F,F,T)", x, {"phase": (2, 3, 1, 512)},
             layer_attns=(False, False, False, True))

    # Mixed attention
    run_test("Mixed attention", x, {"phase": (2, 3, 1, 512)},
             layer_attns=(False, False, False, True),
             use_linear_attn=(True, True, True, False))

    # No middle attention
    run_test("attend_at_middle=False", x, {"phase": (2, 3, 1, 512)}, attend_at_middle=False)

    # Custom attention params
    run_test("attn_dim_head=32", x, {"phase": (2, 3, 1, 512)},
             layer_attns=(False, False, False, True), attn_dim_head=32)

    run_test("layer_attns_depth=2", x, {"phase": (2, 3, 1, 512)},
             layer_attns=(False, False, False, True), layer_attns_depth=2)

    run_test("layer_mid_attns_depth=2", x, {"phase": (2, 3, 1, 512)}, layer_mid_attns_depth=2)

    print("  All attention tests PASSED")


def test_downsampling_options():
    """Test downsampling-related options."""
    print("\n" + "=" * 60)
    print("Downsampling Options Tests")
    print("=" * 60)

    x = get_seismic_data()

    # Cross-embed downsample
    run_test("cross_embed_downsample", x, {"phase": (2, 3, 1, 512)}, cross_embed_downsample=True)

    # Init cross-embed
    run_test("init_cross_embed", x, {"phase": (2, 3, 1, 512)}, init_cross_embed=True)

    # Different stride
    run_test("time_stride=2", x, {"phase": (2, 3, 1, 512)}, time_stride=2)

    # Cross-embed + DAS data
    x = get_das_data()
    run_test("cross_embed + DAS", x, {"phase": (2, 1, 256, 512)}, data_type='das', cross_embed_downsample=True)

    print("  All downsampling tests PASSED")


def test_upsampling_options():
    """Test upsampling-related options."""
    print("\n" + "=" * 60)
    print("Upsampling Options Tests")
    print("=" * 60)

    x = get_seismic_data()

    # Combine upsample feature maps
    run_test("combine_upsample_fmaps", x, {"phase": (2, 3, 1, 512)}, combine_upsample_fmaps=True)

    # Disable pixel shuffle upsample
    run_test("pixel_shuffle=False", x, {"phase": (2, 3, 1, 512)}, pixel_shuffle_upsample=False)

    # DAS with combine fmaps
    x = get_das_data()
    run_test("combine_fmaps + DAS", x, {"phase": (2, 1, 256, 512)}, data_type='das', combine_upsample_fmaps=True)

    print("  All upsampling tests PASSED")


def test_resnet_options():
    """Test ResNet block options."""
    print("\n" + "=" * 60)
    print("ResNet Options Tests")
    print("=" * 60)

    x = get_seismic_data()

    # Skip connection scaling
    run_test("scale_skip_connection=False", x, {"phase": (2, 3, 1, 512)},
             scale_skip_connection=False)

    # No final resnet block
    run_test("final_resnet_block=False", x, {"phase": (2, 3, 1, 512)}, final_resnet_block=False)

    # Dropout
    run_test("dropout=0.1", x, {"phase": (2, 3, 1, 512)}, dropout=0.1)

    # Custom kernel size
    run_test("time_kernel=5", x, {"phase": (2, 3, 1, 512)}, time_kernel=5)

    print("  All ResNet tests PASSED")


def test_conv_options():
    """Test convolution options."""
    print("\n" + "=" * 60)
    print("Convolution Options Tests")
    print("=" * 60)

    x = get_seismic_data()

    # Init conv kernel size (requires init_cross_embed=False)
    run_test("init_conv_kernel_size=5", x, {"phase": (2, 3, 1, 512)},
             init_cross_embed=False, init_conv_kernel_size=5)

    # Final conv kernel size
    run_test("final_conv_kernel_size=5", x, {"phase": (2, 3, 1, 512)}, final_conv_kernel_size=5)

    # Custom init_dim
    run_test("init_dim=64", x, {"phase": (2, 3, 1, 512)}, init_dim=64)

    print("  All convolution tests PASSED")


def test_preprocessing():
    """Test preprocessing options."""
    print("\n" + "=" * 60)
    print("Preprocessing Options Tests")
    print("=" * 60)

    x = get_seismic_data()

    # Log scale
    run_test("log_scale=True", x, {"phase": (2, 3, 1, 512)}, log_scale=True)

    # Custom moving norm
    run_test("moving_norm=(512,128)", x, {"phase": (2, 3, 1, 512)}, moving_norm=(512, 128))

    print("  All preprocessing tests PASSED")


# =============================================================================
# Domain Feature Tests
# =============================================================================

def test_polarity():
    """Test polarity prediction feature."""
    print("\n" + "=" * 60)
    print("Polarity Feature Tests")
    print("=" * 60)

    # Seismic
    x = get_seismic_data()
    run_test("Polarity (seismic)", x,
             {"phase": (2, 3, 1, 512), "polarity": (2, 1, 1, 512)},
             add_polarity=True)

    # DAS
    x = get_das_data()
    run_test("Polarity (DAS)", x,
             {"phase": (2, 1, 256, 512), "polarity": (2, 1, 256, 512)},
             data_type='das', add_polarity=True)

    # Custom channels
    x = get_seismic_data()
    run_test("polarity_channels=2", x,
             {"phase": (2, 3, 1, 512), "polarity": (2, 2, 1, 512)},
             add_polarity=True, polarity_channels=2)

    print("  All polarity tests PASSED")


def test_event():
    """Test event detection feature."""
    print("\n" + "=" * 60)
    print("Event Detection Tests")
    print("=" * 60)

    # Seismic - event at /16 scale
    x = get_seismic_data()
    run_test("Event (seismic)", x,
             {"phase": (2, 3, 1, 512), "event": (2, 1, 1, 32)},
             add_event=True)

    # DAS - event at /16 scale for both spatial and temporal
    x = get_das_data()
    run_test("Event (DAS)", x,
             {"phase": (2, 1, 256, 512), "event": (2, 1, 16, 32)},
             data_type='das', add_event=True)

    # Custom channels
    x = get_seismic_data()
    run_test("event_channels=3", x,
             {"phase": (2, 3, 1, 512), "event": (2, 3, 1, 32)},
             add_event=True, event_channels=3)

    print("  All event tests PASSED")


def test_prompt():
    """Test prompt output feature."""
    print("\n" + "=" * 60)
    print("Prompt Output Tests")
    print("=" * 60)

    mid_dim = 32 * 8  # dim * dim_mults[-1]

    # Seismic
    x = get_seismic_data()
    run_test("Prompt (seismic)", x,
             {"phase": (2, 3, 1, 512), "prompt": (2, mid_dim, 1, 32)},
             add_prompt=True)

    # DAS - prompt at /16 scale for both dimensions
    x = get_das_data()
    run_test("Prompt (DAS)", x,
             {"phase": (2, 1, 256, 512), "prompt": (2, mid_dim, 16, 32)},
             data_type='das', add_prompt=True)

    print("  All prompt tests PASSED")


def test_stft():
    """Test STFT encoder feature (seismic only)."""
    print("\n" + "=" * 60)
    print("STFT Encoder Tests")
    print("=" * 60)

    x = get_seismic_data()

    # Basic STFT
    out = run_test("STFT basic", x, {"phase": (2, 3, 1, 512)}, add_stft=True)
    assert "spectrogram" in out
    assert out["spectrogram"].shape[0] == 2 and out["spectrogram"].shape[1] == 3
    print(f"    Spectrogram shape: {out['spectrogram'].shape}")

    # STFT with different stride
    run_test("STFT + time_stride=2", x, {"phase": (2, 3, 1, 512)},
             add_stft=True, time_stride=2)

    print("  All STFT tests PASSED")


def test_combined_features():
    """Test combinations of domain features."""
    print("\n" + "=" * 60)
    print("Combined Features Tests")
    print("=" * 60)

    mid_dim = 32 * 8

    # All features - seismic
    x = get_seismic_data()
    out = run_test("All features (seismic)", x,
                   {"phase": (2, 3, 1, 512), "polarity": (2, 1, 1, 512),
                    "event": (2, 1, 1, 32), "prompt": (2, mid_dim, 1, 32)},
                   add_stft=True, add_polarity=True, add_event=True, add_prompt=True)
    assert "spectrogram" in out
    print(f"    Spectrogram: {out['spectrogram'].shape}")

    # All features - DAS (no STFT)
    x = get_das_data()
    run_test("All features (DAS)", x,
             {"phase": (2, 1, 256, 512), "polarity": (2, 1, 256, 512),
              "event": (2, 1, 16, 32), "prompt": (2, mid_dim, 16, 32)},
             data_type='das', add_polarity=True, add_event=True, add_prompt=True)

    # Polarity + Event
    x = get_seismic_data()
    run_test("Polarity + Event", x,
             {"phase": (2, 3, 1, 512), "polarity": (2, 1, 1, 512), "event": (2, 1, 1, 32)},
             add_polarity=True, add_event=True)

    # STFT + Event
    run_test("STFT + Event", x,
             {"phase": (2, 3, 1, 512), "event": (2, 1, 1, 32)},
             add_stft=True, add_event=True)

    print("  All combined feature tests PASSED")


# =============================================================================
# Integration Tests
# =============================================================================

def test_all_options_seismic():
    """Test all options together with seismic data."""
    print("\n" + "=" * 60)
    print("Full Integration Test (Seismic)")
    print("=" * 60)

    x = get_seismic_data()
    mid_dim = 32 * 8

    model = create_model(
        # Attention
        layer_attns=(False, False, False, True),
        use_linear_attn=(True, True, True, False),
        attn_dim_head=32,
        layer_attns_depth=2,
        # Downsampling
        cross_embed_downsample=True,
        # Upsampling
        combine_upsample_fmaps=True,
        pixel_shuffle_upsample=True,
        # Domain features
        add_stft=True,
        add_polarity=True,
        add_event=True,
        add_prompt=True,
    )
    out = model(x)

    expected = {
        "phase": (2, 3, 1, 512),
        "polarity": (2, 1, 1, 512),
        "event": (2, 1, 1, 32),
        "prompt": (2, mid_dim, 1, 32),
    }
    check_output(out, expected, "All options (seismic)")
    assert "spectrogram" in out
    print(f"    Spectrogram: {out['spectrogram'].shape}")
    print("  Full integration test (seismic) PASSED")


def test_all_options_das():
    """Test all options together with DAS data."""
    print("\n" + "=" * 60)
    print("Full Integration Test (DAS)")
    print("=" * 60)

    x = get_das_data()
    mid_dim = 32 * 8

    model = create_model(
        data_type='das',
        # Attention
        layer_attns=(False, False, False, True),
        use_linear_attn=(True, True, True, False),
        # Downsampling
        cross_embed_downsample=True,
        # Upsampling
        combine_upsample_fmaps=True,
        # Domain features (no STFT for DAS)
        add_polarity=True,
        add_event=True,
        add_prompt=True,
    )
    out = model(x)

    expected = {
        "phase": (2, 1, 256, 512),
        "polarity": (2, 1, 256, 512),
        "event": (2, 1, 16, 32),
        "prompt": (2, mid_dim, 16, 32),
    }
    check_output(out, expected, "All options (DAS)")
    print("  Full integration test (DAS) PASSED")


def test_output_channels():
    """Test custom output channel configurations."""
    print("\n" + "=" * 60)
    print("Output Channel Configuration Tests")
    print("=" * 60)

    x = get_seismic_data()

    # Default channels
    run_test("Default channels (3,1,1)", x,
             {"phase": (2, 3, 1, 512), "polarity": (2, 1, 1, 512), "event": (2, 1, 1, 32)},
             add_polarity=True, add_event=True)

    # Custom channels
    run_test("Custom channels (4,2,3)", x,
             {"phase": (2, 4, 1, 512), "polarity": (2, 2, 1, 512), "event": (2, 3, 1, 32)},
             phase_channels=4, polarity_channels=2, event_channels=3,
             add_polarity=True, add_event=True)

    print("  All output channel tests PASSED")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Core architecture tests
    test_basic()
    test_attention_options()
    test_downsampling_options()
    test_upsampling_options()
    test_resnet_options()
    test_conv_options()
    test_preprocessing()

    # Domain feature tests
    test_polarity()
    test_event()
    test_prompt()
    test_stft()
    test_combined_features()

    # Integration tests
    test_all_options_seismic()
    test_all_options_das()
    test_output_channels()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
