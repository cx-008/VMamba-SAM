from dataclasses import dataclass
from typing import Optional


@dataclass
class MambaAblationConfig:
    name: str
    description: str
    model_type: str = "vmamba_sam"
    use_mamba: bool = True
    use_skip_connections: bool = True
    use_auxiliary_loss: bool = True
    use_msca: bool = True
    use_csfa: bool = True
    use_mamba_adapter: bool = True
    d_state: int = 4
    epochs: int = 50
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    checkpoint_path: Optional[str] = None
    save_interval: int = 10
    gradient_clip_max_norm: float = 1.0
    accumulation_steps: int = 1


def _cfg(name, description, **kwargs):
    return MambaAblationConfig(name=name, description=description, **kwargs)


# Only the VMamba-SAM variants used by the paper are retained.
MAMBA_ABLATION_CONFIGS = {
    "baseline": _cfg("baseline", "VMamba-SAM without Mamba blocks", use_mamba=False),
    "vmamba_sam": _cfg("vmamba_sam", "Complete VMamba-SAM"),
    "no_skip": _cfg("no_skip", "VMamba-SAM without skip connections", use_skip_connections=False),
    "no_aux": _cfg("no_aux", "VMamba-SAM without auxiliary supervision", use_auxiliary_loss=False),
    "no_msca": _cfg("no_msca", "VMamba-SAM without MSCA", use_msca=False),
    "no_csfa": _cfg("no_csfa", "VMamba-SAM without CSFA", use_csfa=False),
    "no_adapter": _cfg("no_adapter", "VMamba-SAM without Mamba adapter", use_mamba_adapter=False),
    "d_state_4": _cfg("d_state_4", "VMamba-SAM with d_state=4", d_state=4),
    "d_state_8": _cfg("d_state_8", "VMamba-SAM with d_state=8", d_state=8),
    "d_state_16": _cfg("d_state_16", "VMamba-SAM with d_state=16", d_state=16),
}

# Compatibility aliases for commands/checkpoints produced by the previous scripts.
_ALIASES = {
    "baseline_no_mamba": "baseline",
    "full_mamba_model": "vmamba_sam",
    "mamba_v2_baseline": "baseline",
    "mamba_v2_full": "vmamba_sam",
    "mamba_v2_no_skip": "no_skip",
    "mamba_v2_no_aux": "no_aux",
    "mamba_v2_d_state_4": "d_state_4",
    "mamba_v2_d_state_8": "d_state_8",
    "mamba_v2_d_state_16": "d_state_16",
}
MAMBA_ABLATION_ORDER = list(MAMBA_ABLATION_CONFIGS)


def get_mamba_config_by_name(name):
    key = _ALIASES.get(name, name)
    if key not in MAMBA_ABLATION_CONFIGS:
        choices = ", ".join(MAMBA_ABLATION_ORDER)
        raise KeyError(f"Unknown VMamba-SAM config '{name}'. Available: {choices}")
    # Return a fresh object so command-line overrides do not mutate the registry.
    return MambaAblationConfig(**vars(MAMBA_ABLATION_CONFIGS[key]))


def get_mamba_configs_by_order():
    return [get_mamba_config_by_name(name) for name in MAMBA_ABLATION_ORDER]


def print_mamba_ablation_summary():
    print("\nVMamba-SAM ablation configurations:")
    for name in MAMBA_ABLATION_ORDER:
        print(f"  {name:<12} {MAMBA_ABLATION_CONFIGS[name].description}")
