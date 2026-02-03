from dataclasses import dataclass, field
from typing import Optional


class ComputeLocation:
    """Enum for LoRA compute location."""
    GPU = "gpu"
    CPU = "cpu"
    OFF = "off"

    @classmethod
    def from_string(cls, value: str) -> "ComputeLocation":
        """Convert string to ComputeLocation."""
        value = value.strip().lower()
        if value == "gpu":
            return cls.GPU
        elif value == "cpu":
            return cls.CPU
        elif value == "off":
            return cls.OFF
        else:
            raise ValueError(f"Invalid compute location: {value}. Valid: gpu, cpu, off")


@dataclass
class LoRAComputeConfig:
    """Configuration for where to store and compute LoRA for each component.

    Attributes:
        vl_storage: Vision-language adapter LoRA weight storage location
        vl_compute: Vision-language adapter LoRA compute location
        attn_storage: Attention projection LoRA weight storage location
        attn_compute: Attention projection LoRA compute location
        moe_storage: MoE MLP LoRA weight storage location
        moe_compute: MoE MLP LoRA compute location

    Supported combinations per component:
        - storage=gpu, compute=gpu: Default, weights on GPU, compute on GPU
        - storage=gpu, compute=off: Weights on GPU, LoRA disabled
        - storage=cpu, compute=gpu: Weights on CPU, transfer to GPU for compute
        - storage=cpu, compute=cpu: Weights on CPU, compute on CPU
        - storage=cpu, compute=off: Weights on CPU, LoRA disabled

    Note: GPU storage + CPU compute is NOT supported.
    """
    vl_storage: str = field(default=ComputeLocation.GPU)
    vl_compute: str = field(default=ComputeLocation.GPU)
    attn_storage: str = field(default=ComputeLocation.GPU)
    attn_compute: str = field(default=ComputeLocation.GPU)
    moe_storage: str = field(default=ComputeLocation.GPU)
    moe_compute: str = field(default=ComputeLocation.GPU)

    @classmethod
    def from_string(cls, config_str: str) -> "LoRAComputeConfig":
        """Parse config string.

        Format: 'vl_storage:cpu,vl_compute:gpu,attn_storage:gpu,attn_compute:gpu,moe_storage:cpu,moe_compute:cpu'

        Args:
            config_str: Comma-separated component:location pairs

        Returns:
            LoRAComputeConfig instance

        Raises:
            ValueError: If format is invalid
        """
        # Default: all on GPU
        config = cls()

        if not config_str or config_str.strip() == "":
            return config

        # Initialize with defaults
        vl_storage = ComputeLocation.GPU
        vl_compute = ComputeLocation.GPU
        attn_storage = ComputeLocation.GPU
        attn_compute = ComputeLocation.GPU
        moe_storage = ComputeLocation.GPU
        moe_compute = ComputeLocation.GPU

        # Parse format
        for part in config_str.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"Invalid format: '{part}'. Expected 'component:value'")

            key, value = part.split(":", 1)
            key = key.strip().lower()
            value = value.strip().lower()

            if value not in ("gpu", "cpu", "off"):
                raise ValueError(f"Invalid value: '{value}'. Valid: gpu, cpu, off")

            # Check for _storage or _compute suffix
            if key.endswith("_storage"):
                base_key = key[:-len("_storage")]
                if base_key == "vl":
                    vl_storage = value
                elif base_key == "attn":
                    attn_storage = value
                elif base_key == "moe":
                    moe_storage = value
                else:
                    raise ValueError(f"Unknown component: '{base_key}'. Valid: vl, attn, moe")
            elif key.endswith("_compute"):
                base_key = key[:-len("_compute")]
                if base_key == "vl":
                    vl_compute = value
                elif base_key == "attn":
                    attn_compute = value
                elif base_key == "moe":
                    moe_compute = value
                else:
                    raise ValueError(f"Unknown component: '{base_key}'. Valid: vl, attn, moe")
            else:
                # Single value format: 'vl:cpu' sets both storage and compute
                if key not in ("vl", "attn", "moe"):
                    raise ValueError(f"Unknown component: '{key}'. Valid: vl, attn, moe or use vl_storage/vl_compute format")
                if key == "vl":
                    vl_storage = vl_compute = value
                elif key == "attn":
                    attn_storage = attn_compute = value
                elif key == "moe":
                    moe_storage = moe_compute = value

        return cls(
            vl_storage=vl_storage, vl_compute=vl_compute,
            attn_storage=attn_storage, attn_compute=attn_compute,
            moe_storage=moe_storage, moe_compute=moe_compute
        )

    def should_compute_on_cpu(self, component: str) -> bool:
        """Check if a component should be computed on CPU."""
        attr = f"{component}_compute"
        return getattr(self, attr, ComputeLocation.GPU) == ComputeLocation.CPU

    def is_enabled(self, component: str) -> bool:
        """Check if a component's LoRA is enabled (not OFF)."""
        attr = f"{component}_compute"
        return getattr(self, attr, ComputeLocation.GPU) != ComputeLocation.OFF

    def get_storage_device(self, component: str) -> str:
        """Get the storage device for a component."""
        attr = f"{component}_storage"
        return getattr(self, attr, ComputeLocation.GPU)

    def get_compute_device(self, component: str) -> str:
        """Get the compute device for a component."""
        attr = f"{component}_compute"
        return getattr(self, attr, ComputeLocation.GPU)

    def __repr__(self) -> str:
        return (
            f"LoRAComputeConfig("
            f"vl_storage={self.vl_storage}, vl_compute={self.vl_compute}, "
            f"attn_storage={self.attn_storage}, attn_compute={self.attn_compute}, "
            f"moe_storage={self.moe_storage}, moe_compute={self.moe_compute})"
        )
