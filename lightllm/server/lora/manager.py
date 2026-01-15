"""
LoRA Adapter Manager for LightLLM Server

Manages LoRA adapters for detached serving:
- Loading adapters from disk
- Caching adapters in memory
- GPU/CPU swapping
- Per-batch adapter selection
"""
import os
import json
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import torch

import sys
sys.path.insert(0, "/home/shufan/LightLLM")

logger = None


def init_lora_logger():
    """Initialize logger lazily"""
    global logger
    if logger is None:
        from lightllm.utils.log_utils import init_logger
        logger = init_logger(__name__)


@dataclass
class LoRAAdapterInfo:
    """Information about a loaded LoRA adapter"""
    adapter_id: str
    adapter_dir: str
    lora_rank: int
    lora_alpha: float
    num_layers: int
    is_on_gpu: bool = False
    memory_usage: int = 0  # in bytes


class LoRAManager:
    """
    Manager for LoRA adapters in detached serving mode.

    Features:
    - Load adapters from disk on demand
    - Cache loaded adapters in memory
    - GPU/CPU swapping for memory efficiency
    - Thread-safe operations
    """

    def __init__(
        self,
        max_adapters: int = 1024,
        swap_enabled: bool = True,
        device: str = "cuda",
    ):
        self.max_adapters = max_adapters
        self.swap_enabled = swap_enabled
        self.device = device

        # Cached adapters: adapter_id -> adapter object
        self._adapters: Dict[str, Any] = {}
        self._adapter_info: Dict[str, LoRAAdapterInfo] = {}

        # Currently active adapter (for the running batch)
        self._active_adapter_id: Optional[str] = None
        self._active_adapter: Optional[Any] = None

        # Dispatchers per layer
        self._dispatchers: Dict[int, Any] = {}

        # Lock for thread safety
        self._lock = threading.Lock()

        init_lora_logger()

    def _read_adapter_config(self, adapter_dir: str) -> Dict[str, Any]:
        """
        Read LoRA adapter config from adapter_config.json.

        Args:
            adapter_dir: Path to the adapter directory

        Returns:
            Dictionary with lora_rank (r), lora_alpha, and other config values
        """
        config_path = os.path.join(adapter_dir, "adapter_config.json")
        if not os.path.exists(config_path):
            logger.warning(f"adapter_config.json not found in {adapter_dir}")
            return {}

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Read adapter config from {config_path}: r={config.get('r')}, lora_alpha={config.get('lora_alpha')}")
            return config
        except Exception as e:
            logger.error(f"Failed to read adapter config from {config_path}: {e}")
            return {}

    def register_lora_dir(self, lora_dir: str, adapter_id: str = "default") -> LoRAAdapterInfo:
        """
        Register a LoRA adapter directory.

        Args:
            lora_dir: Path to LoRA adapter directory
            adapter_id: Unique identifier for this adapter

        Returns:
            LoRAAdapterInfo with adapter details
        """
        with self._lock:
            if adapter_id in self._adapter_info:
                logger.info(f"Adapter {adapter_id} already registered, updating...")
                info = self._adapter_info[adapter_id]
                info.adapter_dir = lora_dir
            else:
                # Read adapter config to get lora_rank and lora_alpha
                adapter_config = self._read_adapter_config(lora_dir)
                lora_rank = adapter_config.get("r", 0)
                lora_alpha = adapter_config.get("lora_alpha", 1.0)

                # Create info with values from adapter config
                info = LoRAAdapterInfo(
                    adapter_id=adapter_id,
                    adapter_dir=lora_dir,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    num_layers=0,
                )
                self._adapter_info[adapter_id] = info

            logger.info(f"Registered LoRA adapter: {adapter_id} -> {lora_dir}")
            return info

    def load_adapter(self, adapter_id: str, network_config: Dict[str, Any]) -> bool:
        """
        Load a LoRA adapter into memory.

        Args:
            adapter_id: ID of the adapter to load
            network_config: Model network configuration

        Returns:
            True if loaded successfully
        """
        with self._lock:
            if adapter_id not in self._adapter_info:
                logger.error(f"Adapter {adapter_id} not registered")
                return False

            if adapter_id in self._adapters:
                logger.info(f"Adapter {adapter_id} already loaded")
                return True

            # Check if we need to evict an adapter
            if len(self._adapters) >= self.max_adapters:
                # Evict least recently used (simple: first one)
                evict_id = next(iter(self._adapters))
                self._evict_adapter(evict_id)

            # Load the adapter
            info = self._adapter_info[adapter_id]
            try:
                # Build effective network config with adapter-specific values
                # Values from adapter_config.json take precedence over CLI args
                adapter_network_config = dict(network_config)
                adapter_network_config["lora_rank"] = info.lora_rank if info.lora_rank > 0 else network_config.get("lora_rank", 16)
                adapter_network_config["lora_alpha"] = info.lora_alpha if info.lora_alpha > 0 else network_config.get("lora_alpha", 16.0)

                from lightllm.models.qwen3_vl.layer_weights.lora_layer_weight import Qwen3VLLoRAAdapter
                adapter = Qwen3VLLoRAAdapter(
                    adapter_dir=info.adapter_dir,
                    network_config=adapter_network_config,
                    data_type=torch.float16,
                    device=self.device,
                    swap=self.swap_enabled,
                )

                self._adapters[adapter_id] = adapter
                info.lora_rank = adapter_network_config.get("lora_rank", 0)
                info.lora_alpha = adapter_network_config.get("lora_alpha", 1.0)
                info.num_layers = adapter_network_config.get("num_hidden_layers", 0)

                logger.info(f"Loaded LoRA adapter: {adapter_id} (rank={info.lora_rank}, layers={info.num_layers})")
                return True

            except Exception as e:
                logger.error(f"Failed to load adapter {adapter_id}: {e}")
                return False

    def _evict_adapter(self, adapter_id: str):
        """Evict an adapter from memory"""
        if adapter_id in self._adapters:
            adapter = self._adapters[adapter_id]
            # Offload from GPU if needed
            if hasattr(adapter, 'offload_from_gpu'):
                adapter.offload_from_gpu()
            del self._adapters[adapter_id]
            self._adapter_info[adapter_id].is_on_gpu = False
            logger.info(f"Evicted adapter: {adapter_id}")

    def load_to_gpu(self, adapter_id: str):
        """Load adapter weights from CPU to GPU"""
        with self._lock:
            if adapter_id in self._adapters:
                adapter = self._adapters[adapter_id]
                adapter.load_to_gpu()
                self._adapter_info[adapter_id].is_on_gpu = True
                logger.debug(f"Loaded adapter {adapter_id} to GPU")

    def offload_from_gpu(self, adapter_id: str):
        """Offload adapter weights from GPU to CPU"""
        with self._lock:
            if adapter_id in self._adapters:
                adapter = self._adapters[adapter_id]
                adapter.offload_from_gpu()
                self._adapter_info[adapter_id].is_on_gpu = False
                logger.debug(f"Offloaded adapter {adapter_id} from GPU")

    def set_active_adapter(self, adapter_id: str) -> bool:
        """
        Set the active adapter for the current batch.

        Args:
            adapter_id: ID of the adapter to use

        Returns:
            True if successful
        """
        with self._lock:
            if adapter_id not in self._adapters:
                logger.error(f"Adapter {adapter_id} not loaded")
                return False

            self._active_adapter_id = adapter_id
            self._active_adapter = self._adapters[adapter_id]
            logger.debug(f"Active adapter set to: {adapter_id}")
            return True

    def clear_active_adapter(self):
        """Clear the active adapter"""
        with self._lock:
            self._active_adapter_id = None
            self._active_adapter = None
            logger.debug("Active adapter cleared")

    def get_adapter(self, adapter_id: Optional[str] = None) -> Optional[Any]:
        """Get an adapter by ID, or the active adapter"""
        with self._lock:
            if adapter_id is None:
                return self._active_adapter
            return self._adapters.get(adapter_id)

    def get_dispatchers(self) -> Dict[int, Any]:
        """Get dispatchers for all layers"""
        with self._lock:
            return self._dispatchers.copy()

    def set_dispatchers(self, dispatchers: Dict[int, Any]):
        """Set dispatchers for all layers"""
        with self._lock:
            self._dispatchers = dispatchers

    def list_adapters(self) -> List[Dict[str, Any]]:
        """List all registered adapters"""
        with self._lock:
            result = []
            for adapter_id, info in self._adapter_info.items():
                result.append({
                    "id": info.adapter_id,
                    "path": info.adapter_dir,
                    "lora_rank": info.lora_rank,
                    "lora_alpha": info.lora_alpha,
                    "num_layers": info.num_layers,
                    "is_on_gpu": info.is_on_gpu,
                    "is_loaded": adapter_id in self._adapters,
                })
            return result

    def get_adapter_count(self) -> int:
        """Get number of loaded adapters"""
        with self._lock:
            return len(self._adapters)

    def is_adapter_loaded(self, adapter_id: str) -> bool:
        """Check if an adapter is loaded"""
        with self._lock:
            return adapter_id in self._adapters

    def cleanup(self):
        """Clean up all adapters"""
        with self._lock:
            for adapter_id in list(self._adapters.keys()):
                self._evict_adapter(adapter_id)
            self._dispatchers.clear()
            self._active_adapter_id = None
            self._active_adapter = None
            logger.info("LoRA manager cleanup complete")


# Global LoRA manager instance
_lora_manager: Optional[LoRAManager] = None


def get_lora_manager() -> LoRAManager:
    """Get the global LoRA manager instance"""
    global _lora_manager
    if _lora_manager is None:
        _lora_manager = LoRAManager()
    return _lora_manager


def init_lora_manager(
    max_adapters: int = 1024,
    swap_enabled: bool = True,
    device: str = "cuda",
) -> LoRAManager:
    """Initialize the global LoRA manager"""
    global _lora_manager
    _lora_manager = LoRAManager(
        max_adapters=max_adapters,
        swap_enabled=swap_enabled,
        device=device,
    )
    return _lora_manager


def shutdown_lora_manager():
    """Shutdown the global LoRA manager"""
    global _lora_manager
    if _lora_manager is not None:
        _lora_manager.cleanup()
        _lora_manager = None
