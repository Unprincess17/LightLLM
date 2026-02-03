"""
LoRA Dispatch for Qwen3-MOE MoE Layers

This module provides functions for applying LoRA to MoE layers at three points:
1. moe_gate - Router/logits modification
2. w1 - Fused Gate + Up projection
3. w2 - Down projection

Each injection point can be enabled independently for flexible LoRA composition.
"""
import torch
from typing import Dict, Optional, Any

from lightllm.server.core.objs.lora_compute_config import LoRAComputeConfig


class Qwen3MOELoRADispatcher:
    """Dispatcher for applying LoRA to Qwen3-MOE layers.

    Manages multiple adapters and applies LoRA during inference.
    This is the main interface for detached LoRA serving with MoE.

    LoRA injection points:
    - moe_gate: Router/logits modification (affects expert selection)
    - w1: Fused gate+up projection (single LoRA for both)
    - w2: Down projection (applied before moe_sum_reduce)

    CPU Offload:
    - Set compute_on_cpu=True to run LoRA computation on CPU
    - LoRA weights stay on CPU, only input/output transferred
    - Reduces GPU memory bandwidth overhead
    """

    def __init__(
        self,
        lora_rank: int = 64,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        lora_compute_config: Optional[LoRAComputeConfig] = None
    ):
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0
        self.lora_compute_config = lora_compute_config or LoRAComputeConfig()

        # Active adapter (loaded for current batch)
        self.active_adapter: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
        self.active_adapter_dir: Optional[str] = None

        # Check if any LoRA is enabled
        self.has_lora = lora_rank > 0

        # Memory buffers for efficient computation (GPU)
        # moe_gate LoRA buffers
        self.gate_A_buffer: Optional[torch.Tensor] = None
        self.gate_B_buffer: Optional[torch.Tensor] = None

        # w1 LoRA buffers (fused gate+up)
        self.w1_A_buffer: Optional[torch.Tensor] = None
        self.w1_B_buffer: Optional[torch.Tensor] = None

        # w2 LoRA buffers (down projection)
        self.w2_A_buffer: Optional[torch.Tensor] = None
        self.w2_B_buffer: Optional[torch.Tensor] = None

    def load_adapter(self, adapter_weights: Dict[int, Dict[str, torch.Tensor]]):
        """Load adapter weights into dispatcher.

        Args:
            adapter_weights: Dict mapping layer_id -> {gate_lora_A, gate_lora_B, w1_lora_A, ...}
        """
        if not self.has_lora:
            return

        self.active_adapter = adapter_weights

        # Extract A matrices for efficient access
        first_layer = adapter_weights.get(0, {})
        if not first_layer:
            return

        # Get shape from first weight
        sample_weight = first_layer.get("gate_lora_A") or first_layer.get("w1_lora_A")
        if sample_weight is None:
            return

        batch_size = sample_weight.shape[0]
        rank = sample_weight.shape[1]

        # Store buffers on GPU
        self.gate_A_buffer = torch.zeros(batch_size, rank, dtype=sample_weight.dtype, device=sample_weight.device)
        self.gate_B_buffer = torch.zeros(rank, batch_size, dtype=sample_weight.dtype, device=sample_weight.device)
        self.w1_A_buffer = torch.zeros(batch_size, rank, dtype=sample_weight.dtype, device=sample_weight.device)
        self.w1_B_buffer = torch.zeros(rank, batch_size, dtype=sample_weight.dtype, device=sample_weight.device)
        self.w2_A_buffer = torch.zeros(batch_size, rank, dtype=sample_weight.dtype, device=sample_weight.device)
        self.w2_B_buffer = torch.zeros(rank, batch_size, dtype=sample_weight.dtype, device=sample_weight.device)

    def _compute_lora_on_cpu(
        self,
        input_tensor: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float
    ) -> torch.Tensor:
        """Compute LoRA on CPU.

        This method:
        1. Transfers input from GPU to CPU (non-blocking)
        2. Computes LoRA on CPU: input @ A @ B.T * scaling
        3. Transfers result back to GPU

        Args:
            input_tensor: Input tensor on GPU [*, hidden_dim]
            lora_A: LoRA A matrix on CPU [hidden_dim, rank]
            lora_B: LoRA B matrix on CPU [rank, output_dim]
            scaling: LoRA scaling factor (alpha / rank)

        Returns:
            LoRA output tensor on GPU [*, output_dim]
        """
        # Determine output dimension from B matrix
        output_dim = lora_B.shape[1] if lora_B.dim() == 2 else lora_B.shape[0]

        # Transfer input to CPU (non-blocking for overlap)
        input_cpu = input_tensor.to("cpu", non_blocking=True)

        # Ensure weights are on CPU
        if lora_A.device.type != "cpu":
            lora_A = lora_A.to("cpu")
        if lora_B.device.type != "cpu":
            lora_B = lora_B.to("cpu")

        # Compute LoRA on CPU: input @ A @ B.T * scaling
        # Use torch.matmul for efficiency
        with torch.no_grad():
            intermediate = torch.matmul(input_cpu, lora_A)  # [*, rank]
            output_cpu = torch.matmul(intermediate, lora_B.t()) * scaling  # [*, output_dim]

        # Transfer result back to GPU
        output_gpu = output_cpu.to(input_tensor.device, non_blocking=True)

        return output_gpu

    def _compute_lora_on_gpu(
        self,
        input_tensor: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float
    ) -> torch.Tensor:
        """Compute LoRA on GPU.

        Args:
            input_tensor: Input tensor on GPU [*, hidden_dim]
            lora_A: LoRA A matrix on GPU [hidden_dim, rank]
            lora_B: LoRA B matrix on GPU [rank, output_dim]
            scaling: LoRA scaling factor (alpha / rank)

        Returns:
            LoRA output tensor on GPU [*, output_dim]
        """
        return input_tensor @ lora_A @ lora_B.t() * scaling

    def unload_adapter(self):
        """Unload current adapter and free memory."""
        self.active_adapter = None
        self.active_adapter_dir = None

        # Clear buffers
        self.gate_A_buffer = None
        self.gate_B_buffer = None
        self.w1_A_buffer = None
        self.w1_B_buffer = None
        self.w2_A_buffer = None
        self.w2_B_buffer = None

    def apply_moe_gate_lora(
        self,
        hidden_states: torch.Tensor,
        layer_id: int,
        infer_state: Optional[Any] = None
    ) -> torch.Tensor:
        """Apply LoRA to moe_gate (router logits).

        This modifies the routing decisions by adding LoRA contribution
        to the router logits before softmax and top-k selection.

        Args:
            hidden_states: Input tensor [batch, hidden_size]
            layer_id: Current layer ID
            infer_state: Inference state (for compatibility)

        Returns:
            LoRA contribution tensor to be added to router logits [batch, num_experts]
        """
        if self.active_adapter is None or not self.has_lora:
            return torch.zeros(hidden_states.size(0), 1, device=hidden_states.device, dtype=hidden_states.dtype)

        layer_weights = self.active_adapter.get(layer_id, {})
        if not layer_weights:
            return torch.zeros(hidden_states.size(0), 1, device=hidden_states.device, dtype=hidden_states.dtype)

        gate_A = layer_weights.get("gate_lora_A")
        gate_B = layer_weights.get("gate_lora_B")

        if gate_A is None or gate_B is None:
            return torch.zeros(hidden_states.size(0), 1, device=hidden_states.device, dtype=hidden_states.dtype)

        # Compute LoRA on GPU or CPU based on config
        if self.lora_compute_config.moe_compute == "cpu":
            return self._compute_lora_on_cpu(hidden_states, gate_A, gate_B, self.scaling)
        else:
            return hidden_states @ gate_A @ gate_B.t() * self.scaling

    def apply_w1_lora(
        self,
        intermediate_states: torch.Tensor,
        layer_id: int,
        infer_state: Optional[Any] = None
    ) -> torch.Tensor:
        """Apply LoRA to w1 (fused gate+up projection).

        This is applied to the intermediate output after w1 GEMM but before
        SwiGLU activation. The input shape is [token * topk, inter_size].

        Args:
            intermediate_states: Input tensor [token * topk, inter_size]
            layer_id: Current layer ID
            infer_state: Inference state (for compatibility)

        Returns:
            LoRA contribution tensor to be added to intermediate [token * topk, inter_size]
        """
        if self.active_adapter is None or not self.has_lora:
            return torch.zeros_like(intermediate_states)

        layer_weights = self.active_adapter.get(layer_id, {})
        if not layer_weights:
            return torch.zeros_like(intermediate_states)

        w1_A = layer_weights.get("w1_lora_A")
        w1_B = layer_weights.get("w1_lora_B")

        if w1_A is None or w1_B is None:
            return torch.zeros_like(intermediate_states)

        # Compute LoRA on GPU or CPU based on config
        if self.lora_compute_config.moe_compute == "cpu":
            return self._compute_lora_on_cpu(intermediate_states, w1_A, w1_B, self.scaling)
        else:
            return intermediate_states @ w1_A @ w1_B.t() * self.scaling

    def apply_w2_lora(
        self,
        output_states: torch.Tensor,
        layer_id: int,
        infer_state: Optional[Any] = None
    ) -> torch.Tensor:
        """Apply LoRA to w2 (down projection).

        This is applied to the down projection output before moe_sum_reduce.
        The LoRA output will be routed/aggregated along with expert outputs.

        Args:
            output_states: Input tensor [token * topk, hidden_size]
            layer_id: Current layer ID
            infer_state: Inference state (for compatibility)

        Returns:
            LoRA contribution tensor to be added to output [token * topk, hidden_size]
        """
        if self.active_adapter is None or not self.has_lora:
            return torch.zeros_like(output_states)

        layer_weights = self.active_adapter.get(layer_id, {})
        if not layer_weights:
            return torch.zeros_like(output_states)

        w2_A = layer_weights.get("w2_lora_A")
        w2_B = layer_weights.get("w2_lora_B")

        if w2_A is None or w2_B is None:
            return torch.zeros_like(output_states)

        # Compute LoRA on GPU or CPU based on config
        if self.lora_compute_config.moe_compute == "cpu":
            return self._compute_lora_on_cpu(output_states, w2_A, w2_B, self.scaling)
        else:
            return output_states @ w2_A @ w2_B.t() * self.scaling


def create_moe_lora_dispatcher(
    lora_rank: int = 64,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
    lora_compute_config: Optional[LoRAComputeConfig] = None
) -> Qwen3MOELoRADispatcher:
    """Factory function to create a MoE LoRA dispatcher.

    Args:
        lora_rank: LoRA rank for all MoE injection points
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        lora_compute_config: Configuration for compute location per component

    Returns:
        Qwen3MOELoRADispatcher instance

    Example:
        # GPU computation (default)
        dispatcher = create_moe_lora_dispatcher(lora_rank=64, lora_alpha=1.0)

        # CPU offload for MoE (weights stay on CPU, computation on CPU)
        config = LoRAComputeConfig(moe="cpu")
        dispatcher = create_moe_lora_dispatcher(
            lora_rank=64,
            lora_alpha=1.0,
            lora_compute_config=config
        )
    """
    return Qwen3MOELoRADispatcher(lora_rank, lora_alpha, lora_dropout, lora_compute_config)
