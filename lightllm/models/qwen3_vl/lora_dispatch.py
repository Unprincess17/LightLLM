"""
LoRA Dispatch for Qwen3-VL MLP and Attention Layers

This module provides functions for applying LoRA to both MLP layers (gate_proj, up_proj, down_proj)
and Attention layers (q_proj, k_proj, v_proj, o_proj).

Unlike S-LoRA's dispatch_bgmv which is optimized for attention batching, this module provides
direct LoRA application for detached serving.

The dispatch approach:
1. Store LoRA A/B matrices separately from base model
2. Load/unload adapter weights dynamically per batch
3. Apply LoRA: output = base_output + input @ A @ B * scaling
"""
import torch
from typing import Dict, Optional, Any


class LoRADispatcher:
    """Dispatcher for applying LoRA to MLP layers.

    Manages multiple adapters and applies LoRA during inference.
    This is the main interface for detached LoRA serving.

    CPU Offload:
    - Set compute_on_cpu=True to run LoRA computation on CPU
    - LoRA weights stay on CPU, only input/output transferred
    - Reduces GPU memory bandwidth overhead
    """

    def __init__(
        self,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        compute_on_cpu: bool = False
    ):
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_scaling = lora_alpha / lora_rank if lora_rank > 0 else 1.0
        self.compute_on_cpu = compute_on_cpu  # CPU offload flag

        # Active adapter (loaded for current batch)
        self.active_adapter: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
        self.active_adapter_dir: Optional[str] = None

        # Memory for storing A matrices (for efficient computation)
        # MLP LoRA buffers
        self.gate_A_buffer: Optional[torch.Tensor] = None
        self.gate_B_buffer: Optional[torch.Tensor] = None
        self.up_A_buffer: Optional[torch.Tensor] = None
        self.up_B_buffer: Optional[torch.Tensor] = None
        self.down_A_buffer: Optional[torch.Tensor] = None
        self.down_B_buffer: Optional[torch.Tensor] = None

        # Attention LoRA buffers
        self.q_A_buffer: Optional[torch.Tensor] = None
        self.q_B_buffer: Optional[torch.Tensor] = None
        self.k_A_buffer: Optional[torch.Tensor] = None
        self.k_B_buffer: Optional[torch.Tensor] = None
        self.v_A_buffer: Optional[torch.Tensor] = None
        self.v_B_buffer: Optional[torch.Tensor] = None
        self.o_A_buffer: Optional[torch.Tensor] = None
        self.o_B_buffer: Optional[torch.Tensor] = None

    def load_adapter(self, adapter_weights: Dict[int, Dict[str, torch.Tensor]]):
        """Load adapter weights into dispatcher.

        Args:
            adapter_weights: Dict mapping layer_id -> {gate_proj_A, gate_proj_B, ..., q_proj_A, q_proj_B, ...}
        """
        if self.lora_rank == 0:
            return

        self.active_adapter = adapter_weights

        # Extract A matrices for efficient access
        first_layer = adapter_weights.get(0, {})
        if not first_layer:
            return

        # Get shape from first weight (try MLP first, then attention)
        sample_weight = first_layer.get("gate_proj_A") or first_layer.get("q_proj_A")
        if sample_weight is None:
            return

        batch_size = sample_weight.shape[0]  # This is actually hidden_size for our case
        rank = sample_weight.shape[1] if sample_weight.dim() == 2 else self.lora_rank

        # Store buffers on GPU
        # MLP LoRA buffers
        self.gate_A_buffer = torch.zeros(batch_size, rank, dtype=sample_weight.dtype, device=sample_weight.device)
        self.gate_B_buffer = torch.zeros(rank, batch_size, dtype=sample_weight.dtype, device=sample_weight.device)
        self.up_A_buffer = torch.zeros(batch_size, rank, dtype=sample_weight.dtype, device=sample_weight.device)
        self.up_B_buffer = torch.zeros(rank, batch_size, dtype=sample_weight.dtype, device=sample_weight.device)
        self.down_A_buffer = torch.zeros(batch_size, rank, dtype=sample_weight.dtype, device=sample_weight.device)
        self.down_B_buffer = torch.zeros(rank, batch_size, dtype=sample_weight.dtype, device=sample_weight.device)

        # Attention LoRA buffers
        self.q_A_buffer = torch.zeros(batch_size, rank, dtype=sample_weight.dtype, device=sample_weight.device)
        self.q_B_buffer = torch.zeros(rank, batch_size, dtype=sample_weight.dtype, device=sample_weight.device)
        self.k_A_buffer = torch.zeros(batch_size, rank, dtype=sample_weight.dtype, device=sample_weight.device)
        self.k_B_buffer = torch.zeros(rank, batch_size, dtype=sample_weight.dtype, device=sample_weight.device)
        self.v_A_buffer = torch.zeros(batch_size, rank, dtype=sample_weight.dtype, device=sample_weight.device)
        self.v_B_buffer = torch.zeros(rank, batch_size, dtype=sample_weight.dtype, device=sample_weight.device)
        self.o_A_buffer = torch.zeros(batch_size, rank, dtype=sample_weight.dtype, device=sample_weight.device)
        self.o_B_buffer = torch.zeros(rank, batch_size, dtype=sample_weight.dtype, device=sample_weight.device)

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

    def unload_adapter(self):
        """Unload current adapter and free memory."""
        self.active_adapter = None
        self.active_adapter_dir = None
        # MLP LoRA buffers
        self.gate_A_buffer = None
        self.gate_B_buffer = None
        self.up_A_buffer = None
        self.up_B_buffer = None
        self.down_A_buffer = None
        self.down_B_buffer = None
        # Attention LoRA buffers
        self.q_A_buffer = None
        self.q_B_buffer = None
        self.k_A_buffer = None
        self.k_B_buffer = None
        self.v_A_buffer = None
        self.v_B_buffer = None
        self.o_A_buffer = None
        self.o_B_buffer = None

    def apply_mlp_lora(
        self,
        input_embeds: torch.Tensor,
        layer_id: int,
        infer_state: Optional[Any] = None
    ) -> torch.Tensor:
        """Apply LoRA to MLP layer output.

        Computes: output = base_output + LoRA contribution

        Args:
            input_embeds: Input tensor [batch, hidden_size]
            layer_id: Current layer ID
            infer_state: Inference state (for compatibility)

        Returns:
            LoRA contribution tensor to be added to base output
        """
        if self.lora_rank == 0 or self.active_adapter is None:
            return torch.zeros_like(input_embeds)

        layer_weights = self.active_adapter.get(layer_id, {})
        if not layer_weights:
            return torch.zeros_like(input_embeds)

        # Get LoRA weights for this layer
        gate_A = layer_weights.get("gate_proj_A")
        gate_B = layer_weights.get("gate_proj_B")
        up_A = layer_weights.get("up_proj_A")
        up_B = layer_weights.get("up_proj_B")
        down_A = layer_weights.get("down_proj_A")
        down_B = layer_weights.get("down_proj_B")

        if gate_A is None or gate_B is None:
            return torch.zeros_like(input_embeds)

        scaling = self.lora_scaling

        # Note: A matrices are transposed to [hidden, rank] in lora_layer_weight.py
        # B matrices are stored as [output_dim, rank] from safetensors
        # So we need to transpose B for computation: input @ A @ B.t()

        # Compute LoRA contribution for gate_proj (on GPU or CPU)
        # input: [batch, hidden], A: [hidden, rank], B: [output_dim, rank]
        # input @ A: [batch, rank]
        # (input @ A) @ B.t(): [batch, rank] @ [rank, output_dim] = [batch, output_dim]
        if self.compute_on_cpu:
            gate_lora = self._compute_lora_on_cpu(input_embeds, gate_A, gate_B, scaling)
        else:
            gate_lora = input_embeds @ gate_A @ gate_B.t() * scaling

        # Compute LoRA contribution for up_proj (on GPU or CPU)
        if self.compute_on_cpu:
            up_lora = self._compute_lora_on_cpu(input_embeds, up_A, up_B, scaling)
        else:
            up_lora = input_embeds @ up_A @ up_B.t() * scaling

        # Combine gate and up (same as base FFN: silu(gate) * up)
        # gate_up_lora: [batch, 2*intermediate]
        # ffn1_lora: [batch, intermediate]
        gate_up_lora = torch.cat([gate_lora, up_lora], dim=-1)
        intermediate_dim = gate_lora.shape[-1]  # This is actually output_dim
        ffn1_lora = torch.zeros(input_embeds.size(0), intermediate_dim, dtype=input_embeds.dtype, device=input_embeds.device)
        silu_and_mul_fwd(gate_up_lora, ffn1_lora)

        # Compute LoRA contribution for down_proj (on GPU or CPU)
        # ffn1_lora: [batch, intermediate*2], A: [intermediate, rank], B: [hidden, rank]
        if self.compute_on_cpu:
            down_lora = self._compute_lora_on_cpu(ffn1_lora, down_A, down_B, scaling)
        else:
            down_lora = ffn1_lora @ down_A @ down_B.t() * scaling

        return down_lora

    def apply_mlp_lora_fused(
        self,
        input_embeds: torch.Tensor,
        layer_id: int,
        base_output: torch.Tensor
    ) -> torch.Tensor:
        """Apply LoRA and add to base output in one step.

        More efficient than separate calls when base_output is already computed.

        Args:
            input_embeds: Input tensor [batch, hidden_size]
            layer_id: Current layer ID
            base_output: Base MLP output [batch, hidden_size]

        Returns:
            Output tensor with LoRA applied
        """
        if self.lora_rank == 0 or self.active_adapter is None:
            return base_output

        layer_weights = self.active_adapter.get(layer_id, {})
        if not layer_weights:
            return base_output

        gate_A = layer_weights.get("gate_proj_A")
        gate_B = layer_weights.get("gate_proj_B")
        up_A = layer_weights.get("up_proj_A")
        up_B = layer_weights.get("up_proj_B")
        down_A = layer_weights.get("down_proj_A")
        down_B = layer_weights.get("down_proj_B")

        if gate_A is None or gate_B is None:
            return base_output

        scaling = self.lora_scaling

        # Compute LoRA contribution
        gate_lora = input_embeds @ gate_A @ gate_B * scaling
        up_lora = input_embeds @ up_A @ up_B * scaling
        gate_up_lora = torch.cat([gate_lora, up_lora], dim=-1)
        ffn1_lora = torch.zeros_like(gate_up_lora)
        silu_and_mul_fwd(gate_up_lora, ffn1_lora)
        down_lora = ffn1_lora @ down_A @ down_B * scaling

        return base_output + down_lora

    def apply_attention_lora(
        self,
        input_embeds: torch.Tensor,
        layer_id: int,
        infer_state: Optional[Any] = None
    ) -> Dict[str, torch.Tensor]:
        """Apply LoRA to attention layer projections (q, k, v, o).

        Computes LoRA contributions for q_proj, k_proj, v_proj, and o_proj.

        Args:
            input_embeds: Input tensor [batch, hidden_size]
            layer_id: Current layer ID
            infer_state: Inference state (for compatibility)

        Returns:
            Dictionary with q_lora, k_lora, v_lora, o_lora tensors
        """
        if self.lora_rank == 0 or self.active_adapter is None:
            return {
                "q_lora": torch.zeros_like(input_embeds),
                "k_lora": torch.zeros_like(input_embeds),
                "v_lora": torch.zeros_like(input_embeds),
                "o_lora": torch.zeros_like(input_embeds),
            }

        layer_weights = self.active_adapter.get(layer_id, {})
        if not layer_weights:
            return {
                "q_lora": torch.zeros_like(input_embeds),
                "k_lora": torch.zeros_like(input_embeds),
                "v_lora": torch.zeros_like(input_embeds),
                "o_lora": torch.zeros_like(input_embeds),
            }

        scaling = self.lora_scaling

        # Get LoRA weights for attention projections
        q_A = layer_weights.get("q_proj_A")
        q_B = layer_weights.get("q_proj_B")
        k_A = layer_weights.get("k_proj_A")
        k_B = layer_weights.get("k_proj_B")
        v_A = layer_weights.get("v_proj_A")
        v_B = layer_weights.get("v_proj_B")
        o_A = layer_weights.get("o_proj_A")
        o_B = layer_weights.get("o_proj_B")

        # Compute LoRA for q_proj (on GPU or CPU)
        q_lora = torch.zeros_like(input_embeds)
        if q_A is not None and q_B is not None:
            if self.compute_on_cpu:
                q_lora = self._compute_lora_on_cpu(input_embeds, q_A, q_B, scaling)
            else:
                q_lora = input_embeds @ q_A @ q_B.t() * scaling

        # Compute LoRA for k_proj (on GPU or CPU)
        k_lora = torch.zeros_like(input_embeds)
        if k_A is not None and k_B is not None:
            if self.compute_on_cpu:
                k_lora = self._compute_lora_on_cpu(input_embeds, k_A, k_B, scaling)
            else:
                k_lora = input_embeds @ k_A @ k_B.t() * scaling

        # Compute LoRA for v_proj (on GPU or CPU)
        v_lora = torch.zeros_like(input_embeds)
        if v_A is not None and v_B is not None:
            if self.compute_on_cpu:
                v_lora = self._compute_lora_on_cpu(input_embeds, v_A, v_B, scaling)
            else:
                v_lora = input_embeds @ v_A @ v_B.t() * scaling

        # Compute LoRA for o_proj (on GPU or CPU, takes attention output as input)
        o_lora = torch.zeros_like(input_embeds)
        if o_A is not None and o_B is not None:
            if self.compute_on_cpu:
                o_lora = self._compute_lora_on_cpu(input_embeds, o_A, o_B, scaling)
            else:
                o_lora = input_embeds @ o_A @ o_B.t() * scaling

        return {
            "q_lora": q_lora,
            "k_lora": k_lora,
            "v_lora": v_lora,
            "o_lora": o_lora,
        }

    def apply_q_lora(
        self,
        input_embeds: torch.Tensor,
        layer_id: int,
        infer_state: Optional[Any] = None
    ) -> torch.Tensor:
        """Apply LoRA to q_proj only.

        Args:
            input_embeds: Input tensor [batch, hidden_size]
            layer_id: Current layer ID
            infer_state: Inference state (for compatibility)

        Returns:
            LoRA contribution tensor for q_proj
        """
        if self.lora_rank == 0 or self.active_adapter is None:
            return torch.zeros_like(input_embeds)

        layer_weights = self.active_adapter.get(layer_id, {})
        q_A = layer_weights.get("q_proj_A")
        q_B = layer_weights.get("q_proj_B")

        if q_A is None or q_B is None:
            return torch.zeros_like(input_embeds)

        return input_embeds @ q_A @ q_B.t() * self.lora_scaling

    def apply_k_lora(
        self,
        input_embeds: torch.Tensor,
        layer_id: int,
        infer_state: Optional[Any] = None
    ) -> torch.Tensor:
        """Apply LoRA to k_proj only.

        Args:
            input_embeds: Input tensor [batch, hidden_size]
            layer_id: Current layer ID
            infer_state: Inference state (for compatibility)

        Returns:
            LoRA contribution tensor for k_proj
        """
        if self.lora_rank == 0 or self.active_adapter is None:
            return torch.zeros_like(input_embeds)

        layer_weights = self.active_adapter.get(layer_id, {})
        k_A = layer_weights.get("k_proj_A")
        k_B = layer_weights.get("k_proj_B")

        if k_A is None or k_B is None:
            return torch.zeros_like(input_embeds)

        return input_embeds @ k_A @ k_B.t() * self.lora_scaling

    def apply_v_lora(
        self,
        input_embeds: torch.Tensor,
        layer_id: int,
        infer_state: Optional[Any] = None
    ) -> torch.Tensor:
        """Apply LoRA to v_proj only.

        Args:
            input_embeds: Input tensor [batch, hidden_size]
            layer_id: Current layer ID
            infer_state: Inference state (for compatibility)

        Returns:
            LoRA contribution tensor for v_proj
        """
        if self.lora_rank == 0 or self.active_adapter is None:
            return torch.zeros_like(input_embeds)

        layer_weights = self.active_adapter.get(layer_id, {})
        v_A = layer_weights.get("v_proj_A")
        v_B = layer_weights.get("v_proj_B")

        if v_A is None or v_B is None:
            return torch.zeros_like(input_embeds)

        return input_embeds @ v_A @ v_B.t() * self.lora_scaling

    def apply_o_lora(
        self,
        input_embeds: torch.Tensor,
        layer_id: int,
        infer_state: Optional[Any] = None
    ) -> torch.Tensor:
        """Apply LoRA to o_proj only.

        Args:
            input_embeds: Input tensor [batch, hidden_size] (attention output)
            layer_id: Current layer ID
            infer_state: Inference state (for compatibility)

        Returns:
            LoRA contribution tensor for o_proj
        """
        if self.lora_rank == 0 or self.active_adapter is None:
            return torch.zeros_like(input_embeds)

        layer_weights = self.active_adapter.get(layer_id, {})
        o_A = layer_weights.get("o_proj_A")
        o_B = layer_weights.get("o_proj_B")

        if o_A is None or o_B is None:
            return torch.zeros_like(input_embeds)

        return input_embeds @ o_A @ o_B.t() * self.lora_scaling


def silu_and_mul_fwd(x: torch.Tensor, y: torch.Tensor):
    """SiLU activation and element-wise multiplication.

    Computes: y = silu(x[:, :mid]) * x[:, mid:]
    """
    mid = x.shape[-1] // 2
    x0 = x[:, :mid]
    x1 = x[:, mid:]
    y.copy_(torch.nn.functional.silu(x0) * x1)


def create_lora_dispatcher(
    lora_rank: int = 0,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
    compute_on_cpu: bool = False
) -> LoRADispatcher:
    """Factory function to create a LoRA dispatcher.

    Args:
        lora_rank: LoRA rank (0 means disabled)
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        compute_on_cpu: If True, compute LoRA on CPU (reduces GPU memory bandwidth)

    Returns:
        LoRADispatcher instance
    """
    return LoRADispatcher(lora_rank, lora_alpha, lora_dropout, compute_on_cpu)
