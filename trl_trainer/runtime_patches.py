"""Runtime patches for vLLM and compressed-tensors.

QeRL needs three small behaviour changes in third-party packages. They used to be applied by copying whole
files over `site-packages` (see the old `setup_env.sh`), which pinned the repo to one exact version of each
package and silently broke on every upgrade. They are now applied at import time instead:

1. vLLM: accept LoRA adapters passed as in-memory tensors (`TensorLoRARequest`) so the freshly trained adapter
   can be handed to the colocated rollout engine without a round trip through the filesystem.
2. compressed-tensors: `CompressedLinear.forward` must not replace the packed NVFP4 weight with a dense one on
   the first call (upstream decompresses once and keeps the dense weight, which defeats the memory saving).
3. compressed-tensors: a much faster NVFP4 dequantization (single LUT gather instead of the reference
   bit-twiddling) for the forward pass of the training replica.

Call `apply_patches()` once per process before the vLLM engine is created, and `unwrap_compressed_linears`
on the HF training model after loading it.
"""

import os
import sys
from typing import Optional

import torch
from torch.nn.functional import embedding, linear


_PATCHED = False

# ---------------------------------------------------------------------------------------------------------
# NVFP4 dequantization
# ---------------------------------------------------------------------------------------------------------

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_LUT_CACHE: dict = {}


def _nibble_pair_lut(device, dtype) -> torch.Tensor:
    """256x2 table mapping a packed byte to its two FP4 (E2M1) values (low nibble first)."""
    key = (device, dtype)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        vals = torch.tensor([*_E2M1, *[-v for v in _E2M1]], dtype=torch.float32)
        idx = torch.arange(256)
        lut = torch.stack((vals[idx & 0x0F], vals[idx >> 4]), dim=1).to(device=device, dtype=dtype)
        _LUT_CACHE[key] = lut
    return lut


def dequantize_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dense weight of an NVFP4 (`nvfp4-pack-quantized`) layer.

    Numerically equivalent to `NVFP4PackedCompressor.decompress_weight`, but a single gather through a 256-entry
    table instead of ~10 full-size temporaries (including two int64 ones), i.e. several times faster and with
    a fraction of the transient memory.
    """
    out_features, half_in = weight_packed.shape
    lut = _nibble_pair_lut(weight_packed.device, dtype)
    values = embedding(weight_packed.to(torch.int32), lut).view(out_features, half_in * 2)
    # One local scale per group of `group_size` input features, itself stored relative to a per-tensor
    # global scale.
    scale = (weight_scale.to(torch.float32) / weight_global_scale.to(torch.float32)).to(dtype)
    num_groups = scale.shape[1]
    values = values.view(out_features, num_groups, -1) * scale.unsqueeze(-1)
    return values.view(out_features, half_in * 2)


def _is_nvfp4(module) -> bool:
    return hasattr(module, "weight_packed") and hasattr(module, "weight_global_scale")


def _compressed_linear_forward(self, input: torch.Tensor) -> torch.Tensor:
    from compressed_tensors.quantization import QuantizationStatus

    if self.quantization_status != QuantizationStatus.COMPRESSED:
        return linear(input, self.weight, self.bias)

    with torch.no_grad():
        if _is_nvfp4(self):
            weight = dequantize_nvfp4(self.weight_packed, self.weight_scale, self.weight_global_scale)
        else:
            weight = self.compressor.decompress_module(self)
    if weight.dtype != input.dtype and not torch.is_autocast_enabled():
        # Under autocast `linear` picks the dtype itself; casting here would only add a full-size copy.
        weight = weight.to(input.dtype)
    return linear(input, weight, self.bias)


def unwrap_compressed_linears(model: torch.nn.Module) -> int:
    """Make every weight-only-quantized `CompressedLinear` of `model` use the forward pass above.

    compressed-tensors wraps the forward of each quantized module at load time with a calibration hook that
    captured the *upstream* `CompressedLinear.forward`. That one decompresses the weight on the first call,
    keeps it, and from then on the hook re-quantizes the (already quantized) dense weight on every call. For
    weight-only schemes the hook has nothing else to do, so it is simply removed. Returns the number of
    modules that were unwrapped.
    """
    from compressed_tensors.linear.compressed_linear import CompressedLinear

    count = 0
    for module in model.modules():
        if not isinstance(module, CompressedLinear) or "forward" not in module.__dict__:
            continue
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.input_activations is not None or scheme.output_activations is not None:
            continue  # activation quantization still needs the hook
        del module.forward
        count += 1
    return count


def _patch_compressed_tensors() -> None:
    from compressed_tensors.linear.compressed_linear import CompressedLinear

    CompressedLinear.forward = _compressed_linear_forward


# ---------------------------------------------------------------------------------------------------------
# vLLM: LoRA adapters from in-memory tensors
# ---------------------------------------------------------------------------------------------------------


def _patch_vllm_lora() -> None:
    from vllm.lora.peft_helper import PEFTHelper
    from vllm.lora.worker_manager import WorkerLoRAManager

    original_load_adapter = WorkerLoRAManager._load_adapter

    def _load_adapter(self, lora_request):
        tensors = getattr(lora_request, "lora_tensors", None)
        if tensors is None:
            return original_load_adapter(self, lora_request)

        peft_helper = PEFTHelper.from_dict(dict(lora_request.peft_config))
        peft_helper.validate_legal(self.lora_config)
        model = self._adapter_manager.model
        lora = self._lora_model_cls.from_lora_tensors(
            lora_model_id=lora_request.lora_int_id,
            tensors=tensors,
            peft_helper=peft_helper,
            device="cpu",
            dtype=self.lora_config.lora_dtype,
            embeddings=None,
            target_embedding_padding=self.vocab_size + self.lora_config.lora_extra_vocab_size,
            embedding_modules=self.embedding_modules,
            embedding_padding_modules=self.embedding_padding_modules,
            weights_mapper=getattr(model, "hf_to_vllm_mapper", None),
        )
        return lora

    WorkerLoRAManager._load_adapter = _load_adapter

    # An in-memory adapter has no folder to look for a tokenizer in. Without this vLLM tries to resolve
    # `lora_path` (locally, then on the Hub) and logs a warning for every new adapter, i.e. on every sync.
    import vllm.transformers_utils.tokenizer as tokenizer_module
    import vllm.transformers_utils.tokenizer_group as tokenizer_group_module

    original_get_lora_tokenizer = tokenizer_module.get_lora_tokenizer

    def get_lora_tokenizer(lora_request, *args, **kwargs):
        if getattr(lora_request, "lora_tensors", None) is not None:
            return None  # use the base model tokenizer
        return original_get_lora_tokenizer(lora_request, *args, **kwargs)

    tokenizer_module.get_lora_tokenizer = get_lora_tokenizer
    if hasattr(tokenizer_group_module, "get_lora_tokenizer"):
        tokenizer_group_module.get_lora_tokenizer = get_lora_tokenizer


def _configure_triton_ptxas() -> None:
    """Let Triton (used by vLLM's torch.compile backend) assemble for GPUs newer than its bundled `ptxas`.

    The `ptxas` shipped inside the triton wheel that matches a given torch release does not know GPU
    architectures released after it (e.g. `sm_103a`, i.e. B300/GB300, with triton 3.4), and compilation of the
    rollout model then fails. If the `nvidia-cuda-nvcc-cu12` wheel is installed, use its `ptxas` instead.
    """
    if os.environ.get("TRITON_PTXAS_PATH"):
        return
    for entry in sys.path:
        candidate = os.path.join(entry, "nvidia", "cuda_nvcc", "bin", "ptxas")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            os.environ["TRITON_PTXAS_PATH"] = candidate
            return


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _configure_triton_ptxas()
    _patch_compressed_tensors()
    _patch_vllm_lora()
    _PATCHED = True


def make_tensor_lora_request_cls():
    from vllm.lora.request import LoRARequest

    class TensorLoRARequest(LoRARequest):
        """A LoRA request whose weights are passed in memory instead of being read from `lora_path`."""

        peft_config: Optional[dict] = None
        lora_tensors: Optional[dict] = None

    return TensorLoRARequest
