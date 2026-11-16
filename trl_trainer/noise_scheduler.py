"""Adaptive Quantization Noise (AQN).

QeRL perturbs the rollout policy with Gaussian noise whose scale decays over training. The noise is added to
the weights of the RMSNorm layers of the *rollout* (vLLM) model, which is equivalent to a multiplicative
perturbation of the activations feeding the following linear layers.
"""

import math
from typing import Optional

import torch


def sigma_schedule(sigma_start: float, sigma_end: float, num_stages: int) -> list[float]:
    """Exponentially decaying noise scales for stages 1..num_stages-1 (stage 0 uses no extra noise)."""
    if num_stages < 2:
        return []
    if num_stages == 2:
        return [sigma_start]
    ratio = sigma_end / sigma_start
    return [sigma_start * ratio ** (i / (num_stages - 2)) for i in range(num_stages - 1)]


def get_sigma_by_step(step: int, total_steps: int, sigma_trend: list[float]) -> tuple[int, float]:
    """Noise scale for `step`.

    Training is split into `len(sigma_trend) + 1` equal intervals. The first interval relies on the quantization
    noise alone (sigma = 0); interval `i >= 1` uses `sigma_trend[i - 1]`.
    """
    if not sigma_trend or total_steps <= 0:
        return 0, 0.0
    step = min(max(step, 0), total_steps)
    steps_per_interval = total_steps / (len(sigma_trend) + 1)
    interval_id = int(step // steps_per_interval)
    if interval_id == 0:
        return 0, 0.0
    sigma_id = min(interval_id - 1, len(sigma_trend) - 1)
    return sigma_id, sigma_trend[sigma_id]


def _is_rms_norm(module: torch.nn.Module) -> bool:
    # Covers transformers' Qwen2RMSNorm / LlamaRMSNorm as well as vLLM's own RMSNorm, without importing either.
    return type(module).__name__.endswith("RMSNorm") and isinstance(getattr(module, "weight", None), torch.Tensor)


class AdaptiveQuantizationNoise:
    """Applies scheduled Gaussian noise to the RMSNorm weights of a model.

    The clean weights are remembered so that noise never accumulates across calls: each `apply` sets
    `weight = clean_weight + N(0, sigma^2)`. Norm layers whose weights are trained are overwritten with fresh
    values before every rollout; pass their module names as `synced` so the clean copy follows them.

    The noise is drawn from a generator seeded with `(seed, step)` only, so every process of a distributed run
    perturbs its rollout policy identically. Completions of the same prompt may be generated on different
    GPUs, and they must all come from one policy for the group-normalised advantages to be meaningful.
    """

    def __init__(self, sigma_start: float, sigma_end: float, num_stages: int, seed: int = 0):
        if sigma_start <= 0 or sigma_end <= 0:
            raise ValueError("sigma_start and sigma_end must be positive.")
        self.sigma_trend = sigma_schedule(sigma_start, sigma_end, num_stages)
        self.seed = seed
        self._clean: dict[str, torch.Tensor] = {}
        self.last_sigma: Optional[float] = None

    def sigma(self, step: int, total_steps: int) -> float:
        return get_sigma_by_step(step, total_steps, self.sigma_trend)[1]

    @torch.no_grad()
    def apply(self, model: torch.nn.Module, step: int, total_steps: int, synced: Optional[set] = None) -> float:
        """Perturb `model` for rollout `step`. `synced` holds the names of the norm modules whose weights were
        just overwritten with clean (trained) values."""
        sigma = self.sigma(step, total_steps)
        synced = synced or set()
        norms = [(name, module) for name, module in model.named_modules() if _is_rms_norm(module)]
        if not norms:
            raise RuntimeError("AQN is enabled but the rollout model has no RMSNorm layer to perturb.")

        for name, module in norms:
            if name not in self._clean:
                self._clean[name] = module.weight.detach().clone()
            elif name in synced:
                self._clean[name].copy_(module.weight)
            elif self.last_sigma:
                # Not re-synced, so the weight still carries the previous noise: restore it.
                module.weight.copy_(self._clean[name])

        if sigma > 0 and math.isfinite(sigma):
            generator = torch.Generator(device="cpu").manual_seed((self.seed * 1_000_003 + step) % (2**63 - 1))
            for name, module in norms:
                weight = module.weight
                noise = torch.randn(weight.shape, generator=generator, dtype=torch.float32) * sigma
                weight.add_(noise.to(device=weight.device, dtype=weight.dtype))
        self.last_sigma = sigma
        return sigma
