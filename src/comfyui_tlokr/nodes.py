"""Runtime-safe loader for the Anima trainer's T-LoKr safetensor format.

T-LoKr is not an ordinary LoKr checkpoint: its large Kronecker operand is
factorized and its inner rank changes for every flow-matching timestep. A
normal Comfy LoRA loader would materialize an always-full-rank LoKr delta,
silently changing the adapter. This node keeps the schedule live for each
denoiser invocation.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import torch
from torch import nn

try:  # ComfyUI supplies this import at runtime; keeping imports local helps tests.
    import folder_paths
except ImportError:  # pragma: no cover - only used when importing outside ComfyUI.
    folder_paths = None


FORMAT_VERSION = 1
_COMPONENTS = frozenset(
    {"alpha", "lokr_w1", "lokr_w2_a", "lokr_w2_b", "tlokr_schedule"}
)
_KEY_RE = re.compile(r"^(?P<prefix>.+)\.(?P<component>" + "|".join(_COMPONENTS) + r")$")
_TIMESTEPS: ContextVar[torch.Tensor | None] = ContextVar("comfyui_tlokr_timesteps", default=None)


@dataclass(frozen=True)
class TLoKrWeights:
    """One target's validated tensors, still on CPU after safetensor loading."""

    prefix: str
    alpha: float
    w1: torch.Tensor
    w2_a: torch.Tensor
    w2_b: torch.Tensor
    max_rank: int
    min_rank: int

    @property
    def expected_weight_shape(self) -> tuple[int, int]:
        return (self.w1.shape[0] * self.w2_a.shape[0], self.w1.shape[1] * self.w2_b.shape[1])


class TLoKrFormatError(ValueError):
    """Raised before touching a model when a checkpoint is not T-LoKr v1."""


def active_rank(timesteps: torch.Tensor, max_rank: int, min_rank: int) -> torch.Tensor:
    """The trainer's T-LoRA prefix schedule, with t=1 as the noisy endpoint."""
    return torch.floor(
        (max_rank - min_rank) * (1.0 - timesteps.clamp(0.0, 1.0)) + min_rank
    ).to(torch.int64).clamp_(min=min_rank, max=max_rank)


def _as_ints(tensor: torch.Tensor, key: str) -> tuple[int, ...]:
    if tensor.numel() != 3:
        raise TLoKrFormatError(f"{key} must have exactly 3 values [format, max_rank, min_rank]")
    values = tensor.detach().to(device="cpu", dtype=torch.int64).flatten()
    return tuple(int(value) for value in values)


def parse_tlokr_state(
    tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str] | None = None
) -> dict[str, TLoKrWeights]:
    """Strictly parse the portable checkpoint format emitted by anima-trainer."""
    metadata = metadata or {}
    if metadata.get("anima_adapter_type") != "tlokr":
        raise TLoKrFormatError("missing metadata anima_adapter_type=tlokr")
    if metadata.get("anima_adapter_format") != str(FORMAT_VERSION):
        raise TLoKrFormatError(
            f"unsupported anima_adapter_format={metadata.get('anima_adapter_format')!r}; expected {FORMAT_VERSION}"
        )

    groups: dict[str, dict[str, torch.Tensor]] = {}
    unexpected: list[str] = []
    for key, tensor in tensors.items():
        match = _KEY_RE.match(key)
        if match is None:
            unexpected.append(key)
            continue
        groups.setdefault(match["prefix"], {})[match["component"]] = tensor
    if unexpected:
        preview = ", ".join(sorted(unexpected)[:5])
        raise TLoKrFormatError(f"unexpected tensor keys (not T-LoKr v1): {preview}")
    if not groups:
        raise TLoKrFormatError("checkpoint contains no T-LoKr target tensors")

    parsed: dict[str, TLoKrWeights] = {}
    for prefix, values in groups.items():
        missing = _COMPONENTS.difference(values)
        if missing:
            raise TLoKrFormatError(f"{prefix} is missing: {', '.join(sorted(missing))}")

        w1, w2_a, w2_b = values["lokr_w1"], values["lokr_w2_a"], values["lokr_w2_b"]
        if w1.ndim != 2 or w2_a.ndim != 2 or w2_b.ndim != 2:
            raise TLoKrFormatError(f"{prefix} factors must all be rank-2 tensors")
        version, max_rank, min_rank = _as_ints(values["tlokr_schedule"], f"{prefix}.tlokr_schedule")
        if version != FORMAT_VERSION:
            raise TLoKrFormatError(f"{prefix} uses format {version}; expected {FORMAT_VERSION}")
        if not 1 <= min_rank <= max_rank:
            raise TLoKrFormatError(f"{prefix} has invalid schedule max={max_rank}, min={min_rank}")
        if w2_a.shape[1] != max_rank or w2_b.shape[0] != max_rank:
            raise TLoKrFormatError(
                f"{prefix} schedule rank {max_rank} disagrees with W2 factor shapes "
                f"{tuple(w2_a.shape)} and {tuple(w2_b.shape)}"
            )
        alpha = values["alpha"]
        if alpha.numel() != 1:
            raise TLoKrFormatError(f"{prefix}.alpha must be scalar")
        parsed[prefix] = TLoKrWeights(
            prefix=prefix,
            alpha=float(alpha.detach().to(torch.float32).item()),
            w1=w1.contiguous(),
            w2_a=w2_a.contiguous(),
            w2_b=w2_b.contiguous(),
            max_rank=max_rank,
            min_rank=min_rank,
        )
    return parsed


def load_tlokr(path: str | Path) -> dict[str, TLoKrWeights]:
    """Read a safetensor without accepting pickle-based checkpoints."""
    from safetensors import safe_open

    path = Path(path)
    with safe_open(path, framework="pt", device="cpu") as file:
        metadata = file.metadata()
        tensors = {key: file.get_tensor(key) for key in file.keys()}
    return parse_tlokr_state(tensors, metadata)


class _TLoKrAdapter(nn.Module):
    def __init__(self, weights: TLoKrWeights, strength: float):
        super().__init__()
        self.prefix = weights.prefix
        self.max_rank = weights.max_rank
        self.min_rank = weights.min_rank
        # anima-trainer's LyCORIS scale is alpha / network_dim (128 by default).
        self.scale = float(strength) * weights.alpha / weights.max_rank
        self.register_buffer("w1", weights.w1, persistent=False)
        self.register_buffer("w2_a", weights.w2_a, persistent=False)
        self.register_buffer("w2_b", weights.w2_b, persistent=False)

    def delta(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise RuntimeError(f"{self.prefix}: expected a batched Linear input, got {tuple(x.shape)}")
        if x.shape[-1] % self.w1.shape[1]:
            raise RuntimeError(
                f"{self.prefix}: input width {x.shape[-1]} is not divisible by W1 input factor {self.w1.shape[1]}"
            )
        timesteps = timesteps.reshape(-1).to(device=x.device, dtype=torch.float32)
        if timesteps.numel() == 1 and x.shape[0] != 1:
            timesteps = timesteps.expand(x.shape[0])
        if timesteps.numel() != x.shape[0]:
            raise RuntimeError(
                f"{self.prefix}: activation batch {x.shape[0]} does not match timestep batch {timesteps.numel()}"
            )
        dtype = x.dtype
        w1 = self.w1.to(dtype=dtype)
        w2_a = self.w2_a.to(dtype=dtype)
        w2_b = self.w2_b.to(dtype=dtype)
        grouped = x.reshape(x.shape[0], -1, w1.shape[1], x.shape[-1] // w1.shape[1])
        hidden = torch.matmul(grouped, w2_b.transpose(0, 1))
        rank_mask = (
            torch.arange(self.max_rank, device=x.device)
            < active_rank(timesteps, self.max_rank, self.min_rank).unsqueeze(1)
        ).to(dtype)
        # Equivalent to the trainer's rank_mix(hidden, W1, mask), expressed
        # directly to keep the Comfy custom node dependency-free.
        mixed = torch.einsum("ou,bnur,br->bnor", w1, hidden, rank_mask)
        out = torch.matmul(mixed, w2_a.transpose(0, 1))
        return out.reshape(*x.shape[:-1], -1) * self.scale


class TLoKrLinear(nn.Module):
    """An nn.Linear-compatible wrapper that leaves the base layer untouched."""

    def __init__(self, base: nn.Module, adapters: tuple[_TLoKrAdapter, ...]):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TLoKrFormatError(
                f"T-LoKr supports Linear targets only; got {type(base).__module__}.{type(base).__name__}"
            )
        self.base = base
        self.adapters = nn.ModuleList(adapters)

    @classmethod
    def with_adapter(cls, base: nn.Module, weights: TLoKrWeights, strength: float) -> "TLoKrLinear":
        adapter = _TLoKrAdapter(weights, strength)
        if isinstance(base, cls):
            return cls(base.base, tuple(base.adapters) + (adapter,))
        return cls(base, (adapter,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        timesteps = _TIMESTEPS.get()
        if timesteps is None:
            raise RuntimeError("T-LoKr has no denoiser timestep context; use the T-LoKr Loader node")
        output = self.base(x)
        for adapter in self.adapters:
            output = output + adapter.delta(x, timesteps).to(dtype=output.dtype)
        return output


def _model_key_map(model: Any) -> dict[str, str]:
    """Map the trainer's Kohya keys to the exact Comfy model state-dict paths."""
    mapping: dict[str, str] = {}
    for state_key in model.model.state_dict().keys():
        if not state_key.startswith("diffusion_model.") or not state_key.endswith(".weight"):
            continue
        unet_name = state_key[len("diffusion_model.") : -len(".weight")]
        mapping[f"lora_unet_{unet_name.replace('.', '_')}"] = state_key
    return mapping


def _install_timestep_wrapper(model: Any) -> None:
    """Install one composable ModelPatcher wrapper around Comfy's apply_model."""
    if model.model_options.get("_comfyui_tlokr_timestep_wrapper"):
        return
    previous = model.model_options.get("model_function_wrapper")

    def wrapper(apply_model: Callable[..., torch.Tensor], args: dict[str, Any]) -> torch.Tensor:
        timestep = args.get("timestep")
        if not torch.is_tensor(timestep):
            raise RuntimeError("T-LoKr requires ComfyUI to provide a tensor denoiser timestep")
        token = _TIMESTEPS.set(timestep)
        try:
            if previous is not None:
                return previous(apply_model, args)
            return apply_model(args["input"], timestep, **args["c"])
        finally:
            _TIMESTEPS.reset(token)

    model.set_model_unet_function_wrapper(wrapper)
    model.model_options["_comfyui_tlokr_timestep_wrapper"] = True


def apply_tlokr(model: Any, adapters: Mapping[str, TLoKrWeights], strength: float) -> tuple[Any, int]:
    """Return a cloned ModelPatcher whose target Linear modules are T-LoKr-aware."""
    if strength == 0:
        return model, 0
    key_map = _model_key_map(model)
    missing = sorted(set(adapters).difference(key_map))
    if missing:
        preview = ", ".join(missing[:5])
        raise TLoKrFormatError(
            "the MODEL is not the matching Anima DiT, or uses different module names; "
            f"missing {len(missing)} T-LoKr targets (for example: {preview})"
        )

    patched = model.clone()
    for prefix, adapter in adapters.items():
        weight_key = key_map[prefix]
        module_path = weight_key[: -len(".weight")]
        target = patched.get_model_object(module_path)
        if not isinstance(target, (nn.Linear, TLoKrLinear)):
            raise TLoKrFormatError(f"{prefix} maps to {module_path}, which is not an nn.Linear")
        base = target.base if isinstance(target, TLoKrLinear) else target
        if tuple(base.weight.shape) != adapter.expected_weight_shape:
            raise TLoKrFormatError(
                f"{prefix} expects base weight {adapter.expected_weight_shape}, but {module_path} has {tuple(base.weight.shape)}"
            )
        patched.add_object_patch(module_path, TLoKrLinear.with_adapter(target, adapter, strength))
    # Object patches are applied only when Comfy loads the model. Include the
    # adapter buffers in the patcher's budget now so low-VRAM planning does not
    # undercount an otherwise invisible module tree.
    if hasattr(patched, "size") and hasattr(model, "model_size"):
        adapter_bytes = sum(
            tensor.numel() * tensor.element_size()
            for adapter in adapters.values()
            for tensor in (adapter.w1, adapter.w2_a, adapter.w2_b)
        )
        patched.size = model.model_size() + adapter_bytes
    _install_timestep_wrapper(patched)
    return patched, len(adapters)


class TLoKrLoader:
    """Apply an Anima trainer T-LoKr v1 safetensor to an Anima MODEL."""

    CATEGORY = "loaders/adapters"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_tlokr"
    DESCRIPTION = (
        "Applies a timestep-aware Anima T-LoKr v1 adapter. The input must be an Anima DiT MODEL "
        "whose flow timestep is normalized like the trainer: t=1 noisy, t=0 clean."
    )

    @classmethod
    def INPUT_TYPES(cls):
        names = folder_paths.get_filename_list("loras") if folder_paths is not None else []
        return {
            "required": {
                "model": ("MODEL",),
                "tlokr_name": (names, {"tooltip": "T-LoKr v1 safetensor stored in ComfyUI/models/loras."}),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            }
        }

    def load_tlokr(self, model: Any, tlokr_name: str, strength_model: float):
        if strength_model == 0:
            return (model,)
        if folder_paths is None:  # pragma: no cover - protects accidental standalone use.
            raise RuntimeError("T-LoKr Loader must run inside ComfyUI")
        path = folder_paths.get_full_path_or_raise("loras", tlokr_name)
        adapters = load_tlokr(path)
        patched, _ = apply_tlokr(model, adapters, strength_model)
        return (patched,)


NODE_CLASS_MAPPINGS = {"TLoKrLoader": TLoKrLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TLoKrLoader": "T-LoKr Loader (Anima)"}
