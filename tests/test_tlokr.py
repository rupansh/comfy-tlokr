from __future__ import annotations

import sys
import types
import unittest

import torch
from torch import nn

sys.modules.setdefault("folder_paths", types.SimpleNamespace())

from comfyui_tlokr.nodes import (  # noqa: E402
    FORMAT_VERSION,
    NODE_CLASS_MAPPINGS,
    TLoKrFormatError,
    TLoKrLinear,
    TLoKrLoaderWithClip,
    _TIMESTEPS,
    active_rank,
    apply_tlokr,
    parse_tlokr_state,
)


def _state() -> dict[str, torch.Tensor]:
    prefix = "lora_unet_blocks_0_mlp_layer1"
    return {
        f"{prefix}.alpha": torch.tensor(4.0),
        f"{prefix}.lokr_w1": torch.eye(2),
        f"{prefix}.lokr_w2_a": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        f"{prefix}.lokr_w2_b": torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
        f"{prefix}.tlokr_schedule": torch.tensor([FORMAT_VERSION, 2, 1], dtype=torch.int32),
    }


class TLoKrTests(unittest.TestCase):
    def test_clip_passthrough_loader_is_registered(self):
        self.assertIs(NODE_CLASS_MAPPINGS["TLoKrLoaderWithClip"], TLoKrLoaderWithClip)
        self.assertEqual(TLoKrLoaderWithClip.RETURN_TYPES, ("MODEL", "CLIP"))

    def test_schedule_matches_trainer_definition(self):
        ranks = active_rank(torch.tensor([1.0, 0.5, 0.0]), max_rank=8, min_rank=2)
        self.assertEqual(ranks.tolist(), [2, 5, 8])

    def test_tlokr_linear_masks_inactive_rank_per_sample(self):
        metadata = {"anima_adapter_type": "tlokr", "anima_adapter_format": "1"}
        weights = next(iter(parse_tlokr_state(_state(), metadata).values()))
        base = nn.Linear(4, 4, bias=False)
        nn.init.zeros_(base.weight)
        layer = TLoKrLinear.with_adapter(base, weights, 1.0)
        x = torch.ones(2, 1, 4)
        token = _TIMESTEPS.set(torch.tensor([1.0, 0.0]))
        try:
            out = layer(x)
        finally:
            _TIMESTEPS.reset(token)
        # t=1 enables rank 1 only; t=0 enables both. The second rank changes the result.
        self.assertEqual(tuple(out.shape), (2, 1, 4))
        self.assertFalse(torch.equal(out[0], out[1]))

    def test_parser_rejects_non_tlokr_metadata(self):
        with self.assertRaisesRegex(TLoKrFormatError, "anima_adapter_type"):
            parse_tlokr_state(_state(), {})

    def test_apply_replaces_exact_target_on_a_patcher_clone(self):
        class DiffusionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Module()])
                self.blocks[0].mlp = nn.Module()
                self.blocks[0].mlp.layer1 = nn.Linear(4, 4, bias=False)
                nn.init.zeros_(self.blocks[0].mlp.layer1.weight)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.diffusion_model = DiffusionModel()

        class Patcher:
            def __init__(self, model):
                self.model = model
                self.object_patches = {}
                self.model_options = {}
                self.size = 0

            def clone(self):
                clone = Patcher(self.model)
                clone.object_patches = self.object_patches.copy()
                clone.model_options = self.model_options.copy()
                clone.size = self.size
                return clone

            def get_model_object(self, path):
                if path in self.object_patches:
                    return self.object_patches[path]
                value = self.model
                for part in path.split("."):
                    value = getattr(value, part)
                return value

            def add_object_patch(self, path, value):
                self.object_patches[path] = value

            def set_model_unet_function_wrapper(self, wrapper):
                self.model_options["model_function_wrapper"] = wrapper

            def model_size(self):
                return 64

        metadata = {"anima_adapter_type": "tlokr", "anima_adapter_format": "1"}
        adapter = next(iter(parse_tlokr_state(_state(), metadata).values()))
        original = Patcher(Model())
        patched, count = apply_tlokr(original, {adapter.prefix: adapter}, 1.0)
        self.assertEqual(count, 1)
        self.assertIsNot(patched, original)
        self.assertIsInstance(
            patched.get_model_object("diffusion_model.blocks.0.mlp.layer1"), TLoKrLinear
        )
        self.assertTrue(patched.model_options["_comfyui_tlokr_timestep_wrapper"])
        self.assertGreater(patched.size, original.model_size())
