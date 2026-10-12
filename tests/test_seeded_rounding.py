"""Run: ComfyUI/.venv/bin/python tests/test_seeded_rounding.py /path/to/ComfyUI [cpu|cuda:0]."""
import importlib.util
from pathlib import Path
import sys
import unittest

root = Path(sys.argv[1]).resolve()
device_name = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
sys.argv = [sys.argv[0], '--cpu'] if device_name == 'cpu' else [sys.argv[0]]
sys.path.insert(0, str(root))

import torch
import comfy.options
comfy.options.enable_args_parsing()
import comfy.memory_management
import comfy.model_patcher
import comfy.ops
import comfy.utils

pack = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prism_rounding_test', pack / '__init__.py',
                                             submodule_search_locations=[str(pack)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
pt = package.prism_trellis
sys.argv = [sys.argv[0]]


class SeededRoundingTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(device_name)
        self.generator = torch.Generator().manual_seed(20261011)

    def layers(self, convrot=True, groupsize=256, dtype=torch.float32):
        sd = {
            'weight': torch.randint(256, (32, 96), dtype=torch.uint8, generator=self.generator),
            'weight_level': torch.randint(32, (2, 16), dtype=torch.uint8, generator=self.generator),
            'weight_scale': torch.rand(32, 1, generator=self.generator) * 0.01 + 0.001,
            'comfy_quant': pt.td.trellis_file.write_conf({
                'format': pt.FORMAT, 'convrot': convrot, 'convrot_groupsize': groupsize,
            }),
        }
        expanded = pt.td.trellis_file.expand({'projection.' + key: value.clone() for key, value in sd.items()}, device='cpu')
        expanded = {key.removeprefix('projection.'): value for key, value in expanded.items()}
        factory = comfy.ops.mixed_precision_ops(compute_dtype=dtype)
        prism = factory.Linear(256, 32, bias=False, device=self.device)
        native = factory.Linear(256, 32, bias=False, device=self.device)
        prism.load_state_dict(sd)
        native.load_state_dict(expanded)
        self.assertEqual(native.weight._layout_cls, 'TensorWiseINT8Layout')
        torch.testing.assert_close(prism.weight.dequantize(), native.weight.dequantize(), rtol=0, atol=0)
        return prism, native

    def patched_float(self, layer):
        delta = (torch.randn(32, 4, generator=self.generator) @ torch.randn(4, 256, generator=self.generator)) * 0.01
        return layer.weight.dequantize() + delta.to(self.device, layer.weight.dtype)

    def assert_weight(self, actual, expected):
        self.assertTrue(actual._params.plain)
        self.assertEqual(actual._params.levels.numel(), 0)
        self.assertEqual(actual._params.convrot, expected._params.convrot)
        self.assertEqual(actual._params.convrot_groupsize, expected._params.convrot_groupsize)
        self.assertEqual(actual._params.orig_shape, expected._params.orig_shape)
        self.assertEqual(actual._params.orig_dtype, expected._params.orig_dtype)
        torch.testing.assert_close(actual._qdata, expected._qdata, rtol=0, atol=0)
        torch.testing.assert_close(actual._params.scale, expected._params.scale, rtol=0, atol=0)
        torch.testing.assert_close(actual.dequantize(), expected.dequantize(), rtol=0, atol=0)

    def test_set_weight_matches_native_for_seeds_and_layouts(self):
        for convrot, groupsize in ((True, 256), (True, 64), (False, 256)):
            prism, native = self.layers(convrot, groupsize)
            value = self.patched_float(native)
            # CUDA's unsigned seed argument rejects negatives even in native INT8.
            seeds = (None, 0, 123, 456, comfy.utils.string_to_seed('projection'))
            if self.device.type == 'cpu':
                seeds += (-1,)
            for seed in seeds:
                with self.subTest(convrot=convrot, groupsize=groupsize, seed=seed):
                    self.assert_weight(prism.set_weight(value, seed=seed, return_weight=True),
                                       native.set_weight(value, seed=seed, return_weight=True))

    def test_positive_seeds_are_repeatable_and_change_weights(self):
        prism, native = self.layers()
        value = self.patched_float(native)
        first = prism.set_weight(value, seed=123, return_weight=True)
        repeat = prism.set_weight(value, seed=123, return_weight=True)
        other = prism.set_weight(value, seed=456, return_weight=True)
        torch.testing.assert_close(first._qdata, repeat._qdata, rtol=0, atol=0)
        self.assertGreater(torch.count_nonzero(first._qdata != other._qdata).item(), 0)

    def test_default_options_preserve_deterministic_rounding_and_input(self):
        prism, native = self.layers()
        value = self.patched_float(native)
        original = value.clone()
        actual = prism.weight.requantize_from_float(value, scale='recalculate', inplace_ops=True)
        expected = native.weight.requantize_from_float(value, scale='recalculate', inplace_ops=True)
        self.assert_weight(actual, expected)
        torch.testing.assert_close(value, original, rtol=0, atol=0)

    def test_repatch_plain_weights_and_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                prism, native = self.layers(groupsize=64, dtype=dtype)
                value = self.patched_float(native)
                prism.set_weight(value, seed=123)
                native.set_weight(value, seed=123)
                self.assert_weight(prism.weight, native.weight)
                value = self.patched_float(native)
                prism.set_weight(value, seed=456)
                native.set_weight(value, seed=456)
                self.assert_weight(prism.weight, native.weight)

    def test_model_patcher_full_diff_uses_native_seed(self):
        prism, native = self.layers()
        delta = self.patched_float(native) - native.weight.dequantize()
        results = []
        for layer in (prism, native):
            model = torch.nn.Module()
            model.add_module('projection', layer)
            patcher = comfy.model_patcher.ModelPatcher(model, self.device, torch.device('cpu'))
            patcher.add_patches({'projection.weight': ('diff', (delta,))}, strength_patch=0.5)
            results.append(patcher.patch_weight_to_device('projection.weight', return_weight=True))
        self.assert_weight(*results)

    def test_dynamic_resolver_uses_module_seed(self):
        prism, native = self.layers()
        delta = self.patched_float(native) - native.weight.dequantize()
        results = []
        for layer in (prism, native):
            layer.seed_key = 'projection'
            layer.weight_lowvram_function = comfy.model_patcher.LowVramPatch(
                'projection.weight', {'projection.weight': [(0.5, ('diff', (delta,)), 1.0, None, None)]})
            geometry = [layer.weight, None]
            buffer = torch.empty(comfy.memory_management.vram_aligned_size(geometry), dtype=torch.uint8, device=self.device)
            views = comfy.memory_management.interpret_gathered_like(geometry, buffer)
            views[0].copy_(layer.weight)
            # No resident writeback: this test also works independently of the issue #1 fix.
            layer._prefetch = {'resident': False, 'signature': None, 'xfer_dest': buffer,
                               'needs_cast': False, 'cast_geometry': geometry}
            results.append(comfy.ops.resolve_cast_module_with_vbar(
                layer, torch.float32, self.device, torch.float32, torch.float32, True)[0])
        self.assert_weight(*results)


if __name__ == '__main__':
    unittest.main(verbosity=2)
