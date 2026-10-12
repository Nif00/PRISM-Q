"""Run with ComfyUI's Python: python tests/test_dynamic_vram.py /path/to/ComfyUI."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

root = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0], '--cpu']
sys.path.insert(0, str(root))

import torch
import comfy.memory_management
import comfy.model_patcher
import comfy.model_prefetch
import comfy.ops
import comfy.utils
from comfy.weight_adapter.lora import LoRAAdapter
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

pack = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prism_test_pack', pack / '__init__.py',
                                             submodule_search_locations=[str(pack)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
pt = package.prism_trellis
sys.argv = [sys.argv[0]]


def requested_seed(layer):
    """The stochastic-rounding seed ComfyUI's resolver requests when it requantizes a patched layer."""
    return comfy.utils.string_to_seed(layer.seed_key)


class DynamicVRAMTests(unittest.TestCase):
    def layer(self):
        factory = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32)
        layer = factory.Linear(256, 16, bias=False, device='cpu')
        layer.load_state_dict({
            'weight': torch.zeros(16, 96, dtype=torch.uint8),
            'weight_level': torch.zeros(1, 16, dtype=torch.uint8),
            'weight_scale': torch.ones(16, 1),
            'comfy_quant': pt.td.trellis_file.write_conf({
                'format': pt.FORMAT, 'convrot': True, 'convrot_groupsize': 256,
            }),
        })
        layer.seed_key = 'test_layer'
        return layer

    def prefetch(self, layer, signature=True):
        geometry = [layer.weight, None]
        buffer = torch.empty(comfy.memory_management.vram_aligned_size(geometry), dtype=torch.uint8)
        views = comfy.memory_management.interpret_gathered_like(geometry, buffer)
        views[0].copy_(layer.weight)
        layer._prefetch = {'resident': False, 'signature': object() if signature else None,
                           'xfer_dest': buffer, 'needs_cast': False, 'cast_geometry': geometry}
        return views[0]

    def patch(self, layer, strength=1.0):
        delta = torch.linspace(-0.5, 0.5, 4096).reshape(16, 256)
        layer.weight_lowvram_function = comfy.model_patcher.LowVramPatch(
            'test_layer.weight', {'test_layer.weight': [(strength, ('diff', (delta,)), 1.0, None, None)]})
        return delta

    def resolve(self, layer, want_requant=True):
        return comfy.ops.resolve_cast_module_with_vbar(
            layer, torch.float32, torch.device('cpu'), torch.float32, torch.float32, want_requant)[0]

    def test_packed_patch_uses_transient_storage_and_preserves_source(self):
        layer = self.layer()
        original = layer.weight.dequantize()
        delta = self.patch(layer)
        destination = self.prefetch(layer)
        signature = layer._prefetch['signature']
        packed = destination._qdata.clone()
        levels = destination._params.levels.clone()
        result = self.resolve(layer)
        expected = layer.weight.requantize_from_float(original + delta, scale='recalculate',
                                                     stochastic_rounding=requested_seed(layer))
        self.assertTrue(result._params.plain)
        self.assertEqual(result._qdata.shape, (16, 256))
        torch.testing.assert_close(result._qdata, expected._qdata)
        torch.testing.assert_close(result._params.scale, expected._params.scale)
        torch.testing.assert_close(destination._qdata, packed)
        torch.testing.assert_close(destination._params.levels, levels)
        self.assertFalse(layer._prefetch['resident'])
        self.assertIsNone(layer._v_signature)
        self.assertIs(layer._prefetch['signature'], signature)
        # Re-prefetch, change strength, then remove the patch: no stale patched cache.
        self.patch(layer, strength=0.5)
        self.prefetch(layer)
        half = self.resolve(layer)
        expected = layer.weight.requantize_from_float(original + delta * 0.5, scale='recalculate',
                                                     stochastic_rounding=requested_seed(layer))
        torch.testing.assert_close(half._qdata, expected._qdata)
        layer.weight_lowvram_function = None
        self.prefetch(layer)
        self.resolve(layer)
        self.assertTrue(layer._prefetch['resident'])

    def test_packed_patch_without_requant_returns_patched_float(self):
        layer = self.layer()
        original = layer.weight.dequantize()
        delta = self.patch(layer)
        self.prefetch(layer)
        torch.testing.assert_close(self.resolve(layer, want_requant=False), original + delta)
        self.assertFalse(layer._prefetch['resident'])

    def test_unpatched_packed_weights_keep_resident_cache(self):
        layer = self.layer()
        self.prefetch(layer)
        result = self.resolve(layer)
        self.assertFalse(result._params.plain)
        self.assertTrue(layer._prefetch['resident'])
        self.assertIsNotNone(layer._v_signature)
        self.assertIs(self.resolve(layer), result)

    def test_plain_prism_patch_keeps_resident_cache(self):
        layer = self.layer()
        weight = layer.weight.requantize_from_float(layer.weight.dequantize(), scale='recalculate')
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.patch(layer)
        self.prefetch(layer)
        result = self.resolve(layer)
        self.assertTrue(layer._prefetch['resident'])
        self.assertIs(self.resolve(layer), layer._v_weight)
        torch.testing.assert_close(layer._v_weight._qdata, result._qdata)

    def test_native_int8_patch_keeps_resident_cache(self):
        layer = self.layer()
        qdata, params = TensorWiseINT8Layout.quantize(torch.randn(16, 256), is_weight=True, per_channel=True)
        layer.weight = torch.nn.Parameter(QuantizedTensor(qdata, 'TensorWiseINT8Layout', params), requires_grad=False)
        self.patch(layer)
        self.prefetch(layer)
        result = self.resolve(layer)
        self.assertTrue(layer._prefetch['resident'])
        self.assertIs(self.resolve(layer), layer._v_weight)
        torch.testing.assert_close(layer._v_weight._qdata, result._qdata)

    def patcher(self, layer):
        model = torch.nn.Module()
        model.add_module('test_layer', layer)
        return SimpleNamespace(model=model, patches=layer.weight_lowvram_function.patches)

    def test_after_load_preserves_full_diff_patch(self):
        layer = self.layer()
        self.patch(layer)
        lowvram = layer.weight_lowvram_function
        pt.prism_trellis_lora._set(layer, [(torch.ones(16, 1), torch.ones(1, 256), 1.0, None)])
        pt.prism_trellis_lora._after_load(self.patcher(layer))
        self.assertIs(layer.weight_lowvram_function, lowvram)
        self.assertEqual(layer.prism_lora, [])

    def test_plain_lora_keeps_compact_branch_and_resident_cache(self):
        layer = self.layer()
        self.patch(layer)
        up = torch.ones(16, 2) * 0.1
        down = torch.ones(2, 256) * 0.1
        adapter = LoRAAdapter(set(), (up, down, None, None, None, None))
        patcher = self.patcher(layer)
        patcher.patches['test_layer.weight'] = [(0.5, adapter, 1.0, None, None)]
        pt.prism_trellis_lora._after_load(patcher)
        self.assertIsNone(layer.weight_lowvram_function)
        self.prefetch(layer)
        result = self.resolve(layer)
        self.assertFalse(result._params.plain)
        self.assertTrue(layer._prefetch['resident'])
        x = torch.ones(3, 256)
        out = pt.prism_trellis_lora.add_lora(layer, x, torch.zeros(3, 16))
        torch.testing.assert_close(out, (x @ down.t()) @ up.t() * 0.5)

    def test_dora_patch_uses_transient_storage(self):
        layer = self.layer()
        original = layer.weight.dequantize()
        up = torch.ones(16, 2) * 0.1
        down = torch.ones(2, 256) * 0.1
        adapter = LoRAAdapter(set(), (up, down, None, None, original.norm(dim=1, keepdim=True), None))
        patches = {'test_layer.weight': [(0.5, adapter, 1.0, None, None)]}
        layer.weight_lowvram_function = comfy.model_patcher.LowVramPatch('test_layer.weight', patches)
        expected_float = layer.weight_lowvram_function(original.clone())
        self.assertFalse(torch.equal(expected_float, original))
        pt.prism_trellis_lora._after_load(self.patcher(layer))
        self.prefetch(layer)
        result = self.resolve(layer)
        expected = layer.weight.requantize_from_float(expected_float, scale='recalculate',
                                                     stochastic_rounding=requested_seed(layer))
        torch.testing.assert_close(result._qdata, expected._qdata)
        torch.testing.assert_close(result._params.scale, expected._params.scale)
        self.assertTrue(result._params.plain)
        self.assertFalse(layer._prefetch['resident'])

    def test_prefetch_cleanup_releases_the_original_allocation(self):
        layer = self.layer()
        self.patch(layer)
        self.prefetch(layer)
        layer._v = object()
        comfy.ops.resolve_cast_module_with_vbar(
            layer, torch.float32, torch.device('cpu'), torch.float32, None, False, return_weights=False)
        self.assertFalse(layer._prefetch['resident'])
        with patch('comfy_aimdo.model_vbar.vbar_unpin') as unpin:
            comfy.model_prefetch.cleanup_prefetched_modules(layer, [layer])
        unpin.assert_called_once_with(layer._v)
        self.assertFalse(hasattr(layer, '_prefetch'))

    def test_patch_failure_preserves_cleanup_signature(self):
        layer = self.layer()
        def fail(weight):
            raise ValueError('patch failed')
        layer.weight_lowvram_function = fail
        self.prefetch(layer)
        signature = layer._prefetch['signature']
        with self.assertRaisesRegex(ValueError, 'patch failed'):
            self.resolve(layer)
        self.assertIs(layer._prefetch['signature'], signature)

    def test_nonresident_allocation_without_signature_still_works(self):
        layer = self.layer()
        self.patch(layer)
        self.prefetch(layer, signature=False)
        result = self.resolve(layer)
        self.assertTrue(result._params.plain)
        self.assertFalse(layer._prefetch['resident'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
