"""Run: ComfyUI/.venv/bin/python tests/test_bypass_forward.py /path/to/ComfyUI [cpu|cuda:0]."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

root = Path(sys.argv[1]).resolve()
device_name = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
sys.argv = [sys.argv[0], '--cpu'] if device_name == 'cpu' else [sys.argv[0]]
sys.path.insert(0, str(root))

import torch
import comfy.options
comfy.options.enable_args_parsing()
import comfy.ops
from comfy.weight_adapter.bypass import BypassForwardHook
from comfy.weight_adapter.lora import LoRAAdapter

pack = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prism_bypass_test', pack / '__init__.py',
                                             submodule_search_locations=[str(pack)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
pt = package.prism_trellis
sys.argv = [sys.argv[0]]


class BypassForwardTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(device_name)
        self.dtype = torch.float32 if self.device.type == 'cpu' else torch.bfloat16
        self.generator = torch.Generator().manual_seed(42)

    def random(self, *shape):
        return torch.randn(*shape, generator=self.generator).to(self.device, self.dtype)

    def layer(self):
        factory = comfy.ops.mixed_precision_ops(compute_dtype=self.dtype)
        layer = factory.Linear(256, 16, bias=True, device=self.device)
        layer.load_state_dict({
            'weight': torch.zeros(16, 96, dtype=torch.uint8, device=self.device),
            'weight_level': torch.zeros(1, 16, dtype=torch.uint8, device=self.device),
            'weight_scale': torch.ones(16, 1, device=self.device),
            'bias': self.random(16),
            'comfy_quant': pt.td.trellis_file.write_conf({
                'format': pt.FORMAT, 'convrot': True, 'convrot_groupsize': 256,
            }),
        })
        return layer

    def adapter(self):
        return LoRAAdapter(set(), (self.random(16, 2) * 0.2, self.random(2, 256) * 0.2,
                                   2.0, None, None, None))

    def assert_output(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_swiglu_calls_real_bypass_lora(self):
        layer = self.layer()
        x = self.random(2, 3, 512)
        activated = comfy.ops._eager_input_act(x, 'swiglu')
        base = layer(activated)
        adapter = self.adapter()
        hook = BypassForwardHook(layer, adapter)
        hook.inject()
        try:
            expected = adapter.g(base + adapter.h(activated, base))
            self.assertFalse(torch.equal(expected, base))
            actual = comfy.ops.linear_input_act(layer, x, 'swiglu')
            self.assert_output(actual, expected)
        finally:
            hook.eject()
        self.assertFalse(layer.comfy_force_forward)

    def test_residual_and_strength_apply_after_bypass_forward(self):
        layer = self.layer()
        x = self.random(2, 512)
        residual = self.random(2, 16)
        scale = self.random(16)
        for strength in (-0.25, 0.0, 0.5, 1.0):
            with self.subTest(strength=strength):
                hook = BypassForwardHook(layer, self.adapter(), multiplier=strength)
                hook.inject()
                try:
                    expected = pt._linear_input_act(layer, x, 'swiglu', residual=residual, residual_scale=scale)
                    actual = comfy.ops.linear_input_act(layer, x, 'swiglu', residual=residual, residual_scale=scale)
                    self.assert_output(actual, expected)
                finally:
                    hook.eject()

    def test_other_activations_preserve_bypass_and_parameters(self):
        layer = self.layer()
        x = self.random(2, 256)
        norm_weight = self.random(256)
        hook = BypassForwardHook(layer, self.adapter())
        hook.inject()
        try:
            for activation, weight, eps in ((None, None, 0.0), ('gelu_tanh', None, 0.0),
                                             ('rms_norm', norm_weight, 1e-4)):
                with self.subTest(activation=activation):
                    expected = pt._linear_input_act(layer, x, activation, weight, eps)
                    actual = comfy.ops.linear_input_act(layer, x, activation, weight, eps)
                    self.assert_output(actual, expected)
        finally:
            hook.eject()

    def test_flag_preserves_other_forward_hooks(self):
        layer = self.layer()
        x = self.random(2, 512)
        layer.comfy_force_forward = True
        calls = []
        handle = layer.register_forward_hook(lambda module, args, output: calls.append(True) or output + 3)
        try:
            expected = pt._linear_input_act(layer, x, 'swiglu')
            calls.clear()
            actual = comfy.ops.linear_input_act(layer, x, 'swiglu')
            self.assertEqual(calls, [True])
            self.assert_output(actual, expected)
        finally:
            handle.remove()

    def test_existing_prism_lora_branch_is_applied_once_with_bypass(self):
        layer = self.layer()
        pt.prism_trellis_lora._set(layer, [(self.random(16, 2), self.random(2, 256), 0.25, None)])
        x = self.random(2, 512)
        hook = BypassForwardHook(layer, self.adapter())
        hook.inject()
        try:
            expected = pt._linear_input_act(layer, x, 'swiglu')
            actual = comfy.ops.linear_input_act(layer, x, 'swiglu')
            self.assert_output(actual, expected)
        finally:
            hook.eject()

    def test_ejection_restores_fused_path(self):
        layer = self.layer()
        x = self.random(2, 512)
        expected = comfy.ops.linear_input_act(layer, x, 'swiglu')
        hook = BypassForwardHook(layer, self.adapter())
        hook.inject()
        hook.eject()
        self.assertFalse(layer.comfy_force_forward)
        self.assert_output(comfy.ops.linear_input_act(layer, x, 'swiglu'), expected)
        with patch.object(pt, '_linear_input_act', side_effect=AssertionError('unexpected fallback')):
            comfy.ops.linear_input_act(layer, x, 'swiglu')


if __name__ == '__main__':
    unittest.main(verbosity=2)
