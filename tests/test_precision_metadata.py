"""Run: ComfyUI/.venv/bin/python tests/test_precision_metadata.py /path/to/ComfyUI [cpu|cuda:0]."""
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
import comfy.ops

pack = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prism_precision_test', pack / '__init__.py',
                                             submodule_search_locations=[str(pack)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
pt = package.prism_trellis
sys.argv = [sys.argv[0]]


class PrecisionMetadataTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(device_name)
        self.dtype = torch.float32 if self.device.type == 'cpu' else torch.bfloat16
        self.generator = torch.Generator().manual_seed(20261011)
        self.sd = {
            'weight': torch.randint(256, (32, 96), dtype=torch.uint8, generator=self.generator),
            'weight_level': torch.randint(32, (2, 16), dtype=torch.uint8, generator=self.generator),
            'weight_scale': torch.rand(32, 1, generator=self.generator) * 0.01 + 0.001,
            'bias': torch.randn(32, generator=self.generator),
        }

    def random(self, *shape):
        return torch.randn(*shape, generator=self.generator).to(self.device, self.dtype)

    def layer(self, metadata=None, native=False, global_precision=False, plain=False):
        conf = {'format': pt.FORMAT, 'convrot': True, 'convrot_groupsize': 256}
        if metadata is not None:
            conf['full_precision_matrix_mult'] = metadata
        sd = {key: value.clone() for key, value in self.sd.items()}
        sd['comfy_quant'] = pt.td.trellis_file.write_conf(conf)
        if plain:
            sd['weight'] = pt.td.trellis_file.decode_layer(sd['weight'], sd.pop('weight_level'))
        if native:
            if plain:
                sd['comfy_quant'] = pt.td.trellis_file.with_format(sd['comfy_quant'], 'int8_tensorwise')
            else:
                expanded = pt.td.trellis_file.expand({'projection.' + key: value for key, value in sd.items()}, device='cpu')
                sd = {key.removeprefix('projection.'): value for key, value in expanded.items()}
        factory = comfy.ops.mixed_precision_ops(compute_dtype=self.dtype, full_precision_mm=global_precision)
        layer = factory.Linear(256, 32, bias=True, device=self.device)
        layer.load_state_dict(sd)
        return layer

    def assert_output(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_loaded_flags_match_native_and_preserve_global_setting(self):
        for metadata in (None, False, True):
            for global_precision in (False, True):
                with self.subTest(metadata=metadata, global_precision=global_precision):
                    prism = self.layer(metadata, global_precision=global_precision)
                    native = self.layer(metadata, native=True, global_precision=global_precision)
                    expected = bool(metadata or global_precision)
                    self.assertEqual(prism._full_precision_mm_config, bool(metadata))
                    self.assertEqual(prism._full_precision_mm, expected)
                    self.assertEqual(prism._full_precision_mm, native._full_precision_mm)
                    self.assertEqual(prism._full_precision_mm_config, native._full_precision_mm_config)

    def test_metadata_true_linear_matches_float_reference(self):
        prism = self.layer(True)
        native = self.layer(True, native=True)
        quantized = self.layer(False)
        x = self.random(2, 3, 256)
        self.assert_output(prism.weight.dequantize(), native.weight.dequantize())
        reference = torch.nn.functional.linear(x, native.weight.dequantize(), native.bias)
        self.assert_output(native(x), reference)
        self.assert_output(prism(x), reference)
        self.assertFalse(torch.equal(quantized(x), reference))

    def test_metadata_true_activation_and_residual_match_native(self):
        prism = self.layer(True)
        native = self.layer(True, native=True)
        residual, scale = self.random(2, 3, 32), self.random(32)
        norm_weight = self.random(256)
        for activation, width, weight, eps in ((None, 256, None, 0.0), ('swiglu', 512, None, 0.0),
                                               ('gelu_tanh', 256, None, 0.0), ('rms_norm', 256, norm_weight, 1e-4)):
            with self.subTest(activation=activation):
                x = self.random(2, 3, width)
                expected = comfy.ops.linear_input_act(native, x, activation, weight, eps, residual, scale)
                actual = comfy.ops.linear_input_act(prism, x, activation, weight, eps, residual, scale)
                self.assert_output(actual, expected)

    def test_missing_and_false_metadata_preserve_quantized_outputs(self):
        x = self.random(2, 256)
        activation_input = self.random(2, 512)
        residual, scale = self.random(2, 32), self.random(32)
        for metadata in (None, False):
            with self.subTest(metadata=metadata):
                prism = self.layer(metadata)
                native = self.layer(metadata, native=True)
                self.assert_output(prism(x), native(x))
                self.assert_output(comfy.ops.linear_input_act(prism, activation_input, 'swiglu',
                                                             residual=residual, residual_scale=scale),
                                   comfy.ops.linear_input_act(native, activation_input, 'swiglu',
                                                             residual=residual, residual_scale=scale))

    def test_global_precision_matches_float_when_metadata_is_missing_or_false(self):
        x = self.random(2, 256)
        for metadata in (None, False):
            with self.subTest(metadata=metadata):
                prism = self.layer(metadata, global_precision=True)
                native = self.layer(metadata, native=True, global_precision=True)
                reference = torch.nn.functional.linear(x, native.weight.dequantize(), native.bias)
                self.assert_output(prism(x), reference)
                self.assert_output(prism(x), native(x))

    def test_plain_int8_load_respects_precision_metadata(self):
        x = self.random(2, 256)
        prism = self.layer(True, plain=True)
        native = self.layer(True, native=True, plain=True)
        self.assertTrue(prism.weight._params.plain)
        self.assert_output(prism(x), native(x))

    def test_precision_metadata_survives_state_dict_roundtrip(self):
        x = self.random(2, 256)
        for metadata in (False, True):
            with self.subTest(metadata=metadata):
                original = self.layer(metadata)
                saved = original.state_dict()
                conf = pt.td.trellis_file.read_conf(saved['comfy_quant'])
                self.assertEqual(conf.get('full_precision_matrix_mult', False), metadata)
                factory = comfy.ops.mixed_precision_ops(compute_dtype=self.dtype)
                restored = factory.Linear(256, 32, bias=True, device=self.device)
                restored.load_state_dict(saved)
                self.assertEqual(restored._full_precision_mm, metadata)
                self.assert_output(restored(x), original(x))


if __name__ == '__main__':
    unittest.main(verbosity=2)
