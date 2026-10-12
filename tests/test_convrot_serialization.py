"""Run with ComfyUI's Python: tests/test_convrot_serialization.py /path/to/ComfyUI [cpu|cuda:0]."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

root = Path(sys.argv[1]).resolve()
device_name = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
sys.argv = [sys.argv[0], '--cpu'] if device_name == 'cpu' else [sys.argv[0]]
sys.path.insert(0, str(root))
import torch
import comfy.options
comfy.options.enable_args_parsing()
import comfy.ops
from safetensors.torch import load_file, save_file

pack = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prism_save_test', pack / '__init__.py',
                                             submodule_search_locations=[str(pack)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
pt = package.prism_trellis
sys.argv = [sys.argv[0]]


class ConvRotSerializationTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(device_name)
        self.dtype = torch.float32 if self.device.type == 'cpu' else torch.bfloat16
        self.generator = torch.Generator().manual_seed(20261011)
        self.factory = comfy.ops.mixed_precision_ops(compute_dtype=self.dtype)
        self.x = torch.randn(2, 256, generator=self.generator).to(self.device, self.dtype)

    def layer(self, convrot=True, groupsize=256, native=False, plain=False):
        sd = {
            'weight': torch.randint(256, (32, 96), dtype=torch.uint8, generator=self.generator),
            'weight_level': torch.randint(32, (2, 16), dtype=torch.uint8, generator=self.generator),
            'weight_scale': torch.rand(32, 1, generator=self.generator) * 0.01 + 0.001,
            'bias': torch.randn(32, generator=self.generator),
            'comfy_quant': pt.td.trellis_file.write_conf({'format': pt.FORMAT, 'convrot': convrot,
                                                        'convrot_groupsize': groupsize}),
        }
        if plain:
            sd['weight'] = pt.td.trellis_file.decode_layer(sd['weight'], sd.pop('weight_level'))
        if native:
            expanded = pt.td.trellis_file.expand({'projection.' + k: v for k, v in sd.items()}, device='cpu')
            sd = {k.removeprefix('projection.'): v for k, v in expanded.items()}
        layer = self.factory.Linear(256, 32, bias=True, device=self.device)
        layer.load_state_dict(sd)
        return layer

    def assert_roundtrip(self, before, saved=None):
        if saved is None:
            saved = before.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'layer.safetensors')
            save_file({k: v.detach().cpu().contiguous() for k, v in saved.items()}, path)
            on_disk = load_file(path)
        after = self.factory.Linear(256, 32, bias=True, device=self.device)
        after.load_state_dict(on_disk)
        p, q = before.weight._params, after.weight._params
        self.assertEqual(p.convrot, q.convrot)
        if p.convrot or before.quant_format == pt.FORMAT:
            self.assertEqual(p.convrot_groupsize, q.convrot_groupsize)
        for a, b in zip(before.weight.layout_cls.get_plain_tensors(before.weight),
                        after.weight.layout_cls.get_plain_tensors(after.weight)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(before.weight.dequantize(), after.weight.dequantize(), rtol=0, atol=0)
        torch.testing.assert_close(before(self.x), after(self.x), rtol=0, atol=0)
        torch.testing.assert_close(comfy.ops.linear_input_act(before, self.x, 'gelu_tanh'),
                                   comfy.ops.linear_input_act(after, self.x, 'gelu_tanh'), rtol=0, atol=0)
        return on_disk

    def test_default_packed_roundtrip(self):
        self.assert_roundtrip(self.layer())

    def test_group64_packed_roundtrip(self):
        self.assert_roundtrip(self.layer(groupsize=64))

    def test_disabled_convrot_packed_roundtrip(self):
        self.assert_roundtrip(self.layer(convrot=False))

    def test_plain_int8_roundtrips(self):
        for convrot, groupsize in ((True, 256), (True, 64), (False, 256)):
            with self.subTest(convrot=convrot, groupsize=groupsize):
                self.assert_roundtrip(self.layer(convrot, groupsize, plain=True))

    def test_native_int8_controls(self):
        for convrot, groupsize in ((True, 256), (True, 64), (False, 256)):
            with self.subTest(convrot=convrot, groupsize=groupsize):
                layer = self.layer(convrot, groupsize, native=True)
                saved = self.assert_roundtrip(layer)
                conf = pt.td.trellis_file.read_conf(saved['comfy_quant'])
                self.assertEqual(conf['format'], 'int8_tensorwise')
                if not convrot:
                    self.assertNotIn('convrot', conf)

    def test_nested_prefix_and_destination(self):
        layer = self.layer(groupsize=64)
        parent = torch.nn.Module()
        parent.add_module('projection', layer)
        destination = {'sentinel': torch.tensor(42)}
        saved = parent.state_dict(destination=destination, prefix='outer.')
        self.assertIs(saved, destination)
        self.assertEqual(saved['sentinel'].item(), 42)
        self.assert_roundtrip(layer, {k.removeprefix('outer.projection.'): v for k, v in saved.items()
                                      if k.startswith('outer.projection.')})

    def test_extra_metadata_and_parameters_survive(self):
        layer = self.layer(groupsize=64)
        layer.input_scale = torch.tensor(0.25)
        layer.pre_quant_scale = torch.ones(256)
        layer._full_precision_mm_config = True
        extra = {'num_experts': 4}
        saved = comfy.ops._quantized_weight_state_dict(layer, {}, '', extra,
                                                       ('input_scale', 'pre_quant_scale'))
        conf = pt.td.trellis_file.read_conf(saved['comfy_quant'])
        self.assertEqual(conf, {'format': pt.FORMAT, 'convrot': True, 'convrot_groupsize': 64,
                                'full_precision_matrix_mult': True, 'num_experts': 4})
        self.assertEqual(extra, {'num_experts': 4})
        torch.testing.assert_close(saved['input_scale'], layer.input_scale)
        torch.testing.assert_close(saved['pre_quant_scale'], layer.pre_quant_scale)

    def test_unquantized_saver_is_unchanged(self):
        layer = torch.nn.Linear(256, 32).to(self.device, self.dtype)
        saved = comfy.ops._quantized_weight_state_dict(layer, {}, 'projection.')
        self.assertEqual(set(saved), {'projection.weight', 'projection.bias'})
        torch.testing.assert_close(saved['projection.weight'], layer.weight)


if __name__ == '__main__':
    unittest.main(verbosity=2)
