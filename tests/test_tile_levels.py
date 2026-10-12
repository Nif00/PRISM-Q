"""Run: ComfyUI/.venv/bin/python tests/test_tile_levels.py /path/to/ComfyUI [cpu|cuda:0]."""
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

pack = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prism_levels_test', pack / '__init__.py',
                                             submodule_search_locations=[str(pack)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
pt = package.prism_trellis
td = pt.td
sys.argv = [sys.argv[0]]


class TileLevelTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(device_name)

    def state_dict(self, levels):
        return {
            'weight': torch.zeros(16, 96, dtype=torch.uint8, device=self.device),
            'weight_level': levels,
            'weight_scale': torch.ones(16, 1, device=self.device),
            'comfy_quant': td.trellis_file.write_conf({
                'format': pt.FORMAT, 'convrot': True, 'convrot_groupsize': 256,
            }),
        }

    def load(self, levels):
        factory = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32)
        layer = factory.Linear(256, 16, bias=False, device=self.device)
        holder = torch.nn.Module()
        holder.add_module('projection', layer)
        holder.load_state_dict({'projection.' + key: value for key, value in self.state_dict(levels).items()})
        return layer

    def invalid_levels(self):
        for value, dtype in ((32, torch.uint8), (255, torch.uint8), (-1, torch.int8),
                             (-256, torch.int64), (256, torch.int64), (288, torch.int64),
                             (0, torch.int32), (0.5, torch.float32), (float('nan'), torch.float32),
                             (float('inf'), torch.float32), (0, torch.bool)):
            yield value, dtype, torch.full((1, 16), value, dtype=dtype, device=self.device)

    def test_loader_rejects_invalid_levels_before_payload_conversion(self):
        for value, dtype, levels in self.invalid_levels():
            with self.subTest(value=value, dtype=dtype):
                with patch.object(td, 'to_words', wraps=td.to_words) as conversion:
                    with self.assertRaisesRegex(ValueError, 'projection.*weight_level'):
                        self.load(levels)
                    conversion.assert_not_called()

    def test_loader_checks_every_tile(self):
        levels = torch.zeros(1, 16, dtype=torch.uint8, device=self.device)
        levels[0, -1] = 32
        with self.assertRaisesRegex(ValueError, 'projection.*weight_level'):
            self.load(levels)

    def test_loader_preserves_valid_levels_and_decoded_weights(self):
        for value in (0, 31):
            with self.subTest(value=value):
                levels = torch.full((1, 16), value, dtype=torch.uint8, device=self.device)
                layer = self.load(levels)
                torch.testing.assert_close(layer.weight._params.levels, levels, rtol=0, atol=0)
                expected = td.trellis_file.decode_layer(torch.zeros(1, 16, 96, dtype=torch.uint8), levels.cpu())
                actual = pt.PrismTrellisLayout.int8_weight(layer.weight._qdata, layer.weight._params)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_direct_decoder_rejects_invalid_levels_before_backend_dispatch(self):
        words = td.to_words(torch.zeros(1, 16, 96, dtype=torch.uint8, device=self.device))
        for value, dtype, levels in self.invalid_levels():
            with self.subTest(value=value, dtype=dtype):
                # Never send invalid indices to either CUDA or the reference decoder, even before the fix.
                with patch.object(td, 'available', return_value=False) as available:
                    with patch.object(td.trellis_file, 'decode_layer', return_value=torch.empty(0)) as fallback:
                        with self.assertRaisesRegex(ValueError, 'weight_level'):
                            td.decode_words(words, levels)
                        available.assert_not_called()
                        fallback.assert_not_called()

    def test_direct_decoder_rejects_level_shape_and_device_mismatch(self):
        words = td.to_words(torch.zeros(1, 16, 96, dtype=torch.uint8, device=self.device))
        bad_levels = [torch.zeros(1, 15, dtype=torch.uint8, device=self.device),
                      torch.zeros(16, dtype=torch.uint8, device=self.device),
                      torch.empty(1, 16, dtype=torch.uint8, device='meta')]
        for levels in bad_levels:
            with self.subTest(shape=levels.shape, device=levels.device):
                with patch.object(td, 'available', return_value=False):
                    with patch.object(td.trellis_file, 'decode_layer', return_value=torch.empty(0)):
                        with self.assertRaisesRegex(ValueError, 'weight_level'):
                            td.decode_words(words, levels)

    def test_all_levels_and_bit_widths_match_reference(self):
        generator = torch.Generator().manual_seed(303)
        for kb in range(1, 6):
            with self.subTest(kb=kb):
                payload = torch.randint(256, (2, 17, 32 * kb), dtype=torch.uint8, generator=generator)
                levels = (torch.arange(34) % 32).to(torch.uint8).reshape(2, 17)
                expected = td.trellis_file.decode_layer(payload, levels)
                words = td.to_words(payload.to(self.device))
                # The public decoder accepts strided levels by making its own contiguous input.
                strided = torch.zeros(2, 34, dtype=torch.uint8, device=self.device)
                strided[:, ::2] = levels.to(self.device)
                actual = td.decode_words(words, strided[:, ::2])
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                contiguous = levels.to(self.device)
                out = torch.empty(32, 272, dtype=torch.int8, device=self.device)
                actual = td.decode_words(words, contiguous, out=out)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                if self.device.type == 'cuda':
                    self.assertIs(actual, out)
                    self.assertNotIn(self.device.index, td._FAILED)

    def test_plain_int8_reload_still_works_without_levels(self):
        factory = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32)
        layer = factory.Linear(256, 16, bias=False, device=self.device)
        sd = self.state_dict(None)
        del sd['weight_level']
        sd['weight'] = torch.full((16, 256), 7, dtype=torch.int8, device=self.device)
        layer.load_state_dict(sd)
        self.assertTrue(layer.weight._params.plain)
        self.assertEqual(layer.weight._params.levels.numel(), 0)
        torch.testing.assert_close(layer.weight._qdata, sd['weight'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
