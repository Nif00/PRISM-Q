"""ComfyUI quantization format ``prism_trellis`` for packed PRISM-Q trellis checkpoints (prism_trellis_file.py).

A trellis layer is an int8_tensorwise ConvRot layer whose int8 weight is stored as its QTIP bitstream:

    {layer}.weight        uint8 [N, K*Kb/8]   16x16 tile payloads [N/16, K/16, 32*Kb] flattened to N rows
    {layer}.weight_level  uint8 [N/16, K/16]  codebook level per tile
    {layer}.weight_scale  fp32  [N, 1]        per-row scale
    {layer}.comfy_quant   {"format": "prism_trellis", "convrot": true, "convrot_groupsize": 256}

The payload stays resident (as big-endian int32 words) and each call decodes the layer just in time into a
transient int8 weight (prism_trellis_decode.py, a CUDA kernel compiled at first use with NVRTC) for comfy_kitchen's
int8_linear with ConvRot, the exact kernel call of the official INT8 ConvRot checkpoint, so outputs are bit-identical
to the expanded checkpoint while the weights take Kb + 5/256 bits in VRAM. Loading goes through the stock "Load
Diffusion Model" node via prism_trellis_hook. A plain LoRA runs as a low-rank branch next to the trellis weight
(prism_trellis_lora.py); other patch types requantize the layer to plain int8, as ConvRot does (not possible
under dynamic VRAM, where they are skipped).
"""
import dataclasses
import logging
from dataclasses import dataclass

import torch
import comfy.model_management
import comfy.ops
import comfy.quant_ops
from comfy_kitchen.tensor import QuantizedLayout, QuantizedTensor, register_layout_class, register_layout_op
from comfy_kitchen.tensor.base import BaseLayoutParams, dequantize_args
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout, _dtype_code

from . import prism_trellis_decode as td
from . import prism_trellis_hook
from . import prism_trellis_lora

FORMAT = td.trellis_file.QUANT_FORMAT
LAYOUT = 'PrismTrellisLayout'
prism_trellis_lora.LAYOUT = LAYOUT
prism_trellis_lora.install()


class PrismTrellisLayout(QuantizedLayout):
    """QTIP-INT8 trellis weights decoded per call to ConvRot int8; qdata is the resident payload words, or the plain
    int8 weight after a LoRA patch (params.plain)."""

    MIN_SM_VERSION = (7, 5)
    QUANTIZES_INPUT = False

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        levels: torch.Tensor = None
        convrot: bool = True
        convrot_groupsize: int = 256
        plain: bool = False
        transposed: bool = False

        def _tensor_fields(self):
            return ['scale', 'levels']

    @classmethod
    def requantize_kwargs(cls, qtensor):
        return {'parent': qtensor}

    @classmethod
    def quantize(cls, tensor, parent=None, **kwargs):
        """A patched (LoRA) weight leaves the trellis: requantize it to ConvRot's plain int8 like an int8 layer."""
        p = parent._params if parent is not None else None
        convrot = p.convrot if p is not None else True
        groupsize = p.convrot_groupsize if p is not None else 256
        qdata, p8 = TensorWiseINT8Layout.quantize(tensor, is_weight=True, per_channel=True, convrot=convrot,
                                                  convrot_groupsize=groupsize, **kwargs)
        levels = torch.zeros(0, dtype=torch.uint8, device=qdata.device)
        return qdata, cls.Params(scale=p8.scale, orig_dtype=p8.orig_dtype, orig_shape=p8.orig_shape, levels=levels,
                                 convrot=convrot, convrot_groupsize=groupsize, plain=True)

    @classmethod
    def int8_weight(cls, qdata, params):
        if params.plain:
            return qdata
        # Checkpoint levels are validated at load time, not on every forward.
        return td._decode_words(qdata, params.levels)

    @classmethod
    def int8_params(cls, params):
        return TensorWiseINT8Layout.Params(scale=params.scale, orig_dtype=params.orig_dtype,
                                           orig_shape=params.orig_shape, convrot=params.convrot,
                                           convrot_groupsize=params.convrot_groupsize)

    @classmethod
    def dequantize(cls, qdata, params):
        return TensorWiseINT8Layout.dequantize(cls.int8_weight(qdata, params), cls.int8_params(params))

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return cls.int8_weight(qtensor._qdata, qtensor._params), qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        """Saved as the bitstream again (or plain int8 after a patch; the loader accepts both)."""
        if params.plain:
            return {'': qdata, '_scale': params.scale}
        return {'': td.trellis_file.to_file(td.from_words(qdata)), '_scale': params.scale, '_level': params.levels}


register_layout_class(LAYOUT, PrismTrellisLayout)
comfy.quant_ops.QUANT_ALGOS[FORMAT] = {
    'storage_t': torch.uint8,
    'parameters': {'weight_scale'},
    'comfy_tensor_layout': LAYOUT,
    'quantize_input': False,
}


# ---------------------------------------------------------------- forward

def trellis_linear(x, weight, bias=None):
    p = weight._params
    out_dtype = x.dtype if x.dtype in (torch.float32, torch.float16, torch.bfloat16) else torch.bfloat16
    w8 = PrismTrellisLayout.int8_weight(weight._qdata, p)
    return torch.ops.comfy_kitchen.int8_linear(x.contiguous(), w8, p.scale, bias, _dtype_code(out_dtype),
                                               p.convrot, p.convrot_groupsize)


@register_layout_op(torch.ops.aten.t.default, PrismTrellisLayout)
def _handle_t(qt, args, kwargs):
    tensor = args[0]
    if not isinstance(tensor, QuantizedTensor):
        return torch.ops.aten.t.default(*args, **kwargs)
    old = tensor._params
    params = dataclasses.replace(old, orig_shape=(old.orig_shape[1], old.orig_shape[0]), transposed=not old.transposed)
    return QuantizedTensor(tensor._qdata, LAYOUT, params)


@register_layout_op(torch.ops.aten.linear.default, PrismTrellisLayout)
def _handle_linear(qt, args, kwargs):
    x, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else kwargs.get('bias')
    if not isinstance(weight, QuantizedTensor) or weight._params.transposed:
        return torch.nn.functional.linear(*dequantize_args((x, weight, bias)))
    if isinstance(x, QuantizedTensor):
        x = x.dequantize()
    return trellis_linear(x, weight, bias)


def _fast_rhs(a, b):
    return isinstance(b, QuantizedTensor) and not isinstance(a, QuantizedTensor) and b._params.transposed


def _untransposed(b):
    old = b._params
    params = dataclasses.replace(old, orig_shape=(old.orig_shape[1], old.orig_shape[0]), transposed=False)
    return QuantizedTensor(b._qdata, LAYOUT, params)


@register_layout_op(torch.ops.aten.mm.default, PrismTrellisLayout)
def _handle_mm(qt, args, kwargs):
    a, b = args[0], args[1]
    if not _fast_rhs(a, b):
        return torch.mm(*dequantize_args((a, b)))
    return trellis_linear(a, _untransposed(b))


@register_layout_op(torch.ops.aten.addmm.default, PrismTrellisLayout)
def _handle_addmm(qt, args, kwargs):
    bias, a, b = args[0], args[1], args[2]
    if not _fast_rhs(a, b) or kwargs.get('beta', 1) != 1 or kwargs.get('alpha', 1) != 1:
        return torch.addmm(*dequantize_args((bias, a, b)), **kwargs)
    return trellis_linear(a, _untransposed(b), bias)


_linear_input_act = comfy.ops.linear_input_act


def linear_input_act(linear, x, input_act, act_weight=None, act_eps=0.0, residual=None, residual_scale=None):
    """comfy.ops.linear_input_act for trellis layers. ComfyUI folds the input activation (Qwen-Image 2.1's SwiGLU before
    img_mlp.out) into the INT8 input quantizer, and the residual into the GEMM epilogue, only for TensorWiseINT8Layout;
    here a trellis layer takes that same int8_linear call with its decoded weight, so it stays bit-identical to the
    int8 checkpoint. Everything else goes to ComfyUI's function."""
    weight = linear.weight
    if (comfy.model_management.in_training or getattr(linear, 'comfy_force_forward', False)
            or not isinstance(weight, QuantizedTensor) or weight._layout_cls != LAYOUT
            or weight._params.transposed or getattr(linear, '_full_precision_mm', False)):
        return _linear_input_act(linear, x, input_act, act_weight, act_eps, residual, residual_scale)
    weight, bias, offload_stream = comfy.ops.cast_bias_weight(linear, x, offloadable=True, compute_dtype=x.dtype,
                                                              want_requant=True)
    try:
        if not isinstance(weight, QuantizedTensor):
            xa = comfy.ops._eager_input_act(x, input_act, act_weight, act_eps)
            out = prism_trellis_lora.add_lora(linear, xa, torch.nn.functional.linear(xa, weight, bias))
            return out if residual is None else torch.addcmul(residual, out, residual_scale)
        w8, scale = PrismTrellisLayout.get_plain_tensors(weight)
        out = comfy.quant_ops.ck.int8_linear(
            x, w8, scale, bias, x.dtype, convrot=weight._params.convrot,
            convrot_groupsize=weight._params.convrot_groupsize, input_act=input_act, input_act_weight=act_weight,
            input_act_eps=act_eps, residual=residual, residual_scale=residual_scale)
        if getattr(linear, 'prism_lora', None):  # the branch sees the activated input; the residual scale applies to it
            prism_trellis_lora.add_lora(linear, comfy.ops._eager_input_act(x, input_act, act_weight, act_eps), out,
                                        residual_scale if residual is not None else None)
        return out
    finally:
        comfy.ops.uncast_bias_weight(linear, weight, bias, offload_stream)


if not getattr(comfy.ops.linear_input_act, 'prism_trellis', False):
    linear_input_act.prism_trellis = True
    comfy.ops.linear_input_act = linear_input_act


# ---------------------------------------------------------------- loading

def _load(module, super_load, state_dict, prefix, layer_conf, local_metadata, strict,
          missing_keys, unexpected_keys, error_msgs):
    device = module.factory_kwargs['device']
    compute_dtype = module.factory_kwargs['dtype']
    layer = prefix.rstrip('.')
    loaded = [f'{prefix}weight', f'{prefix}comfy_quant', f'{prefix}weight_scale']
    weight = state_dict.pop(f'{prefix}weight')
    state_dict.pop(f'{prefix}comfy_quant')
    scale = state_dict.pop(f'{prefix}weight_scale', None)
    levels = state_dict.pop(f'{prefix}weight_level', None)
    if scale is None:
        raise ValueError(f'PRISM trellis layer {layer} is missing weight_scale')
    n, k = module._orig_shape
    plain = weight.dtype == torch.int8
    if plain:  # saved after a LoRA patch
        if weight.shape != (n, k):
            raise ValueError(f'PRISM trellis layer {layer}: int8 weight shape does not match ({n}, {k})')
        qdata = weight.to(device=device).contiguous()
        levels = torch.zeros(0, dtype=torch.uint8, device=device)
    else:
        loaded.append(f'{prefix}weight_level')
        if (levels is None or weight.dtype != torch.uint8 or weight.dim() != 2 or n % 16 or k % 16
                or weight.shape[0] != n or (weight.shape[1] * 8) % k or not 1 <= weight.shape[1] * 8 // k <= 5
                or levels.shape != (n // 16, k // 16)):
            raise ValueError(f'PRISM trellis layer {layer}: storage shapes do not match ({n}, {k})')
        if levels.dtype != torch.uint8 or torch.any(levels >= td.trellis_file.LEVELS):
            raise ValueError(f'PRISM trellis layer {layer}: weight_level must be uint8 with values in 0..31')
        qdata = td.to_words(td.trellis_file.from_file(weight.to(device=device), levels))
        levels = levels.to(device=device, dtype=torch.uint8).contiguous()
    module.quant_format = FORMAT
    module.layout_type = LAYOUT
    module._full_precision_mm_config = False
    params = PrismTrellisLayout.Params(
        scale=scale.to(device=device, dtype=torch.float32), orig_dtype=compute_dtype, orig_shape=module._orig_shape,
        levels=levels, convrot=bool(layer_conf.get('convrot', True)),
        convrot_groupsize=int(layer_conf.get('convrot_groupsize', 256)), plain=plain)
    module.weight = torch.nn.Parameter(QuantizedTensor(qdata, LAYOUT, params), requires_grad=False)
    if not getattr(module, 'prism_lora_hook', False):
        module.register_forward_hook(prism_trellis_lora.forward_hook)
        module.prism_lora_hook = True
    super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    for key in loaded:
        if key in missing_keys:
            missing_keys.remove(key)


prism_trellis_hook.LOADERS[FORMAT] = _load
logging.info('PRISM-Q: registered %s (just-in-time decode via NVRTC%s)', FORMAT,
             '' if hasattr(torch.cuda, '_compile_kernel') else ' unavailable in this PyTorch: decoding in torch, slow')
