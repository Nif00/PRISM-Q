"""LoRAs on trellis layers as a low-rank branch.

ComfyUI applies a LoRA by merging it into the weight (dequantize, add the delta, requantize). For a trellis layer that
replaces the compact payload with a full int8 weight. Instead, plain LoRA patches on a trellis layer are kept out of
the weight and added next to it:

    y = int8_linear(x, decode(W)) + sum_i strength_i * alpha_i / rank_i * (x @ down_i^T) @ up_i^T

restricted to the patched rows when ComfyUI addresses part of a fused layer (Qwen-Image 2.1's img_mlp.gate_up).
The base weight keeps its size and the delta is not rounded to int8. Patches the branch can't express (DoRA, LoHa,
LoKr, OFT, LoCon mid weights, full diffs, a strength_model other than 1) go through ComfyUI's merge as before, which
leaves the layer as plain int8. Under dynamic VRAM the resolver uses transient merged weights instead of writing
them back into the smaller packed resident allocation.

ComfyUI applies patches in three places, and all three are covered: merged at load (ModelPatcher.
patch_weight_to_device), at cast time for offloaded layers (LowVramPatch in module.weight_function) and at cast time
under dynamic VRAM (module.weight_lowvram_function)."""
import torch
import comfy.model_patcher as mp
import comfy.utils
from comfy.weight_adapter.lora import LoRAAdapter
from comfy_kitchen.tensor import QuantizedTensor

LAYOUT = None  # set by prism_trellis


def _is_trellis(weight):
    return isinstance(weight, QuantizedTensor) and weight._layout_cls == LAYOUT and not weight._params.plain


def _terms(patches):
    """[(up, down, scale, rows)] for a key's patch list, or None if any patch needs ComfyUI's merge."""
    terms = []
    for strength, v, strength_model, offset, function in patches:
        if not isinstance(v, LoRAAdapter) or function is not None or strength_model != 1.0:
            return None
        up, down, alpha, mid, dora_scale, reshape = v.weights[:6]
        if mid is not None or dora_scale is not None or reshape is not None or up.dim() != 2 or down.dim() != 2:
            return None
        if offset is not None and offset[0] != 0:
            return None
        scale = float(strength) * (float(alpha) / down.shape[0] if alpha is not None else 1.0)
        if scale != 0.0:
            terms.append((up, down, scale, None if offset is None else (int(offset[1]), int(offset[2]))))
    return terms


def _set(module, terms):
    module.prism_lora = terms
    module.prism_lora_cache = {}


def _attach(patcher, key, module=None):
    """Keep key's patches out of the weight if the branch can express them; returns True when it did."""
    if not key.endswith('.weight') or key not in patcher.patches:
        return False
    if module is None:
        module = comfy.utils.get_attr(patcher.model, key[:-len('.weight')])
    if not _is_trellis(getattr(module, 'weight', None)):
        return False
    terms = _terms(patcher.patches[key])
    if terms is None:
        return False
    _set(module, terms)
    return True


def add_lora(module, x, out, out_scale=None):
    """out += the LoRA branch of module for input x (in place); out_scale multiplies the branch (a fused residual)."""
    terms = getattr(module, 'prism_lora', None)
    if not terms:
        return out
    if not out.is_contiguous():
        return out.add_(add_lora(module, x, torch.zeros(out.shape, device=out.device, dtype=out.dtype), out_scale))
    cache = module.prism_lora_cache
    key = (x.device, x.dtype)
    if key not in cache:  # all down projections in one matrix, so x is read once
        downs = torch.cat([down.to(x.device, x.dtype) for up, down, s, rows in terms])
        ups, start = [], 0
        for up, down, s, rows in terms:
            ups.append((up.to(x.device, x.dtype).t(), start, start + down.shape[0], s, rows))
            start += down.shape[0]
        cache[key] = downs, ups
    downs, ups = cache[key]
    h = torch.nn.functional.linear(x, downs).reshape(-1, downs.shape[0])
    o = out.view(-1, out.shape[-1])
    for up_t, a, b, s, rows in ups:
        dst = o if rows is None else o.narrow(-1, rows[0], rows[1])
        if out_scale is None:
            dst.addmm_(h[:, a:b], up_t, alpha=s)  # in place: one read and one write of out
        else:
            scale = out_scale.expand(out.shape).reshape(o.shape)
            dst.addcmul_((h[:, a:b] @ up_t), scale if rows is None else scale.narrow(-1, rows[0], rows[1]), value=s)
    return out


def forward_hook(module, args, output):
    if getattr(module, 'prism_lora', None) and isinstance(output, torch.Tensor):
        add_lora(module, args[0], output)
    return output


def _after_load(patcher):
    """Attach the branch on every trellis layer whose patches allow it and drop ComfyUI's cast-time patch for it;
    clear stale branches."""
    for name, module in patcher.model.named_modules():
        if not _is_trellis(getattr(module, 'weight', None)):
            continue
        key = f'{name}.weight'
        lowvram = getattr(module, 'weight_lowvram_function', None)
        dynamic = isinstance(lowvram, mp.LowVramPatch) and lowvram.key == key
        if _attach(patcher, key, module):
            fns = getattr(module, 'weight_function', None)
            if fns:
                module.weight_function = [f for f in fns if not (isinstance(f, mp.LowVramPatch) and f.key == key)]
            if dynamic:
                module.weight_lowvram_function = None
            continue
        if getattr(module, 'prism_lora', None):
            _set(module, [])


def _clear(patcher):
    for module in patcher.model.modules():
        if getattr(module, 'prism_lora', None):
            _set(module, [])


def _wrap_patch_weight_to_device(original):
    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, force_cast=False):
        if not return_weight and _attach(self, key):
            return None
        return original(self, key, device_to=device_to, inplace_update=inplace_update, return_weight=return_weight,
                        force_cast=force_cast)
    return patch_weight_to_device


def _wrap_load(original):
    def load(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        _after_load(self)
        return result
    return load


def _wrap_unpatch_model(original):
    def unpatch_model(self, device_to=None, unpatch_weights=True):
        result = original(self, device_to=device_to, unpatch_weights=unpatch_weights)
        if unpatch_weights:
            _clear(self)
        return result
    return unpatch_model


def install():
    targets = [(mp.ModelPatcher, 'patch_weight_to_device', _wrap_patch_weight_to_device),
               (mp.ModelPatcher, 'load', _wrap_load), (mp.ModelPatcher, 'unpatch_model', _wrap_unpatch_model)]
    dynamic = getattr(mp, 'ModelPatcherDynamic', None)
    if dynamic is not None:
        targets += [(dynamic, 'load', _wrap_load), (dynamic, 'unpatch_model', _wrap_unpatch_model)]
    for cls, name, wrap in targets:
        method = cls.__dict__.get(name)
        if method is not None and not getattr(method, 'prism_lora', False):
            wrapped = wrap(method)
            wrapped.prism_lora = True
            setattr(cls, name, wrapped)
