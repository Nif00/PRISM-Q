# PRISM-Q for ComfyUI

Loads PRISM-Q trellis checkpoints of **Qwen-Image 2.1** with ComfyUI's stock
**Load Diffusion Model** (`UNETLoader`) node. The pack adds no nodes and the
workflow doesn't change: pick the file in Load Diffusion Model and generate.

![Fidelity vs file size](assets/fidelity.png)

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/Nif00/PRISM-Q.git
```

Restart ComfyUI. There is nothing to pip install and nothing to compile: the
decode kernel is built at first use with the NVRTC that ships inside PyTorch.

Requirements:

* A ComfyUI that loads the official Qwen-Image 2.1 INT8 ConvRot checkpoint
  (`int8_tensorwise` with ConvRot through `comfy_kitchen`). PRISM-Q files run
  the same int8 kernel.
* An NVIDIA GPU with INT8 tensor cores (Turing or newer; tested on Ampere:
  A100, RTX 3060 Ti).
* A recent PyTorch with `torch.cuda._compile_kernel` (tested on 2.13 with CUDA
  13.0). Without it the pack falls back to decoding in plain torch. That
  fallback is exact but about 250× slower per layer, and the log says so at
  startup.

Put the model files in `ComfyUI/models/diffusion_models/`. Text encoder, VAE,
LoRAs and samplers are the same as for the official checkpoints.

## Files

The model files are on Hugging Face: [nif0/PRISM-Q-Qwen-Image-2.1-Turbo](https://huggingface.co/nif0/PRISM-Q-Qwen-Image-2.1-Turbo). They are derived from
[Qwen-Image 2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) and follow its Qwen Research
License: non-commercial use only.

| File | Size | Fidelity | Seconds / image | Peak VRAM | Pack needed |
|---|---:|---:|---:|---:|:---:|
| `prism_q_qwen_image_2.1_turbo_compact.safetensors` | 4.23 GB | 60.5 | 2.01 | 4.6 GiB | yes |
| `prism_q_qwen_image_2.1_turbo_balanced.safetensors` | 5.15 GB | 81.4 | 1.99 | 5.5 GiB | yes |
| `prism_q_qwen_image_2.1_turbo_int8.safetensors` | 7.26 GB | 106.4 | 1.91 | 7.4 GiB | no |

Fidelity is a composite of LPIPS, PSNR, SSIM and Pearson against BF16 images,
where ComfyUI's official int8 ConvRot file scores 100 and runs 1.92 s. Timing
is 1024 × 1024, 8 steps, on an A100 40 GB, denoiser only. The model card has
the full comparison with BF16, ComfyUI's W4A8 and the GGUF quants.

![Seconds per image](assets/speed.png)

On an RTX 3060 Ti (8 GB) a 1024 × 1024 image takes 9.2 s with compact, 9.0 s
with balanced and 9.2 s with int8, against 10.9 s for ComfyUI's int8 ConvRot
file. The int8 files don't fit in 8 GB and are partly offloaded.

## How it works

PRISM-Q quantizes each linear layer of the diffusion transformer to the int8
weight that ComfyUI's ConvRot INT8 path uses. The compact and balanced files
store most of these int8 weights as a QTIP trellis bitstream (16×16 tiles,
3 to 5 bits per weight, a 1MAD codebook and one level byte per tile), so the
file and the resident weights are about half the size of int8.

The pack registers a quantization format, `prism_trellis`, with ComfyUI's
mixed-precision loader. A trellis layer keeps its bitstream in VRAM. On every
call a CUDA kernel (`prism_trellis_decode.py`) expands the bitstream into a
transient int8 weight. On an A100 that is about 90 ms of GPU time per
compact image, 4 to 5% of the run. The int8 weight then goes to comfy_kitchen's
`int8_linear` with ConvRot, the same call the official INT8 checkpoint makes.
That includes the fused variant, where ComfyUI folds the MLP's SwiGLU into
the int8 input quantizer (`comfy.ops.linear_input_act`).
The decode is integer-exact, so outputs are bit-identical to the int8
checkpoint the trellis file was packed from. The int8 layers and the BF16
keep-outs in these files are ordinary ComfyUI tensors.

LoRAs work. A plain LoRA (up/down matrices: kohya, diffusers and PEFT files)
runs as a low-rank branch next to each trellis layer, so the layer keeps its
compact size and the LoRA isn't rounded into int8. Measured on an RTX 3060 Ti
with a rank-32 LoRA on every attention and MLP linear of the compact file:
peak VRAM rose by 0.2 GiB and an image took 10 to 15% longer. Merging that
LoRA into int8 instead, as ComfyUI does for an INT8 checkpoint, needs 7.2 GiB.

Other patch types (DoRA, LoHa, LoKr, full diffs) requantize the layer to
plain int8, as on the official INT8 checkpoint. Under dynamic VRAM (ComfyUI's
default on NVIDIA), the packed source stays resident and the merged weight is
transient: it cannot be cached in the smaller packed allocation. These patches
are reapplied each forward, adding overhead for the affected layers. Plain LoRA
branches and unpatched layers keep normal resident caching.

The pack can be installed next to other node packs that hook ComfyUI's
quantized loader. Formats it doesn't know pass through to the previous hook.

## Files in this pack

* `prism_trellis.py`: the `prism_trellis` layout (forward, fused input
  activation, requantize for patches, save) and its loader.
* `prism_trellis_lora.py`: plain LoRAs as a low-rank branch.
* `prism_trellis_decode.py`: the NVRTC decode kernel and its torch fallback.
* `prism_trellis_file.py`: the file format and the reference decoder.
* `prism_trellis_hook.py`: routes `prism_trellis` layers from ComfyUI's
  loader to this pack.
* `assets/`: the charts above.

## License

The code in this repository is MIT licensed (see [LICENSE](LICENSE)). It contains
no model weights. The PRISM-Q model files are derived from Qwen-Image 2.1 Turbo
and are under the Qwen Research License (non-commercial use only); see the
[model card](https://huggingface.co/nif0/PRISM-Q-Qwen-Image-2.1-Turbo).
