"""Just-in-time decode of PRISM-Q trellis layers (format in prism_trellis_file.py) to ConvRot's int8 weight.

The CUDA kernel is compiled at first use with PyTorch's bundled NVRTC (torch.cuda._compile_kernel): no CUDA
toolkit, compiler or Triton is needed. One thread decodes one 16-weight row of one 16x16 tile: weight t of a tile is
state t = the 16 bits from bit Kb*t of the tile's cyclic bitstream, value = (r(state) * mult[level] + 32768) >> 16
with the QTIP 1MAD codebook r. Threads run in output row-major order, so each writes one 16-byte segment next to its
neighbour's. Integer-exact: bit-identical to prism_trellis_file.decode_layer, which is the fallback on CPU or when
NVRTC is unavailable."""
import ctypes
import logging
import threading

import torch

from . import prism_trellis_file as trellis_file

_SRC = r'''
extern "C" __global__ void prism_trellis_decode(const unsigned int* __restrict__ words,
                                                const unsigned char* __restrict__ levels,
                                                const int* __restrict__ mults,
                                                unsigned int* __restrict__ out,
                                                int n, int tk) {
    // one thread per (output row, tile column); block = 64 tile columns x 4 rows, so no index division
    int tj = blockIdx.x * 64 + threadIdx.x;
    int row = blockIdx.y * 4 + threadIdx.y;
    if (tj >= tk) return;
    long long idx = (long long)row * tk + tj;
    long long tile = (long long)(row >> 4) * tk + tj;
    int r = (int)(row & 15);
    const unsigned int* w = words + tile * (8 * KB);
    // bits r*16*KB .. r*16*KB + 15*KB + 15 of the cyclic bitstream sit in 4 consecutive words (KB <= 5)
    int b0 = r * 16 * KB;
    int w0 = b0 >> 5;
    // window = the 128 bits from b0 (MSB first) as two 64-bit halves, so bit offsets below are compile-time
    int base = b0 & 31;
    unsigned long long a = ((unsigned long long)w[w0 % (8 * KB)] << 32) | w[(w0 + 1) % (8 * KB)];
    unsigned long long b = ((unsigned long long)w[(w0 + 2) % (8 * KB)] << 32) | w[(w0 + 3) % (8 * KB)];
    if (base) {
        a = (a << base) | (b >> (64 - base));
        b <<= base;
    }
    // value = ((bytesum - 510) * mult + 32768) >> 16 = byte 2 of bytesum * mult + bias (two's complement), so each
    // weight is: extract the 16-bit state, one IMAD (hash), one DP4A (byte sum), one IMAD (scale); bytes are packed
    // four at a time with PRMT. The decode is ALU-bound, so instruction count is its speed.
    int mult = mults[levels[tile]];
    int bias = 32768 - 510 * mult;
    unsigned int x[16];
#pragma unroll
    for (int c = 0; c < 16; c++) {
        const int off = c * KB;  // the window has 15 * KB + 16 <= 91 valid bits
        unsigned int s;
        if (off + 16 <= 64) s = (unsigned int)(a >> (48 - off)) & 0xFFFFu;
        else if (off >= 64) s = (unsigned int)(b >> (112 - off)) & 0xFFFFu;
        else s = (unsigned int)((a << (off - 48)) | (b >> (112 - off))) & 0xFFFFu;
        unsigned int m = s * 34038481u + 76625530u;
        int sum = (int)__dp4a(m, 0x01010101u, 0u);
        x[c] = (unsigned int)(sum * mult + bias);
    }
    unsigned int packed[4];
#pragma unroll
    for (int q = 0; q < 4; q++)  // bytes 2 of x[4q..4q+3] -> one word, little-endian
        packed[q] = __byte_perm(__byte_perm(x[4 * q], x[4 * q + 1], 0x0062), __byte_perm(x[4 * q + 2], x[4 * q + 3], 0x0062),
                                0x5410);
    uint4* dst = (uint4*)(out) + idx;  // 16 int8 = one uint4, row-major
    *dst = make_uint4(packed[0], packed[1], packed[2], packed[3]);
}
'''
_LINKAGE = 'extern "C" typedef int prism_trellis_nvrtc_linkage_t;\n'
_KERNELS = {}
_LAUNCHERS = {}
_MULTS = {}
_FAILED = set()


def to_words(payload):
    """(n/16, k/16, 32Kb) uint8 payload -> (n/16, k/16, 8Kb) int32 big-endian words: the resident form."""
    return payload.view(*payload.shape[:-1], -1, 4).flip(-1).contiguous().view(torch.int32).view(*payload.shape[:-1], -1)


def from_words(words):
    """Inverse of to_words: the (n/16, k/16, 32Kb) uint8 payload."""
    return words.contiguous().view(torch.uint8).view(*words.shape[:-1], -1, 4).flip(-1).reshape(*words.shape[:-1], -1)


def _kernel(device, kb):
    key = (device.index, kb)
    if key not in _KERNELS:
        cc = torch.cuda.get_device_capability(device)
        with torch.cuda.device(device):
            _KERNELS[key] = torch.cuda._compile_kernel(_LINKAGE + f'#define KB {kb}\n' + _SRC, 'prism_trellis_decode',
                                                       compute_capability=f'{cc[0]}{cc[1]}', nvcc_options=['-std=c++17'])
    return _KERNELS[key]


class _Launcher:
    """cuLaunchKernel with argument buffers built once. torch's NVRTC launcher rebuilds its ctypes arguments and looks
    up a stream object on every call (about 80 us), longer than the decode itself runs on an A100, so the CPU fell
    behind the GPU. The driver copies the arguments at launch, so the buffers can be refilled for the next call."""

    def __init__(self, kernel):
        self.func = kernel.func
        self.lib = torch.cuda._utils._get_gpu_runtime_library()
        self.ptrs = [ctypes.c_void_p() for _ in range(4)]
        self.ints = [ctypes.c_int() for _ in range(2)]
        self.argv = (ctypes.c_void_p * 6)(*[ctypes.addressof(a) for a in self.ptrs + self.ints])
        self.lock = threading.Lock()

    def __call__(self, grid, tensors, ints, device_index):
        with self.lock:
            for slot, t in zip(self.ptrs, tensors):
                slot.value = t.data_ptr()
            for slot, v in zip(self.ints, ints):
                slot.value = v
            stream = ctypes.c_void_p(torch._C._cuda_getCurrentRawStream(device_index))
            err = self.lib.cuLaunchKernel(self.func, grid[0], grid[1], 1, 64, 4, 1, 0, stream, self.argv, None)
        if err:
            raise RuntimeError(f'cuLaunchKernel failed with CUDA error {err}')


def _launcher(device, kb):
    key = (device.index, kb)
    if key not in _LAUNCHERS:
        _LAUNCHERS[key] = _Launcher(_kernel(device, kb))
    return _LAUNCHERS[key]


def available(device):
    return device.type == 'cuda' and hasattr(torch.cuda, '_compile_kernel') and device.index not in _FAILED


def decode_words(words, levels, out=None):
    """to_words(payload) (n/16, k/16, 8Kb) and levels (n/16, k/16) -> int8 weight (n, k)."""
    if levels.shape != words.shape[:2] or levels.device != words.device:
        raise ValueError('PRISM trellis weight_level must match the payload tile shape and device')
    if levels.dtype != torch.uint8 or torch.any(levels >= trellis_file.LEVELS):
        raise ValueError('PRISM trellis weight_level must be uint8 with values in 0..31')
    return _decode_words(words, levels.contiguous(), out)


def _decode_words(words, levels, out=None):
    """Decode contiguous uint8 levels already validated by the loader or decode_words."""
    tn, tk, nwords = words.shape
    n, k = tn * 16, tk * 16
    dev = words.device
    if available(dev):
        try:
            launch = _launcher(dev, nwords // 8)
            mults = _MULTS.get(dev.index)
            if mults is None:
                mults = _MULTS[dev.index] = trellis_file.level_mults(dev).to(torch.int32)
            out = torch.empty(n, k, dtype=torch.int8, device=dev) if out is None else out
            args = (((tk + 63) // 64, n // 4), (words, levels, mults, out), (n, tk), dev.index)
            if torch.cuda.current_device() == dev.index:
                launch(*args)
            else:
                with torch.cuda.device(dev):
                    launch(*args)
            return out
        except Exception as e:  # NVRTC missing or failing: decode in torch
            _FAILED.add(dev.index)
            logging.warning('PRISM-Q trellis: NVRTC decode unavailable (%s); decoding in torch (slower)', e)
    return trellis_file.decode_layer(from_words(words), levels)
