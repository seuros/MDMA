The cooperative-matrix Vulkan kernel in
src/backend/vulkan/kernels/gemm_i8.glsl and the shared packed-weight loader
in src/backend/vulkan/kernels/int8_weight_tile.glsl are adapted from llama.cpp commit
70c4e1582e37e4fd94104eb09301711a0f2675bc (mul_mmq_cm1.comp and
mul_mmq_cm1_funcs.glsl).

src/backend/vulkan/kernels/attn_tile.comp adapts the generic flash_attn.comp
tiling and online-softmax path from llama.cpp commit
bed0a856606ee4a24a164066f73d2379447033f5.

The Q1_0 layout/dequantization in src/backend/cpu/dequant.rs and static YaRN
formula in src/model/arch/qwen/yarn.rs follow the ggml implementation in
PrismML-Eng/llama.cpp commit c96baf4abe24d11a50f326e40b6223b687449d63.
The same MIT notice below applies to these adaptations.

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
