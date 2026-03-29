# AdiTurbo Engine

Ultra-aggressive quantization engine for running 120B+ MoE models on consumer laptops.
Built on llama.cpp with custom TurboQuant (TQ3/TQ4) quantization types and ARM NEON SIMD kernels.

## What It Does

AdiTurbo compresses large language models to fit on $300 laptops:

| Format | Bits Per Weight | 7B Model Size | 120B MoE Active RAM |
|--------|----------------|---------------|---------------------|
| FP16 (standard) | 16.0 | 14.0 GB | 120+ GB |
| Q4_K_M (llama.cpp) | 4.5 | 4.1 GB | N/A |
| **TQ4_0 (AdiTurbo)** | **4.12** | **3.8 GB** | **~5 GB** |
| **TQ3_0 (AdiTurbo)** | **3.42** | **3.0 GB** | **~3.8 GB** |

## Performance (Apple M4 Pro, Qwen2.5-7B)

| Format | Prompt (t/s) | Generation (t/s) |
|--------|-------------|-------------------|
| TQ4_0 | 42.7 | 19.2 |
| TQ3_0 | 33-47 | 17.0 |

## Target Platforms

| Tier | RAM | Target Model | Active RAM |
|------|-----|-------------|------------|
| Lite | 8 GB | Qwen3-30B-A3B (MoE) | 1.2 GB |
| Pro | 16-24 GB | Qwen2.5-Coder-32B | 12 GB |
| Ultra | 32 GB | DeepSeek-V3 (671B MoE) | 14 GB |

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) llama-cli llama-quantize
```

## Quantize a Model

```bash
./build/bin/llama-quantize model-f16.gguf model-tq3.gguf TQ3_0
./build/bin/llama-quantize model-f16.gguf model-tq4.gguf TQ4_0
```

## Run

```bash
# CPU (ARM NEON optimized)
./build/bin/llama-cli -m model-tq3.gguf -p "Hello" -n 128 -ngl 0 -t 10

# With repeat penalty (recommended for TQ3)
./build/bin/llama-cli -m model-tq3.gguf -p "Hello" -n 128 -ngl 0 -t 10 --repeat-penalty 1.1
```

## Architecture

AdiTurbo adds two quantization types to the GGML engine:

- **TQ4_0**: 4-bit symmetric, 16 levels, QK_K=256 block size
- **TQ3_0**: 3-bit symmetric, 8 levels, QK_K=256 block size

Both include full ARM NEON SIMD kernels with `vdotq_s32` (dotprod) support.
TQ3_0 uses `vqtbl1q_u8` table lookup for parallel 3-bit extraction.

## Roadmap

- [x] TQ4_0 quantization + ARM NEON kernel
- [x] TQ3_0 quantization + ARM NEON kernel (2.7x speedup)
- [x] KV cache type support
- [x] MoE compatibility verified
- [x] Type fallback table (non-256-aligned tensors)
- [ ] Metal GPU compute shaders
- [ ] MoE expert offloading (mmap per-expert)
- [ ] TurboQuant KV cache (PolarQuant + QJL)
- [ ] Windows build verification
- [ ] Eyla AIOS integration

## Part of the Eyla Ecosystem

AdiTurbo is the inference engine for [Eyla AIOS](https://github.com/Adiuk24) — the agentic operating system.
The pipeline: AdiTurbo compresses models -> Eyla runs free on consumer hardware -> offline, $0/token, forever.

## License

MIT (inherits from llama.cpp)

## Credits

Built by Arif Ahmed Aditto (Adioris Tech Ltd, Dhaka)
Based on [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov
