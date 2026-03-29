# Eyla AIOS — Model Tier Strategy & Build Order

## Goal
Beat Claude Code reasoning with local models. Ship 3 tiers for 8/16-24/32 GB RAM laptops.
All models quantized with AdiTurbo TQ3_0 engine.

## Target Hardware

| Tier | RAM | OS | Target Users |
|------|-----|----|-------------|
| Lite | 8 GB | Windows/Mac | Bangladesh mass market, $300 laptops |
| Pro | 16-24 GB | Windows/Mac | Developers, students |
| Ultra | 32 GB | Windows/Mac | Power users, dev machines |

## Model Selection

### Tier 1: Lite (8 GB RAM)

RAM budget: ~4 GB model + 2 GB KV + 2 GB OS

| Rank | Model | Total | Active | TQ3 RAM | TQ3 Disk | HumanEval+ | Notes |
|------|-------|-------|--------|---------|----------|------------|-------|
| 1 | Qwen3-30B-A3B | 30B MoE | 3B | 1.2 GB | 11 GB | ~85% | Best reasoning per active param |
| 2 | Qwen2.5-Coder-7B | 7B dense | 7B | 3.0 GB | 3.0 GB | 79% | Best code-specific at this size |
| 3 | DeepSeek-R1-Distill-Qwen-7B | 7B dense | 7B | 3.0 GB | 3.0 GB | ~80% | Reasoning-focused |
| 4 | Qwen3-8B | 8B dense | 8B | 3.1 GB | 3.1 GB | 78% | General purpose + Bangla |

**Ship: Qwen3-30B-A3B** — MoE, only 1.2 GB active, leaves 6.8 GB for KV + OS.

### Tier 2: Pro (16-24 GB RAM)

RAM budget: ~12-18 GB model

| Rank | Model | Total | Active | TQ3 RAM | TQ3 Disk | HumanEval+ | Notes |
|------|-------|-------|--------|---------|----------|------------|-------|
| 1 | Qwen2.5-Coder-32B-Instruct | 32B dense | 32B | 12 GB | 12 GB | 90% | Best pure code model |
| 2 | Qwen3-235B-A22B | 235B MoE | 22B | 8.5 GB | 90 GB | 90% | Claude-competitive, needs expert offload |
| 3 | DeepSeek-Coder-V2-Lite | 16B MoE | 2.4B | 1 GB | 6 GB | 89% | Ultralight, huge context room |
| 4 | Qwen2.5-Coder-14B | 14B dense | 14B | 5.4 GB | 5.4 GB | 84% | Mid-range code model |

**Ship: Qwen2.5-Coder-32B** (24 GB Macs) / **Qwen3-235B-A22B** (16 GB with expert offload)

### Tier 3: Ultra (32 GB RAM)

RAM budget: ~24 GB model

| Rank | Model | Total | Active | TQ3 RAM | TQ3 Disk | HumanEval+ | Notes |
|------|-------|-------|--------|---------|----------|------------|-------|
| 1 | DeepSeek-V3 | 671B MoE | 37B | 14 GB | 250 GB | 91% | Beats Claude on most benchmarks |
| 2 | DeepSeek-R1 | 671B MoE | 37B | 14 GB | 250 GB | ~90% | Best open reasoning model |
| 3 | Qwen3-235B-A22B | 235B MoE | 22B | 8.5 GB | 90 GB | 90% | Comfortable fit, 128K context |

**Ship: DeepSeek-V3** — 671B total, 37B active = 14 GB. The Claude-beater.

## Benchmark Comparison vs Claude

| Model | HumanEval+ | SWE-bench | LiveCodeBench | Aider |
|-------|-----------|-----------|---------------|-------|
| Claude 3.5 Sonnet | ~92% | ~49% | ~35% | 72% |
| DeepSeek-V3 (671B) | 91% | 42% | 33% | 65% |
| Qwen3-235B-A22B | 90% | ~38% | 32% | ~60% |
| Qwen2.5-Coder-32B | 90% | 26% | 28% | 55% |
| DeepSeek-Coder-V2 (236B) | 89% | 30% | 27% | 52% |
| Qwen3-30B-A3B | 85% | ~20% | 25% | ~45% |
| Qwen2.5-Coder-7B | 79% | 12% | 18% | 32% |

## Build Order (Pre-RunPod)

Everything below must be done BEFORE converting models on RunPod.

### Step 1: Metal GPU Shaders for TQ3/TQ4 (2-3 days)
- Files: `ggml/src/ggml-metal/ggml-metal.metal`
- What: Write MSL compute shaders for TQ3_0/TQ4_0 dequantize + mul_mat + get_rows
- Why: Without Metal, everything runs CPU-only. 7B = 17 t/s. With Metal = 40-60+ t/s.
- Current blocker: `ggml-metal-device.m` lines 1173-1176 explicitly reject TQ3/TQ4
- Reference: Look at how Q4_0 Metal shaders work, adapt for 3-bit/4-bit TQ packing

### Step 2: MoE Expert Offloading via mmap (3-4 days)
- Files: `src/llama-model.cpp`, `src/llama-context.cpp`
- What: mmap the full GGUF, only fault-in active expert weights per token
- Why: 235B MoE = 90 GB disk but only 8.5 GB active. Without this, llama.cpp tries to load everything.
- Current: llama.cpp has `--mmap` but loads all layers. Need per-expert granularity.
- Critical for: Tier 2 (235B) and Tier 3 (671B)

### Step 3: Small-Block TQ3 for KV Cache (1-2 days)
- Files: `ggml-common.h`, `ggml-quants.c`, `ggml.c`, `arch/arm/quants.c`
- What: TQ3 variant with block_size=128 (fits head_dim=128)
- Why: Current TQ3 block=256 > head_dim=128. KV cache can't use it.
- Without this: KV stays F16, 128K context = 16+ GB. With this: 128K = 2-4 GB.

### Step 4: Default Repeat Penalty for TQ3 (30 min)
- File: `common/arg.cpp` or model defaults
- What: Auto-set `--repeat-penalty 1.1` when TQ3 ftype detected
- Why: 3-bit quantization causes repetition loops without it

### Step 5: Windows Build Verification (1 day)
- What: Cross-compile and test on Windows (MSVC + ARM64 / x86_64)
- Why: Half the target users are on Windows
- Needs: Test NEON path on Windows ARM (Snapdragon), SSE/AVX path on x86

### Step 6: Eyla Integration API (2 days)
- What: C API for Eyla to call AdiTurbo (load model, generate, stream tokens)
- Why: Eyla currently uses Ollama. Replace with AdiTurbo.
- Interface: `aditurbo_load(tier, ram_mb)` → auto-selects best model for available RAM

## RunPod Conversion Plan (After Steps 1-4)

| Order | Model | F16 Size | GPU Needed | Time Est |
|-------|-------|----------|-----------|----------|
| 1 | Qwen3-30B-A3B | ~60 GB | A100 80GB | ~1 hr |
| 2 | Qwen2.5-Coder-32B | ~64 GB | A100 80GB | ~1 hr |
| 3 | Qwen3-235B-A22B | ~470 GB | 2xH100 or CPU convert | ~4 hrs |
| 4 | DeepSeek-V3 | ~1.3 TB | 8xH100 or CPU convert | ~8 hrs |

Note: For models larger than 80 GB F16, we can convert on CPU (slower but no VRAM limit).
RunPod disk: Need 1 TB+ for DeepSeek-V3 conversion.

## Disk Budget (This Mac, 61 GB free)

| What | Size | Cumulative |
|------|------|-----------|
| Qwen3-30B-A3B TQ3 | 11 GB | 11 GB |
| Qwen2.5-Coder-32B TQ3 | 12 GB | 23 GB |
| Existing Qwen2.5-7B models | 20 GB | 43 GB |
| Remaining free | 18 GB | - |

Can fit Tier 1 + Tier 2 locally. Tier 3 (DeepSeek-V3, 250 GB) needs external drive.
User plans to move some files to external HDD to free space.
