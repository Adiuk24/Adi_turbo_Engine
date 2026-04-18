# AdiTurbo QAT Fine-Tune — GGUF-Native Training Roadmap

**Status:** Planned
**Priority:** High — eliminates cloud dependency for QAT
**Goal:** Train quantized GGUF models directly on consumer hardware, no Python/PyTorch

## The Command

```bash
./build/bin/llama-finetune \
  --teacher gemma4-26B-BF16.gguf \     # teacher model (frozen, inference only)
  --model gemma4-26B-IQ3_M.gguf \      # student model (the one being trained)
  --lora-train --lora-rank 64 \         # train LoRA adapters on GGUF directly
  --distill --temperature 2.0 \         # KL divergence from teacher logits
  --data eyla-training-data.txt \       # training data
  --steps 5000                          # Google uses 5000 steps for QAT
```

## Why This Matters

Currently QAT requires:
- Modal/cloud GPU ($20+)
- PyTorch + HuggingFace + PEFT stack
- Download 50GB BF16 model twice (HF format)
- Convert back to GGUF after training

With GGUF-native QAT:
- Runs locally on M4 24GB or RTX 3060 12GB
- Zero cloud cost
- No Python dependency
- Uses existing GGUF files directly (no format conversion)
- Works with any model AdiTurbo supports

## What Needs to Be Built

### 1. LoRA Training in llama-finetune
Current `llama-finetune` uses `llama_opt_param_filter_all` (trains ALL parameters).
Need: `llama_opt_param_filter_lora` that only trains LoRA adapter weights.

```c
// New param filter: only train LoRA adapter tensors
bool llama_opt_param_filter_lora(const struct ggml_tensor * tensor, void * userdata) {
    const char * name = ggml_get_name(tensor);
    return strstr(name, "lora_A") || strstr(name, "lora_B");
}
```

Memory with LoRA (rank 64):
- IQ3_M model: 12 GB (mmap, read-only)
- LoRA adapters: ~200 MB (FP32, trainable)
- Gradients: ~200 MB (only for LoRA params)
- Optimizer: ~400 MB (8-bit Adam for LoRA)
- KV cache + activations: ~4 GB
- Total: ~17 GB → fits M4 24GB

### 2. Teacher Model Loading
Load a second GGUF model as frozen teacher for distillation.

```c
// In finetune.cpp
llama_model * teacher = llama_model_load(teacher_path, teacher_params);
// Teacher: inference only, no gradients, can be mmap'd
```

Memory for teacher:
- Q8_0 teacher: ~27 GB (instead of 50GB BF16 — saves RAM, close enough quality)
- Or: same IQ3_M as student (self-distillation without teacher)

### 3. KL Divergence Loss
Replace standard cross-entropy with teacher-student distillation loss.

```c
// Pseudocode
logits_teacher = forward(teacher, input_ids);  // no grad
logits_student = forward(student, input_ids);  // with grad

// Softmax with temperature
p_teacher = softmax(logits_teacher / T);
p_student = log_softmax(logits_student / T);

// KL divergence
loss = kl_div(p_student, p_teacher) * T * T;
```

### 4. LoRA Adapter Save/Load
Save trained LoRA as separate file, apply at inference.

```bash
# Train → saves lora-adapter.gguf
./build/bin/llama-finetune --lora-train ...

# Inference with adapter
./build/bin/llama-server -m model.gguf --lora lora-adapter.gguf
```

## Memory Budget Scenarios

### M4 24GB (student only, no teacher)
```
IQ3_M student: 12 GB (mmap)
LoRA r=64:      0.2 GB
Gradients:      0.2 GB
Optimizer:      0.4 GB
Activations:    4 GB (grad checkpointing)
Total:         ~17 GB ← fits
```

### M4 24GB (with Q8_0 teacher) — WON'T FIT
```
Q8_0 teacher:  27 GB
IQ3_M student: 12 GB
Total:         39 GB ← OOM
```

### RTX 3060 Desktop (12GB VRAM + 32GB RAM)
```
Student on GPU: 12 GB (IQ3_M)
Teacher on CPU: 27 GB (Q8_0, CPU inference)
LoRA + grads:   GPU remaining
Total:          fits with CPU offload
```

### Self-Distillation (no teacher needed)
```
One model loaded, two forward passes:
  Pass 1: clean forward (no fake quant) → teacher logits
  Pass 2: with fake quant → student logits
  Loss = KL(student, teacher)
Memory: same as single model (~17 GB on M4)
```

## Implementation Order

1. **LoRA training support** — modify `param_filter` + add LoRA tensor creation
2. **Self-distillation** — fake quant hooks + KL loss (no teacher model needed)
3. **Teacher model loading** — load second GGUF for full distillation
4. **LoRA adapter save/merge** — export adapter, merge into base model

## References

- Google Gemma 3 Technical Report Section 2.3 (QAT method)
- Jacob et al., 2018 — original QAT paper
- Current AdiTurbo finetune: `examples/training/finetune.cpp`
- Current LoRA inference: `src/llama-adapter.cpp`
