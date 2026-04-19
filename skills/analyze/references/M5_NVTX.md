# Mode 5 — NVTX / code mapping

Reference for `/nsys-ai` Mode 5. **Read `PRINCIPLES.md` first** — §4 guards, §5 evidence,
§7 fail template, §10 checklist.

---

## 1. Precondition gate

Run a single-row probe first:

```bash
nsys-ai skill run nvtx_layer_breakdown <profile> --format json --max-rows 1
```

If empty, render the §7 template:

```
Mode 5 requires a user choice.

Why: no NVTX annotations found in this profile.
Fix: add `with torch.cuda.nvtx.range("layer_name"):` around key regions and re-profile.
Alternative: Path B uses kernel-name heuristics (no per-layer attribution).

Reply with: "B", "stop", or re-profile with annotations.
```

If user replies "B" → Path B fallback (§3). Otherwise proceed normally.

---

## 2. Stages

| # | Question | Condition/Default |
|---|----------|-------------------|
| 1 | Profile path | Only if not supplied |
| 2 | Sub-focus: `layer` / `iteration` / `kernel-attribution` | Optional; default `layer` |

Skip Stage 2 if keyword already maps to a sub-focus (e.g. "which layer" → `layer`,
"slow iterations" → `iteration`, "what kernel runs in attention" → `kernel-attribution`).

---

## 3. Skills

| Sub-focus | Skills (in order) |
|-----------|-------------------|
| `layer` (default) | `nvtx_layer_breakdown --max-rows 20` → `nvtx_kernel_map` |
| `iteration` | `iteration_timing` → `iteration_detail -p iteration=<slow_N>` → `speedup_estimator -p iteration_ms=<median_ms>` |
| `kernel-attribution` | `nvtx_kernel_map` → `kernel_instances -p name=<name>` |

Device propagation: `iteration_timing` accepts `-p device=N` (param name `device`, not
`device_id`). `nvtx_layer_breakdown`, `nvtx_kernel_map`, `speedup_estimator` do not accept
device. `kernel_instances` accepts `-p device=N`. See PRINCIPLES.md §6.

**Path B** (no NVTX — fallback):

```bash
nsys-ai skill run top_kernels <profile> -p limit=30 --format json --max-rows 30
nsys-ai skill run kernel_launch_pattern <profile> --format json
```

Apply kernel→source heuristic to attribute kernels to model components:

| Kernel name pattern | Likely source |
|--------------------|--------------|
| `ampere_sgemm*` / `volta_sgemm*` / `sm80_xmma_gemm*` | `nn.Linear` (FP32/BF16 dense matmul) |
| `flash_fwd*` / `flash_bwd*` / `flash_attn_*` | `scaled_dot_product_attention` / FlashAttention |
| `cunn_*` / `cudnn_*` | cuDNN op (conv, BN, RNN) |
| `ncclDevKernel*` | `dist.all_reduce` / `dist.reduce_scatter` / NCCL collective |
| `void at::native::*` | PyTorch native CUDA kernel |
| `elementwise_kernel*` | Elementwise op (activation, dropout) |
| `reduce_kernel*` | Reduction (layer norm, softmax) |

`speedup_estimator` is less reliable in Path B without NVTX markers; it may still be used if
heuristic iteration detection succeeds or if `iteration_ms` is provided manually.

---

## 4. Signals

**From `nvtx_layer_breakdown`** (per-region fields: `nvtx_region`, `nvtx_path`,
`total_gpu_ms`, `compute_ms`, `nccl_ms`, `nccl_pct`, `tc_achieved_pct`,
`kernel_count`, `avg_kernel_ms`; embedded `top_kernels[].{kernel_name, total_ms}`):

| Field / condition | Diagnosis | Action |
|-------------------|-----------|--------|
| One region `total_gpu_ms` >> all others | Slow layer — compute or NCCL | drill into `nvtx_kernel_map` for that region |
| Region `nccl_pct > 50` | NCCL inside this layer dominates | Mode 2 for NCCL analysis |
| Region `tc_achieved_pct < 20` | Poor tensor core usage in this layer | check dtype; is it a custom op? |
| `avg_kernel_ms` high but `kernel_count` low | Few, slow kernels | consider operator fusion |

**From `iteration_timing`** (per-iteration fields: `iteration`, `duration_ms`,
`kernel_count`, `nccl_count`, `compute_ms`):

| Field / condition | Diagnosis | Action |
|-------------------|-----------|--------|
| `duration_ms` p99/p50 > 1.5 | Iteration spikes → suggest Mode 9 | check GC / DataLoader |
| `nccl_count` spikes in slow iteration | NCCL retry or timeout | Mode 2 with per-rank-variance |
| `kernel_count` spikes in slow iteration | Recomputation / gradient checkpointing stall | check activation checkpoint strategy |

**From `speedup_estimator`** (per-optimization fields: `optimization`, `speedup`,
`new_iteration_ms`, `current_ms`, `saved_ms`):

Use `speedup` value directly in delivery: "fixing idle gaps → estimated `N`× speedup per step."

---

## 5. Cross-mode exits

After delivery, suggest a second mode only if a distinct critical finding exists.

- Hot layer's top kernel is custom/Triton AND > 60% of that region's time → suggest **Mode 3**
  (`specific-kernel` sub-focus with that kernel name)
- `iteration_timing` shows spikes (p99/p50 > 1.5) → suggest **Mode 9** (variance analysis)

---

## 6. Delivery

Follow `PRINCIPLES.md §5` for evidence build + timeline URL. Then 3-part summary:

1. **Root cause** — layer name + quantified % + mechanism:
   > "The `TransformerLayer.12.attn_fwd` region accounts for 34% of GPU time. Top kernel is
   > `flash_fwd_hdim128_bf16_sm80` — attention computation is the bottleneck, not NCCL."

2. **Specific fix** — matching the attribution:
   - Slow attention layer: upgrade to `torch.nn.functional.scaled_dot_product_attention` (2–4×)
   - Slow FFN: fuse with `torch.compile(mode='reduce-overhead')`; try INT8 quantization
   - Slow backward pass only: gradient checkpointing trade-off; recompute fewer activations
   - Path B (heuristic): "attribution is approximate — re-profile with NVTX for accuracy"

3. **Expected gain** — from `speedup_estimator` if NVTX present; omit in Path B.

See `ROOT_CAUSE.md §1` for the cross-mode cause→fix matrix.
