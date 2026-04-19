# Root Cause Reference

Cross-mode cause→fix matrix. Loaded on demand (Layer 3) — referenced by M*.md §6 Delivery
when a mode needs delivery framing beyond its own §6 section.

---

## 1. Cross-mode cause→fix matrix

| Pattern | Primary mode | Fix | Typical speedup |
|---------|-------------|-----|----------------|
| NCCL serialized with compute (`overlap_pct < 30`) | 2 | `DDP(model, bucket_cap_mb=256)` or `FSDP(..., forward_prefetch=True)` | 1.2–1.8× |
| Single straggler GPU (one rank slow) | 2 | rebalance dataset sharding; check NIC / switch port; `NCCL_DEBUG=INFO` | 1.05–1.2× |
| GPU idle > 15% (DataLoader starvation) | 6 | `DataLoader(..., num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)` | 1.1–1.4× |
| Excessive CPU→GPU syncs (`.item()` loop) | 6 | accumulate loss as tensor; call `.item()` once outside loop | 1.05–1.15× |
| PP micro-batch bubble > 10% | 6 | `num_micro_batches = 2 × pipeline_stages` (1F1B schedule) | 1.1–1.3× |
| Compute hotspot (single kernel > 60%) | 3 | BF16 / TF32 dtype; fuse ops; replace with library kernel | varies |
| Tensor cores inactive (FP32 fallback) | 3 | `model.to(torch.bfloat16)` or `torch.set_float32_matmul_precision('high')` | 1.3–2.5× |
| H2D spread-out (every step copies data) | 4 | `pin_memory=True`, `num_workers ≥ 4`, `prefetch_factor=2` | 1.05–1.2× |
| H2D spike (rogue `.cpu()` in loop) | 4→5 | remove scalar sync from inner loop; locate via Mode 5 layer attribution | 1.05–1.15× |
| Slow model layer (NVTX) | 5 | op-level: fuse attention; quantize; `torch.compile` | varies |
| Iteration spikes (GC / DataLoader) | 9 | `gc.freeze()` before training; fix DataLoader randomness | 1.02–1.1× |
| JIT / cuModuleLoad stall at step 0 | 6 | pre-warm with 1 dummy forward pass before profiling window | n/a |

---

## 2. Root cause framing examples

One 3-part example per bottleneck class. Use these as delivery templates in Mode 1.

**NCCL serialized**:
> Root cause: "NCCL AllReduce is serialized with compute (overlap = 18%, DDP default bucket
> 25 MB). Estimated 3.4 s of every 8.1 s step is blocked waiting for gradient sync."
> Fix: `model = DDP(model, bucket_cap_mb=256)`
> Gain: `speedup_estimator: ≈ 1.5× faster per step`

**GPU idle / DataLoader**:
> Root cause: "GPU is idle 23% of profile time. `gpu_idle_gaps` attributes 78% of gaps to
> DataLoader workers — CPU cannot prefetch fast enough."
> Fix: `DataLoader(..., num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)`
> Gain: `speedup_estimator: ≈ 1.3× faster per step`

**H2D spread-out**:
> Root cause: "H2D transfers account for 22% of span. Every step copies a fresh batch from
> CPU — GPU stalls waiting for data to arrive."
> Fix: `DataLoader(..., pin_memory=True, num_workers=8, prefetch_factor=4)`
> Gain: `speedup_estimator: ≈ 1.2× faster per step`

**Slow layer (NVTX)**:
> Root cause: "`TransformerLayer.12.attn_fwd` accounts for 34% of GPU time — attention
> is the bottleneck, not NCCL or idle."
> Fix: `nn.functional.scaled_dot_product_attention` (replaces manual QKV + softmax)
> Gain: "2–4× attention speedup typical on A100/H100 with FlashAttention backend"

**Compute hotspot**:
> Root cause: "Top kernel accounts for 68% of GPU time. Tensor cores are inactive (FP32
> fallback) — peak throughput is 16× lower than BF16."
> Fix: `model.to(torch.bfloat16)` + `torch.set_float32_matmul_precision('high')`
> Gain: `speedup_estimator: ≈ 2.1× faster per step`

---

## 3. Pointers

- Deeper NCCL topology (communicator tables, per-GPU imbalance): see `DISTRIBUTED.md`
- MFU math and model FLOPs formulas: see `MFU.md`
- Inference framing (ms/token metrics): see `M1_AUTO.md` §6 "Inference framing"
