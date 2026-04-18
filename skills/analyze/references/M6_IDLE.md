# Mode 6 — Idle / Sync

Reference for `/analyze` Mode 6. **Read `PRINCIPLES.md` first** — §4 guards, §5 evidence,
§7 fail template, §10 checklist.

---

## 1. Precondition gate

No hard precondition. Mode 6 runs on any profile with kernel activity (PRINCIPLES.md §4.1
row 3 is a global guard — reaching Mode 6 guarantees at least one kernel is present). If
manifest `idle.idle_pct < 5` AND `sync.sync_density_pct < 5`,
note: "GPU appears busy — idle/sync overhead is low; Mode 3 or Mode 2 may be more relevant."
Do not block.

---

## 2. Stages

| # | Question | Condition/Default |
|---|----------|-------------------|
| 1 | Profile path | Only if not supplied |
| 2 | Sub-focus: `gaps` / `sync` / `launch` / `cpu-pipeline` / `all` | Optional; default `all` |

Skip Stage 2 if keyword already implies sub-focus (e.g. "dataloader" → `cpu-pipeline`,
"cudaStreamSynchronize" → `sync`, "launch overhead" → `launch`).

---

## 3. Skills

| Sub-focus | Skills (in order) |
|-----------|-------------------|
| `gaps` | `gpu_idle_gaps -p min_gap_ns=1000000` → `stream_concurrency` |
| `sync` | `sync_cost_analysis` |
| `launch` | `kernel_launch_overhead` |
| `cpu-pipeline` | `cpu_gpu_pipeline` → `thread_utilization` |
| `all` (default) | `gpu_idle_gaps` → `stream_concurrency` → `sync_cost_analysis` → `kernel_launch_overhead` → `cpu_gpu_pipeline` → `thread_utilization` → `module_loading` → `gc_impact` (if `starvation_events > 0`) → `pipeline_bubble_metrics` (if `nccl.collectives > 0`) |

Device propagation: `gpu_idle_gaps` accepts `-p device=N`. `stream_concurrency`,
`sync_cost_analysis`, `kernel_launch_overhead`, `cpu_gpu_pipeline` do not accept device.
See PRINCIPLES.md §6.

---

## 4. Signals

| Field / condition | Diagnosis | Action |
|-------------------|-----------|--------|
| manifest `idle.idle_pct > 15` OR `pipeline_bubble_metrics.bubble_pct > 15` | GPU starved | check DataLoader / CPU dispatch |
| `sync_cost_analysis.sync_density_pct > 20` | Excessive CPU→GPU syncs | remove `.item()` / `.cpu()` in loop |
| `kernel_launch_overhead.overhead_us` dominated by one caller | High CPU dispatch latency | async launch; reduce Python overhead |
| `cpu_gpu_pipeline.starvation_events > 0` | CPU can't feed GPU fast enough | `num_workers`, `pin_memory`, `prefetch_factor` |
| `thread_utilization` GIL contention (non-empty result) | DataLoader threads blocking | `persistent_workers=True`, reduce workers |
| `module_loading` stalls on step 0 | JIT compilation | pre-warm; skip step 0 in benchmarks |
| `gc_impact` `cudaMalloc/Free` spikes | Python GC stalls GPU | `torch.cuda.empty_cache()` placement |
| `pipeline_bubble_metrics.bubble_pct > 10` | PP micro-batch bubble | increase `num_micro_batches`; check schedule |
| `stream_concurrency` low despite multi-stream | Kernel serialization | check stream assignment / dependencies |

**Idle gap attribution**: `gpu_idle_gaps` attributes each gap via `attribution.description`
and `attribution.top_apis[].name`. Look for `DataLoader` / `cudaMemcpyAsync` /
`cudaStreamSynchronize` in those fields.

---

## 5. Cross-mode exits

After delivery, suggest a second mode only if a distinct critical finding exists. Cap 2 chains.

- Large idle gaps attributed to NVTX region → suggest **Mode 5** (layer attribution)
- `nccl.collectives > 0` AND idle gaps correlate with NCCL timeline → suggest **Mode 2**
- Idle gap after step 0 only (JIT) → note: re-profile with `torch.cuda.nvtx.range` to verify

---

## 6. Delivery

Follow `PRINCIPLES.md §5` for evidence build + timeline URL. Then 3-part summary:

1. **Root cause** — stall class + quantified waste:
   > "GPU is idle 23% of profile time. `gpu_idle_gaps` attributes 78% of gaps to
   > DataLoader workers — the CPU cannot prefetch fast enough to keep the GPU busy."

2. **Specific fix** — matching the stall class:
   - DataLoader starvation: `DataLoader(..., num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)`
   - `.item()` sync loop: remove from inner loop; accumulate as tensor, call once outside
   - Launch overhead: batch small kernel launches; use CUDA graphs for repeated patterns
   - GC stall: `torch.backends.cuda.matmul.allow_tf32 = True` + `gc.disable()` in training loop
   - PP bubble: `num_micro_batches = pipeline_stages * 2` (1F1B schedule)

3. **Expected gain** — from `speedup_estimator` if NVTX present; omit otherwise.
