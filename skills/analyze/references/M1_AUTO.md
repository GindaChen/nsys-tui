# Mode 1 — Auto Triage

Reference for `/nsys-ai` Mode 1. **Read `PRINCIPLES.md` first** — §4 guards, §5 evidence,
§7 fail template, §10 checklist.

## 1. Precondition gate

§4.1 rows 1–4 (PRINCIPLES.md). Any fail → render §7 template and abort.

## 2. Stages

| # | Question | Condition |
|---|----------|-----------|
| 1 | Profile path | Only if not supplied |
| 2 | "Training or inference?" | Only if `fingerprint.framework == "Generic CUDA"` AND no top-10 kernel matches `ncclDevKernel*`, `flash_*`, `ampere_*gemm*`, `volta_*gemm*`, `cutlass_*`, `sm80_*`, `sm90_*`, `void at::*` |
| 3 | "Detected N iterations. Analyze 2 to N-1? (skips JIT warmup)" | Only if `iteration_timing` reports ≥3 iterations |

Stage 2 answer overrides `fingerprint.framework` for session only ("inference" → ms/token
framing; "training" → look for `_bwd`/`wgrad`/`dgrad`). No file writes.

**Device 0 auto-retry** (silent, NOT a stage): if `overlap.error == "no kernels found"` AND
`overlap.available_devices` non-empty, re-run manifest with `-p device=N` (lowest available).
Tell user once: `"Detected active GPU: device N"`.

## 3. Skills

Primary: `nsys-ai skill run profile_health_manifest <profile> --format json`
→ follow §4 routing from `suspected_bottleneck`.

Fallback (if `suspected_bottleneck` empty AND `root_causes[]` empty):
`nsys-ai skill run root_cause_matcher <profile> --format json`
→ re-route via §4 if patterns found; still empty → offer menu 2/3/4/6.

Device propagation: if auto-retry picked device N, pass `-p device=N` to all subsequent
device-scoped skills (see PRINCIPLES.md §6 for the 13/1/9/10 partition).

## 4. Signals — routing from `suspected_bottleneck`

| Keyword match | Drill into | Skills invoked |
|---------------|------------|----------------|
| `nccl` / `comm` | Mode 2 | `overlap_breakdown`, `nccl_breakdown`, `kernel_overlap_matrix` |
| `sync` / `idle` / `bubble` | Mode 6 | `gpu_idle_gaps`, `sync_cost_analysis`, `kernel_launch_overhead`, `cpu_gpu_pipeline` (+`pipeline_bubble_metrics` if NCCL) |
| `hotspot` / `kernel` | Mode 3 | `top_kernels`, `kernel_launch_pattern`, `tensor_core_usage` |
| `h2d` / `transfer` | Mode 4 | `memory_bandwidth`, `memory_transfers`, `h2d_distribution` |

### 4.1 CLI-field schema contract (source of truth — do NOT restate in other mode refs)

| Field | Type | Meaning |
|-------|------|---------|
| `gpu` | str | GPU name (not `gpu_name`) |
| `profile_span_ms` | float | Total span in ms |
| `fingerprint.framework` | str | "vLLM" / "Megatron-LM" / "Generic CUDA" / … |
| `fingerprint.distributed` | bool | **Unreliable alone** — also check `nccl.collectives` |
| `nccl.collectives` | int | >0 = multi-GPU workload |
| `top_kernels[]` | list | `{name, total_ms, count}` |
| `overlap.overlap_pct` | float | <30% with NCCL → comm-bound (Mode 2) |
| `overlap.error` | str | "no kernels found" → device auto-retry |
| `overlap.available_devices` | dict | `{N: kernel_count}` — retry target |
| `idle.idle_pct` | float | >15% → Mode 6 |
| `sync.sync_density_pct` | float | >20% → Mode 6 |
| `root_causes[].pattern` | str | e.g. "NCCL Serialization" |
| `root_causes[].severity` | str | `critical` / `warning` / `info` |
| `suspected_bottleneck` | str | Free-form — keyword-match per §4 |
| `data_quality.overhead_pct` | float | >1% → surface note (not a block) |

## 5. Cross-mode exits

Suggest specialist mode after delivery only if a second critical finding exists (soft cap:
≤2 suggestions per delivery message — UX invariant 7).

| Mode | Reference | When to suggest |
|------|-----------|----------------|
| 2 | `M2_COMMS.md` | `nccl`/`comm` bottleneck |
| 3 | `M3_COMPUTE.md` | `hotspot`/`kernel` bottleneck |
| 4 | `M4_MEMORY.md` | `h2d`/`transfer` bottleneck |
| 5 | `M5_NVTX.md` | layer attribution needed |
| 6 | `M6_IDLE.md` | `sync`/`idle`/`bubble` bottleneck |
| 7 | `M7_CUTRACER.md` (C1) | top kernel > 60%, custom/Triton |
| 8 | `M8_DIFF.md` (C2) | regression analysis requested |
| 9 | `M9_VARIANCE.md` (C2) | iteration spikes detected |

## 6. Delivery

Follow `PRINCIPLES.md §5` for evidence + timeline URL. Then 3-part summary:

1. **Root cause** — bottleneck + quantified impact (see `ROOT_CAUSE.md §2` for examples).
2. **Specific fix** — code example (see `ROOT_CAUSE.md §1` matrix).
3. **Expected gain** — `speedup_estimator` if NVTX present; omit otherwise.

**Inference framing**: `fingerprint.framework ∈ {vLLM, SGLang, TensorRT}` OR no
`_bwd`/`wgrad`/`dgrad` in top-10 → metrics in ms/token; label "inference workload".

Optional text report (on explicit request): `PRINCIPLES.md §5.4`.
Required output checklist: `PRINCIPLES.md §10` — run before ending the turn.
