# Mode 7 — CUTracer (SASS analysis)

Reference for `/nsys-ai` Mode 7. **Read `PRINCIPLES.md` first** — §4 guards, §7 fail
template, §10 checklist. **Evidence exception**: Mode 7 skips `evidence build` — see
PRINCIPLES.md §5.5.

---

## 1. Precondition gate

Three checks in sequence — any failure aborts via §7 template:

1. **Tool chain**: `nsys-ai cutracer check` — §4.1 row 5.
   - Exits 0 only when Python package AND `.so` are both present (full SASS mode).
   - Exits 1 with "cutracer Python package : NOT FOUND" → hard block; direct user to `pip install cutracer`.
   - Exits 1 with "cutracer.so             : NOT FOUND" → soft note only; plugin proceeds in kernel-launch-logger
     mode (launch counts + warp dims, no SASS mix). Full SASS requires `nsys-ai cutracer install`
     (needs nvcc / g++ / make / git / libzstd-dev — §4.2 auto-handled at run time).

2. **GPU available**: `nvidia-smi` exits 0, **or** user already chose Modal in Stage 4.
   If no GPU: §4.3 row 2 — "Mode 7 requires a user choice. Why: no GPU on this host.
   Fix: run on GPU-enabled host. Alternative: Modal backend (`--backend modal`),
   ~$`<X>` at H100 default (see cost formula in §2). Reply `modal`, `local-gpu-ready`, or `stop`."

3. **Launch command**: user supplies a reproducible launch command (Stage 2). No default.
   Hint: `nsys-ai skill run schema_inspect <profile>` may surface the stored script name from
   profile metadata (only if captured via `nsys profile python train.py …`).

**Entry from Mode 3** — never auto-enter. All four must be true:
1. Hotspot kernel > 60% of total GPU time
2. Kernel is custom / Triton — NOT cuBLAS / cuDNN / FlashAttention
3. `tensor_core_usage` OR `arithmetic_intensity` is ambiguous for that kernel
4. User explicitly confirms re-run is acceptable

---

## 2. Stages

| # | Question | Condition / Default |
|---|----------|--------------------|
| 1 | Profile path | **Required** — positional arg for `cutracer run`/`plan`. If user gives only a kernel name with no profile: "Profile the workload first via Mode 1, then return here." |
| 2 | Launch command (`--launch-cmd`) | No default; try `schema_inspect` hint first |
| 3 | Output directory | `./cutracer_out` |
| 4 | Backend: `local` / `modal` / `modal-run` | `local` if `nvidia-smi` OK; else prompt |
| 5 | Cost confirmation | Must reply `yes` — compute and display before asking |

**Stage 5 cost display** (render computed values — do NOT echo the formula):

```
Cost estimate (profile span ≈ <span_s> s, 1.5× instrumentation overhead):
  local run:    ≈ <local_s> s of GPU time
  modal H100:   ≈ $<X.XX>   (default --modal-gpu H100; ~$4.56/hr)
  modal A100:   ≈ $<X.XX>   (~$2.55/hr)
  modal A10:    ≈ $<X.XX>   (~$1.10/hr)

Proceed with local backend? Reply "yes" to continue, "modal" to switch, "stop" to abort.
```

**Cost formula** (`span_ms` from `profile_health_manifest`):
```
local_run_seconds = span_ms / 1000 × 1.5
modal_usd = local_run_seconds × rate
  rates (2026-04): H100 $0.00127/s · A100 $0.00071/s · A10 $0.00031/s
```
Round seconds to 1 decimal; USD to 2 decimals.

---

## 3. Commands

```bash
# 1. Verify tool chain
nsys-ai cutracer check

# 2. Inspect candidate kernels
nsys-ai cutracer plan <profile> --top-n 3

# 3. Generate instrumentation script (confirm launch-cmd with user first)
nsys-ai cutracer plan <profile> --script --launch-cmd '<cmd>' --save run_cutracer.sh

# 4a. Run locally (after cost confirm)
bash run_cutracer.sh ./cutracer_out

# 4b. OR: Modal backend
nsys-ai cutracer run <profile> --launch-cmd '<cmd>' --backend modal --modal-gpu H100

# 5. Analyze results
nsys-ai cutracer analyze <profile> ./cutracer_out --format json
```

---

## 4. Signals

From `nsys-ai cutracer analyze <profile> <outdir> --format json` (per-kernel fields:
`kernel_name`, `pct_of_gpu`, `total_instructions`, `instruction_mix_pct`,
`tensor_core_active`, `bottleneck`, `bank_conflict_hint`, `achieved_warps`, `top_stalls[]`):

| Field / condition | Diagnosis | Action |
|-------------------|-----------|--------|
| `bottleneck = "memory"` | Memory-bound (LDG/STG stalls > 25% or `instruction_mix_pct.memory > 30%`) | Tile into shared memory; use `__ldg` / `cp.async`; reduce redundant loads |
| `bottleneck = "compute"` | Compute-bound (`instruction_mix_pct.compute + instruction_mix_pct.tensor > 60%`) | Tune register reuse; increase tile size; try `torch.compile` |
| `bottleneck = "sync"` | Sync-bound (`instruction_mix_pct.sync > 15%`) | Replace `__syncthreads` with `__syncwarp`; reduce barrier frequency |
| `bank_conflict_hint = true` | Shared memory bank conflicts (high LDS/STS stall) | Pad shared memory arrays by 1 element per row |
| `tensor_core_active = false` AND eligible kernel | Tensor cores inactive | Force BF16/TF32; check matmul shape — pad to multiple of 8/16 |
| `top_stalls[0].stall_score > 0.3` | One opcode dominates stall cycles | Focus on that instruction; see mnemonic for LDG/STG/FFMA/BAR diagnosis |
| `bottleneck = "unknown"` AND `total_instructions = 0` | `.so` not installed — launch-logger mode only | Run `nsys-ai cutracer install` then re-run for SASS mix |

**Kernel-launch-logger fallback** (when `.so` absent): `cutracer analyze` still reports
`pct_of_gpu` and `achieved_warps` but `instruction_mix_pct` will be empty and `bottleneck`
will be `"unknown"`. Surface the launch-count data; advise building the `.so` for full SASS analysis.

---

## 5. Cross-mode exits

Mode 7 is **terminal** — no chained mode suggestions.

If `bottleneck = "unknown"` due to missing `.so`:
> "Full SASS analysis requires the CUTracer `.so`. Run `nsys-ai cutracer install`
> (requires CUDA toolkit + g++) then repeat Mode 7."

---

## 6. Delivery

**Evidence**: Mode 7 skips `evidence build`. Open an unannotated timeline for context:
```bash
nsys-ai timeline-web <profile>
```
Print the URL emitted by `timeline-web` (`http://127.0.0.1:PORT`). No `--findings` flag.

Then 3-part summary:

1. **Root cause** — SASS verdict + kernel share:
   > "`my_triton_kernel` accounts for 68% of GPU time. SASS bottleneck: MEMORY-BOUND.
   > Top stall opcode is `LDG` (stall score 0.41) — global memory latency dominates."

2. **Specific fix** — matching `bottleneck`:
   - `memory`: tile into shared memory; use `cp.async` prefetch; reduce redundant global loads
   - `compute`: increase register reuse; tile size tuning; `torch.compile(mode='max-autotune')`
   - `sync`: replace `__syncthreads` with `__syncwarp`; fuse loops to reduce barrier count
   - `bank_conflict`: pad shared memory arrays (`__shared__ float s[32][33]`)
   - `.so` missing: "Build `.so` via `nsys-ai cutracer install` for SASS-level breakdown"

3. **Expected gain** — qualitative only (`speedup_estimator` not applicable in Mode 7):
   > "Eliminating LDG stalls typically yields 1.5–2× speedup for memory-bound CUDA kernels."

**Always note** that Mode 7 required a re-run of the workload: "Note: CUTracer analysis
required re-running the workload with instrumentation overhead (~1.5×)."
