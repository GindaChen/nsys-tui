# Mode 9 — Variance (iteration spikes)

Reference for `/nsys-ai` Mode 9. **Read `PRINCIPLES.md` first** — §4 guards, §5 evidence,
§7 fail template, §10 checklist.

---

## 1. Precondition gate

§4.1 row 8 / §4.3 row 3 — run `nsys-ai skill run iteration_timing <profile> --format json`.
If result is `[]`:

```
Mode 9 requires a user choice.

Why: no NVTX iteration markers and kernel-gap heuristic found no iterations.
Fix: add a top-level NVTX range per step, e.g.:
  with torch.cuda.nvtx.range("sample_0"): ...
then re-profile.
Alternative: Mode 5 Path B (kernel-name heuristics, no per-iteration attribution).

Reply "B", re-profile, or "stop".
```

---

## 2. Stages

| # | Question | Condition / Default |
|---|----------|--------------------|
| 1 | Profile path | Required if not supplied |
| 2 | Granularity: `iterations` / `ranks` / `kernels` | Optional; default `iterations` |
| 3 | Metric: `max/min` / `p95/p50` / `p99/p50` | Optional; default `max/min` |

Stage 2 `ranks` requires `nccl.collectives > 0` from manifest; if zero, offer `iterations`
or `kernels` only.

---

## 3. Skills

```bash
# 1. Detect iterations and surface slow outliers
nsys-ai skill run iteration_timing <profile> --format json

# 2. Drill into a slow iteration (N from step 1 — skip index 0)
nsys-ai skill run iteration_detail <profile> --format json -p iteration=<N>

# 3. Top kernels in that iteration only
nsys-ai skill run top_kernels <profile> --format json --iteration <N>

# 4. Idle gaps in that iteration
nsys-ai skill run gpu_idle_gaps <profile> --format json --iteration <N>

# 5. Specific kernel instances in slow iteration
nsys-ai skill run kernel_instances <profile> --format json --iteration <N> -p name=<hot>

# 6. NCCL straggler check (only if nccl.collectives > 0)
nsys-ai skill run nccl_anomaly <profile> --format json
```

---

## 4. Signals

From `iteration_timing` (per-row: `iteration`, `duration_ms`; skip index 0) and
`iteration_detail` (fields: `duration_ms`, `vs_median`, `kernel_count`, `nccl_count`,
`compute_ms`):

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Every Nth iteration slow (N = 4/8/16) | Gradient accumulation or eval step | Expected — confirm NVTX label; advise profiling eval separately |
| First 2–3 iters slow, rest stable | JIT compilation / CUDA cache warmup | Expected — exclude iter 0 from benchmarks; profile from iter 2+ |
| Random spikes, no period | DataLoader I/O latency variance | `pin_memory=True`; `num_workers ≥ 4`; `prefetch_factor=2` |
| Slow iters have elevated `nccl_count` | NCCL retry or straggler rank | `NCCL_DEBUG=INFO`; check network link; Mode 2 |
| `duration_ms` increases monotonically over run | GPU thermal throttle | `nvidia-smi -q -d CLOCK` → check `THERMAL` status; reduce power limit |
| Alternating fast/slow pairs | Prefetch pipeline stall or double-buffer underrun | Increase prefetch depth; check `num_workers` |
| `gpu_idle_gaps` large in slow iter, `compute_ms` unchanged | CPU starvation (DataLoader/GIL) | Mode 6 for idle breakdown |

**Variance ratio**: `max_ms / median_ms > 1.5` → report as outlier. < 1.5 → within normal
(no actionable spike; tell user and close).

---

## 5. Cross-mode exits

| Mode | When to suggest |
|------|----------------|
| 2 (Comms) | Slow iters correlate with elevated `nccl_count` → straggler rank analysis |
| 6 (Idle) | Slow iter has large `gpu_idle_gaps` with `compute_ms` unchanged → CPU cause |

---

## 6. Delivery

Follow `PRINCIPLES.md §5` for evidence + timeline URL. Then 3-part summary:

1. **Root cause** — outlier fraction + pattern:
   > "3 of 12 iterations are ≥ 2× median (1420 ms vs 680 ms median). Pattern: random spikes.
   > Top cause: DataLoader stall — `gpu_idle_gaps` spikes 740 ms in slow iters vs 12 ms normal."

2. **Specific fix** — matched to pattern:
   - DataLoader: `DataLoader(num_workers=4, pin_memory=True, prefetch_factor=2)`
   - JIT: exclude iter 0 from benchmark; profile from iteration 2+
   - Thermal: `nvidia-smi -pl <TDP_W>` to enforce power limit
   - NCCL retry: `NCCL_DEBUG=INFO`; investigate straggler rank (Mode 2)

3. **Expected gain** (qualitative):
   > "Eliminating DataLoader stalls should reduce spiked iterations from 1420 ms to ~680 ms (2.1×)."
