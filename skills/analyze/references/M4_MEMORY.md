# Mode 4 — Memory (H2D / D2H / bandwidth)

Reference for `/nsys-ai` Mode 4. **Read `PRINCIPLES.md` first** — §4 guards, §5 evidence,
§7 fail template, §10 checklist.

---

## 1. Precondition gate

No hard gate beyond §4.1 row 3 (global kernel-table check). Soft note if `memory_bandwidth`
returns empty (no memcpy table in profile): "No CUDA memory copies detected — profile may
have been captured without memory tracing. Mode 3 (compute) or Mode 6 (idle) may be more
relevant."

---

## 2. Stages

| # | Question | Condition/Default |
|---|----------|-------------------|
| 1 | Profile path | Only if not supplied |

No sub-menu. Runs all three memory skills in sequence; classifies the transfer pattern.

---

## 3. Skills

Run in order:

```bash
nsys-ai skill run memory_bandwidth <profile> --format json
nsys-ai skill run memory_transfers <profile> --format json
nsys-ai skill run h2d_distribution <profile> --format json
```

Device propagation: `memory_bandwidth` and `h2d_distribution` accept `-p device=N`;
`memory_transfers` does not. See PRINCIPLES.md §6.

---

## 4. Signals

**From `memory_bandwidth`** (per-row fields: `copyKind`, `op_count`, `total_mb`,
`total_dur_ms`, `avg_bandwidth_gbps`, `peak_bandwidth_gbps`; also `_metadata.anomalies`):

| Field / condition | Diagnosis | Action |
|-------------------|-----------|--------|
| H2D `total_dur_ms` / `profile_span_ms` > 15% | Memory-bound critical path | pin memory; pre-stage data on GPU |
| `avg_bandwidth_gbps` << `peak_bandwidth_gbps` | Many small transfers (low efficiency) | batch transfers; use `torch.Tensor.pin_memory()` |
| D2H large with high `op_count` | Metric logging overhead (`.item()` / `.numpy()` in loop) | log only every N steps; avoid `.item()` in hot loop |
| `_metadata.anomalies` non-empty (transfers > 10 MB) | Single large blocking copy | check timeline; consider async copy |

**From `memory_transfers`** (per-row fields: `copyKind`, `count`, `total_mb`, `total_ms`):

| Field / condition | Diagnosis | Action |
|-------------------|-----------|--------|
| P2P `total_mb` large | Cross-device tensor copies not hidden by NCCL | check tensor placement; overlap with comm |
| H2D `count` >> other directions | Repeated small copies every step | accumulate on CPU then single large H2D; or keep tensor on GPU |

**From `h2d_distribution`** (pattern metadata: `type`, `detail`, `spike_seconds`, `spikes`):

| `type` value | Diagnosis | Action |
|-------------|-----------|--------|
| `init_heavy` | Model weight loading — normal | no action required |
| `spread_out` | Every step copies data to GPU | `pin_memory=True`, `num_workers ≥ 4`, `prefetch_factor=2` |
| `spike` | Sudden H2D burst mid-training | check `spike_seconds` timestamps; likely `.cpu()` + back to GPU in loop |

---

## 5. Cross-mode exits

After delivery, suggest a second mode only if a distinct critical finding exists.

- H2D spike at specific `spike_seconds` → suggest **Mode 5** (`kernel-attribution` sub-focus
  to locate the NVTX region responsible)
- P2P large AND `nccl.collectives > 0` → suggest **Mode 2** (NCCL may be using P2P unnecessarily)

---

## 6. Delivery

Follow `PRINCIPLES.md §5` for evidence build + timeline URL. Then 3-part summary:

1. **Root cause** — transfer class + direction + quantified waste:
   > "H2D transfers account for 22% of profile time. `h2d_distribution` shows a `spread_out`
   > pattern — every training step copies a fresh batch of tensors from CPU to GPU, preventing
   > GPU from staying fully occupied."

2. **Specific fix** — matching the transfer class:
   - Spread-out: `DataLoader(..., pin_memory=True, num_workers=8, prefetch_factor=4, persistent_workers=True)`
   - Spike (rogue `.item()` or `.cpu()`): locate the call via Mode 5; remove from inner loop
   - D2H metric logging: `if step % 100 == 0: loss_val = loss.item()`
   - Large one-time copy: `model.cuda()` before training loop, not inside

3. **Expected gain** — `speedup_estimator` if NVTX present; omit otherwise.

See `ROOT_CAUSE.md §1` for the cross-mode cause→fix matrix.
