# Mode 2 — Comms (NCCL / overlap)

Reference for `/analyze` Mode 2. **Read `PRINCIPLES.md` first** — §4 guards, §5 evidence,
§7 fail template, §10 checklist.

---

## 1. Precondition gate

Check `nccl.collectives` from `profile_health_manifest`:

```
if nccl.collectives == 0:
    "Mode 2 blocked. Why: single-GPU profile — no NCCL activity recorded.
    Fix: run with multi-GPU workload. Alternative: Mode 1 or Mode 6 for single-GPU."
```

---

## 2. Stages

| # | Question | Condition/Default |
|---|----------|-------------------|
| 1 | Profile path | Only if not supplied |
| 2 | Sub-focus: `overlap` / `outlier` / `topology` / `per-rank-variance` | Optional; default `overlap` |

Skip Stage 2 if user provided a keyword that maps to a sub-focus
(e.g. "straggler" → `per-rank-variance`, "communicator" → `topology`).

---

## 3. Skills

| Sub-focus | Skills (in order) |
|-----------|-------------------|
| `overlap` (default) | `overlap_breakdown` → `kernel_overlap_matrix` |
| `outlier` | `nccl_anomaly -p threshold=3.0` → `nccl_breakdown` |
| `topology` | `nccl_communicator_analysis` → `nccl_breakdown` (requires profile exported with `nsys export --include-blobs=true`; returns diagnostic if blobs missing) |
| `per-rank-variance` | `nccl_breakdown` (all devices) → `nccl_anomaly -p threshold=3.0` |

Device propagation: if Mode 1 auto-retry selected device N, pass `-p device=N` to
`overlap_breakdown`, `kernel_overlap_matrix`, `nccl_breakdown`, and
`nccl_communicator_analysis`. `nccl_anomaly` is not device-scoped. See PRINCIPLES.md §6.

**Collective type → parallelism strategy** (from `nccl_breakdown` stream grouping):

| Dominant collective | Strategy |
|--------------------|----------|
| AllReduce | DDP gradient sync |
| ReduceScatter + AllGather | FSDP / ZeRO-2/3 sharded params |
| AllGather (forward only) | Tensor Parallelism or FSDP inference |
| SendRecv | Pipeline Parallelism (P2P) |
| Broadcast | Checkpoint sync / param broadcast at init |

---

## 4. Signals

| Field / condition | Diagnosis | Action |
|-------------------|-----------|--------|
| `overlap_pct < 30` (from `overlap_breakdown`) | NCCL serialized with compute | critical; DDP bucket or FSDP fix |
| `overlap_pct` 30–60 | Partial pipelining | warning; tune prefetch |
| `overlap_pct > 60` | NCCL well-hidden | info; not the bottleneck |
| `nccl_anomaly.ratio_to_avg > 3` | Straggler NCCL op or rank | check NIC / dataset imbalance |
| One device `total_ms >> others` (from `nccl_breakdown -p device=N`) | Single straggler GPU | NCCL_DEBUG=INFO; check switch port |
| All devices `total_ms` up uniformly (from `nccl_breakdown`) | Cross-node congestion | NCCL_SOCKET_IFNAME; check IB/RoCE |
| `overlap_breakdown.idle_ms` up across devices, `nccl_breakdown.total_ms` flat | CPU launch overhead → Mode 6 | check kernel dispatch latency |

**Overlap thresholds are not hard boundaries** — FSDP/ZeRO-3 with prefetch typically
reaches 50–70%; DDP without bucketing often stays below 20%.

---

## 5. Cross-mode exits

After delivery, suggest a second mode only if a distinct critical finding exists. Cap 2 chains.

- `overlap_pct < 30` AND `idle.idle_pct > 15` → suggest **Mode 6** (idle underlies NCCL gap)
- `nccl_anomaly.ratio_to_avg > 3` across iterations → suggest **Mode 9** (variance / straggler)
- `nccl.collectives > 0` → run `pipeline_bubble_metrics` inline; if `bubble_pct > 10`,
  report PP bubble % in delivery

---

## 6. Delivery

Follow `PRINCIPLES.md §5` for evidence build + timeline URL. Then 3-part summary:

1. **Root cause** — NCCL class + quantified waste:
   > "NCCL AllReduce is serialized with compute (overlap = 18%, DDP default bucket 25 MB).
   > Estimated 3.4 s of every 8.1 s step is blocked waiting for gradient sync."

2. **Specific fix** — matching the collective type:
   - DDP serialized: `model = DDP(model, bucket_cap_mb=256)`
   - FSDP slow prefetch: `FSDP(..., forward_prefetch=True, limit_all_gathers=True)`
   - Straggler rank: check NIC on node; rebalance dataset sharding
   - Cross-node: `NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=0`

3. **Expected gain** — from `speedup_estimator` if NVTX present; omit otherwise.

See `DISTRIBUTED.md` for deeper NCCL topology reference (communicator tables, kernel name
patterns, per-GPU imbalance SQL).
