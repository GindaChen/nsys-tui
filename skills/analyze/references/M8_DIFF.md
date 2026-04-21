# Mode 8 — Diff (before vs after)

Reference for `/nsys-ai` Mode 8. **Read `PRINCIPLES.md` first** — §4 guards, §5 evidence,
§7 fail template, §10 checklist.

---

## 1. Precondition gate

Three checks; any failure → §7 template and abort:

1. **Both profiles pass §4.1 rows 1–4** — exists, has kernel table. If one fails:
   "Mode 8 blocked (before=`<a>` after=`<b>`): `<which>` has no kernel table.
   Fix: re-export or pick a valid profile."
2. **Same-inode guard** (§4.1 row 7): `os.stat(before).st_ino != os.stat(after).st_ino`.
   If equal: "Mode 8 blocked. Why: both paths resolve to the same file.
   Fix: provide two distinct profiles. Alternative: Mode 1 for single-profile analysis."
3. **Workload check**: after running diff, if `warnings[]` is non-empty surface each string
   before proceeding — do NOT block on warnings, but note them prominently. If both
   `top_regressions` and `top_improvements` are empty, warn "No kernel changes detected."

---

## 2. Stages

| # | Question | Condition / Default |
|---|----------|--------------------|
| 1 | Baseline profile path (before) | Required if not supplied |
| 2 | Candidate profile path (after) | Required if not supplied |
| 3 | Iteration anchor: `auto` / `N` | Optional; default `auto` (global diff) |

**Stage 3 auto strategy** — run `nsys-ai iters` on each side first:

| iters result | Action |
|-------------|--------|
| ≥ 3 on both AND median `duration_ms > 10` | Use `--iteration 1` (skip JIT iteration 0) |
| ≥ 3 on both but median `duration_ms ≤ 10` | Global diff; note iterations are heuristic noise |
| 2 on either | Use `--iteration 1`; note single-iteration caveat |
| 1 on either | Global diff (no `--iteration`); note profile too short |
| Counts differ | Global diff; surface "iteration count mismatch" warning first |

---

## 3. Skills

```bash
# 1. Check iteration counts for Stage 3 decision
nsys-ai iters <before>
nsys-ai iters <after>

# 2a. Global diff (default — no --iteration)
nsys-ai diff <before> <after> --format json

# 2b. Iteration-scoped diff (after iters confirm alignment)
nsys-ai diff <before> <after> --iteration 1 --format json

# 2c. GPU-scoped diff (multi-GPU; focus on one device)
nsys-ai diff <before> <after> --gpu 0 --format json

# 3. Search for NVTX region name before drilling (PRINCIPLES §3 rule 1: never guess)
nsys-ai search <before> --query <keyword> --type nvtx
nsys-ai search <after>  --query <keyword> --type nvtx

# 4. Visual diff timeline
nsys-ai diff-web <before> <after>
```

---

## 4. Signals

From `nsys-ai diff --format json` (top-level keys: `warnings[]`, `top_regressions[]`,
`top_improvements[]`, `nvtx_regressions[]`, `nvtx_improvements[]`, `overlap`; per-kernel
fields: `name`, `demangled`, `delta_ns`, `before_share`, `after_share`, `classification`):

| Field / condition | Diagnosis | Action |
|-------------------|-----------|--------|
| `warnings[]` non-empty | Mismatch: schema diff, very different kernel sets, or missing overlap data | Surface each string; skip per-kernel conclusions if workloads differ |
| `top_regressions[0].delta_ns / 1e6 > 10` | Kernel regression ≥ 10 ms | Examine `name`, `before_share` vs `after_share`; correlate with code change |
| `top_regressions[0].classification = "new"` | Kernel appeared after change | New op introduced; `demangled` for source hint; ask user what changed |
| `nvtx_regressions[0].delta_ns > 0` AND `top_regressions` small | NVTX-level slowdown without kernel cause | CPU starvation, DataLoader, or GIL — cross into Mode 6 |
| `overlap.after.overlap_pct < overlap.before.overlap_pct - 10` | Lost compute/NCCL pipeline overlap (≥ 10 pp) | NCCL config changed; bucket size reduced; cross into Mode 2 |
| Both `top_regressions` and `top_improvements` empty | No kernel change detected | Verify profiles are different; re-run global diff without `--iteration` |

**Skip iteration 0** in any `--iteration` diff — JIT warmup inflates it (PRINCIPLES §3 rule 4).

---

## 5. Cross-mode exits

| Mode | When to suggest |
|------|----------------|
| 5 (NVTX) | `nvtx_regressions` non-empty — attribute regression to specific layer |
| 2 (Comms) | `overlap.after.overlap_pct` drops > 10 pp vs `overlap.before.overlap_pct` |
| 6 (Idle) | `nvtx_regressions` spike with `top_regressions` small — CPU/DataLoader |

---

## 6. Delivery

**Evidence**: follow PRINCIPLES §5.6 (Mode 8 adaptation) — run `evidence build <after>`
only, never the before profile. Craft a findings JSON with regression labels, then serve
the after-profile timeline:

```bash
nsys-ai skill run kernel_instances <after> --format json -p name=<hot_kernel>
# Use start_ns / end_ns from output to build findings:
nsys-ai timeline-web <after> --findings /tmp/findings.json
```

`findings.json` shape (use `type: "region"` — `"regression"` is not a recognized finding
type and is silently ignored by the timeline renderer; encode regression context in
`label` / `note`):
```json
{"findings": [{"type": "region", "label": "Regression: <name> +<delta_ms>ms",
  "note": "Regressed in after profile vs before",
  "start_ns": <start>, "end_ns": <end>, "severity": "critical"}]}
```

Then 3-part summary:

1. **Root cause** — regression magnitude + trigger:
   > "`flash_bwd` accounts for +31% runtime after the change (+58 ms, 62% → 82% share).
   > Classification: regression. Likely cause: sequence length increase."

2. **Specific fix** — matched to `classification`:
   - `regression` in `*gemm*` / `*flash*` → check dtype, shape alignment, matmul tile size
   - `regression` in `nccl*` → bucket size, async overlap (Mode 2)
   - `new` kernel → unnecessary op introduced; bisect the commit that added it
   - NVTX-only regression → CPU pipeline stall (Mode 6)

3. **Expected gain** (qualitative):
   > "Reverting the sequence-length change should recover ~58 ms per step."

**Always note** whether diff was global or iteration-scoped, and list any `warnings[]` items.
