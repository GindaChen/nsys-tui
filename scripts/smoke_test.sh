#!/usr/bin/env bash
# Smoke test for nsys-ai plugin — validates that every CLI command referenced
# in skills/analyze/SKILL.md + M*.md still works with the installed nsys-ai version.
#
# Usage:  ./scripts/smoke_test.sh <profile.sqlite>
# Exit 0 if all checks succeed, non-zero otherwise.
#
# Stage A: Mode 1 end-to-end (manifest → evidence → timeline surface check)
# Stage B1: Mode 2 + Mode 6 drill-down skills
# Stage B2: Mode 3, Mode 4, Mode 5 drill-down skills + field-shape validation

set -euo pipefail

PROFILE="${1:-}"
if [[ -z "$PROFILE" || ! -f "$PROFILE" ]]; then
  echo "usage: $0 <profile.sqlite>" >&2
  exit 2
fi

FAIL=0

run() {
  local label="$1"; shift
  printf "  %-50s " "$label"
  if "$@" >/dev/null 2>&1; then
    echo "OK"
  else
    echo "FAIL"
    FAIL=$((FAIL+1))
  fi
}

run_capture() {
  local label="$1"; shift
  local out="$1"; shift
  printf "  %-50s " "$label"
  if "$@" >"$out" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL"
    FAIL=$((FAIL+1))
  fi
}

check_regex() {
  local label="$1"; local file="$2"; local pattern="$3"
  printf "  %-50s " "$label"
  if grep -qE "$pattern" "$file" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL (pattern /$pattern/ not found)"
    FAIL=$((FAIL+1))
  fi
}

check_json_key() {
  # Verify a key path exists and is non-null in a JSON file.
  # key_path uses dot notation: e.g. "nccl.collectives" or "idle.idle_pct"
  local label="$1"; local file="$2"; local key_path="$3"
  printf "  %-50s " "$label"
  local py_expr
  py_expr="import json,sys; d=json.load(open('$file')); d=d[0] if isinstance(d,list) else d"
  for part in ${key_path//./ }; do
    py_expr+="; d=d['$part']"
  done
  py_expr+="; sys.exit(0 if d is not None else 1)"
  if python3 -c "$py_expr" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL (key $key_path missing or null)"
    FAIL=$((FAIL+1))
  fi
}

# Detect profile capabilities once upfront
HAS_NVTX=$(sqlite3 "$PROFILE" "SELECT COUNT(*) FROM sqlite_master WHERE name='NVTX_EVENTS';" 2>/dev/null || echo 0)
HAS_NCCL=$(sqlite3 "$PROFILE" "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL';" 2>/dev/null || echo 0)

MANIFEST_OUT="$(mktemp)"
ITER_OUT="$(mktemp)"
TC_OUT="$(mktemp)"
MEM_XFER_OUT="$(mktemp)"
H2D_OUT="$(mktemp)"
trap 'rm -f "$MANIFEST_OUT" "$ITER_OUT" "$TC_OUT" "$MEM_XFER_OUT" "$H2D_OUT" /tmp/findings_smoke.json' EXIT

# ── Top-level ─────────────────────────────────────────────────────────────────
echo "== Top-level commands =="
run "nsys-ai --help"             nsys-ai --help
run "nsys-ai skill list"         nsys-ai skill list
run "schema_inspect"             nsys-ai skill run schema_inspect "$PROFILE" --format json
printf "  %-50s " "cutracer check (info)"
if nsys-ai cutracer check >/dev/null 2>&1; then
  echo "OK (.so installed)"
else
  echo "SKIP (.so not built)"
fi

# ── Mode 1: manifest + field validation ───────────────────────────────────────
echo "== Mode 1 — profile_health_manifest + field validation =="
run_capture "profile_health_manifest" "$MANIFEST_OUT" \
  nsys-ai skill run profile_health_manifest "$PROFILE" --format json
check_regex  "  manifest: gpu field"              "$MANIFEST_OUT" '"gpu"'
check_regex  "  manifest: profile_span_ms"        "$MANIFEST_OUT" '"profile_span_ms"'
check_regex  "  manifest: suspected_bottleneck"   "$MANIFEST_OUT" '"suspected_bottleneck"'
check_json_key "  manifest: nccl.collectives"     "$MANIFEST_OUT" "nccl.collectives"
check_json_key "  manifest: idle.idle_pct"        "$MANIFEST_OUT" "idle.idle_pct"
# overlap.overlap_pct only present when device 0 has kernels; skip if overlap.error exists
printf "  %-50s " "  manifest: overlap.overlap_pct"
OVERLAP_ERR=$(python3 -c "import json; d=json.load(open('$MANIFEST_OUT')); d=d[0] if isinstance(d,list) else d; print('yes' if 'error' in d.get('overlap',{}) else 'no')" 2>/dev/null || echo "no")
if [[ "$OVERLAP_ERR" == "yes" ]]; then
  echo "SKIP (device 0 empty — overlap.error present; auto-retry needed)"
else
  if python3 -c "import json,sys; d=json.load(open('$MANIFEST_OUT')); d=d[0] if isinstance(d,list) else d; assert d['overlap']['overlap_pct'] is not None" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL (key overlap.overlap_pct missing or null)"
    FAIL=$((FAIL+1))
  fi
fi
run "root_cause_matcher"         nsys-ai skill run root_cause_matcher "$PROFILE" --format json
if [[ "$HAS_NVTX" -gt 0 ]]; then
  run "nvtx_layer_breakdown"     nsys-ai skill run nvtx_layer_breakdown "$PROFILE" --format json --max-rows 1
else
  printf "  %-50s " "nvtx_layer_breakdown"; echo "SKIP (no NVTX_EVENTS)"
fi

# ── Mode 2: comms ─────────────────────────────────────────────────────────────
echo "== Mode 2 — comms (NCCL / overlap) =="
run "overlap_breakdown"          nsys-ai skill run overlap_breakdown "$PROFILE" --format json
run "nccl_breakdown"             nsys-ai skill run nccl_breakdown "$PROFILE" --format json
run "kernel_overlap_matrix"      nsys-ai skill run kernel_overlap_matrix "$PROFILE" --format json
run "nccl_anomaly"               nsys-ai skill run nccl_anomaly "$PROFILE" --format json -p threshold=3.0

# ── Mode 3: compute ───────────────────────────────────────────────────────────
echo "== Mode 3 — compute (kernels / tensor core / MFU) =="
run_capture "tensor_core_usage" "$TC_OUT" \
  nsys-ai skill run tensor_core_usage "$PROFILE" --format json
check_regex "  tensor_core_usage: tc_achieved_pct field" "$TC_OUT" '"tc_achieved_pct"'
run "top_kernels"                nsys-ai skill run top_kernels "$PROFILE" --format json --max-rows 5
run "kernel_instances (longest)" nsys-ai skill run kernel_instances "$PROFILE" --format json --max-rows 3
run "kernel_launch_pattern"      nsys-ai skill run kernel_launch_pattern "$PROFILE" --format json
# arithmetic_intensity requires theoretical_flops — use a sentinel value (1e12 FLOPs)
run "arithmetic_intensity"       nsys-ai skill run arithmetic_intensity "$PROFILE" --format json \
  -p theoretical_flops=1000000000000

# ── Mode 4: memory ────────────────────────────────────────────────────────────
echo "== Mode 4 — memory (H2D / D2H / bandwidth) =="
run "memory_bandwidth"           nsys-ai skill run memory_bandwidth "$PROFILE" --format json
run_capture "memory_transfers" "$MEM_XFER_OUT" \
  nsys-ai skill run memory_transfers "$PROFILE" --format json
check_regex "  memory_transfers: total_ms field" "$MEM_XFER_OUT" '"total_ms"'
run_capture "h2d_distribution" "$H2D_OUT" \
  nsys-ai skill run h2d_distribution "$PROFILE" --format json
# h2d_distribution appends pattern metadata only when H2D transfers exist
printf "  %-50s " "  h2d_distribution: pattern type"
H2D_NONEMPTY=$(python3 -c "import json; d=json.load(open('$H2D_OUT')); print('yes' if d else 'no')" 2>/dev/null || echo "no")
if [[ "$H2D_NONEMPTY" == "no" ]]; then
  echo "SKIP (no H2D transfers in this profile)"
else
  if grep -qE '"type"' "$H2D_OUT" 2>/dev/null; then
    echo "OK"
  else
    echo "FAIL (pattern /\"type\"/ not found)"
    FAIL=$((FAIL+1))
  fi
fi

# ── Mode 5: code mapping ──────────────────────────────────────────────────────
echo "== Mode 5 — NVTX / code mapping =="
run_capture "iteration_timing" "$ITER_OUT" \
  nsys-ai skill run iteration_timing "$PROFILE" --format json
if [[ "$HAS_NVTX" -gt 0 ]]; then
  run "nvtx_kernel_map"          nsys-ai skill run nvtx_kernel_map "$PROFILE" --format json --max-rows 5
  # iteration_detail: use iteration=1 (safe default)
  run "iteration_detail iter=1"  nsys-ai skill run iteration_detail "$PROFILE" --format json -p iteration=1
  # speedup_estimator: extract median_ms from iteration_timing output if available
  ITER_MS=$(python3 -c "
import json, sys
rows=[r for r in json.load(open('$ITER_OUT')) if 'duration_ms' in r]
if rows:
    durs=[r['duration_ms'] for r in rows]
    print(sorted(durs)[len(durs)//2])
else:
    print(100)
" 2>/dev/null || echo 100)
  run "speedup_estimator"        nsys-ai skill run speedup_estimator "$PROFILE" --format json \
    -p iteration_ms="$ITER_MS"
else
  printf "  %-50s " "nvtx_kernel_map";        echo "SKIP (no NVTX_EVENTS)"
  printf "  %-50s " "iteration_detail iter=1"; echo "SKIP (no NVTX_EVENTS)"
  printf "  %-50s " "speedup_estimator";       echo "SKIP (no NVTX_EVENTS)"
fi

# ── Mode 6: idle / sync ───────────────────────────────────────────────────────
echo "== Mode 6 — idle / sync =="
run "gpu_idle_gaps"              nsys-ai skill run gpu_idle_gaps "$PROFILE" --format json -p min_gap_ns=1000000
run "stream_concurrency"         nsys-ai skill run stream_concurrency "$PROFILE" --format json
run "sync_cost_analysis"         nsys-ai skill run sync_cost_analysis "$PROFILE" --format json
run "kernel_launch_overhead"     nsys-ai skill run kernel_launch_overhead "$PROFILE" --format json
run "cpu_gpu_pipeline"           nsys-ai skill run cpu_gpu_pipeline "$PROFILE" --format json
run "thread_utilization"         nsys-ai skill run thread_utilization "$PROFILE" --format json
run "module_loading"             nsys-ai skill run module_loading "$PROFILE" --format json
run "gc_impact"                  nsys-ai skill run gc_impact "$PROFILE" --format json
run "pipeline_bubble_metrics"    nsys-ai skill run pipeline_bubble_metrics "$PROFILE" --format json

# ── Stage A evidence / timeline surface ───────────────────────────────────────
echo "== Stage A — evidence build + timeline-web surface =="
run "evidence build"             nsys-ai evidence build "$PROFILE" --format json -o /tmp/findings_smoke.json
run "findings JSON valid"        bash -c "python3 -c 'import json; json.load(open(\"/tmp/findings_smoke.json\"))'"
run "findings has findings key"  bash -c "python3 -c 'import json; assert \"findings\" in json.load(open(\"/tmp/findings_smoke.json\"))'"
run "timeline-web --help"        nsys-ai timeline-web --help

# ── /nsys-ai skill name check ─────────────────────────────────────────────────
echo "== Plugin skill name (/nsys-ai) =="
SKILL_NAME=$(python3 -c "
import re, pathlib
content = pathlib.Path('skills/analyze/SKILL.md').read_text()
m = re.search(r'^name:\s*(\S+)', content, re.MULTILINE)
print(m.group(1) if m else 'NOT_FOUND')
" 2>/dev/null || echo NOT_FOUND)
printf "  %-50s " "SKILL.md name == nsys-ai"
if [[ "$SKILL_NAME" == "nsys-ai" ]]; then
  echo "OK"
else
  echo "FAIL (got: $SKILL_NAME)"
  FAIL=$((FAIL+1))
fi

echo ""
if [[ $FAIL -eq 0 ]]; then
  echo "All checks passed."
  exit 0
else
  echo "$FAIL check(s) FAILED."
  exit 1
fi
