# 🗺️ nsys-ai Roadmap

Two pillars: **UI** (making profiles effortless to view) and **AI** (making profiles effortless to understand).

---

## Priority Order

### 🔴 P0 — Critical (next sprint)

| # | Item | Pillar |
|---|------|--------|
| [#1](../../issues/1) | **`nsys-ai analyze`** — full auto-report from a profile | AI |
| [#2](../../issues/2) | **One-click Perfetto** — server → local transport, zero friction | UI |

### 🟠 P1 — High (near term)

| # | Item | Pillar |
|---|------|--------|
| [#3](../../issues/3) | `nsys-ai ask` — natural language queries on profiles | AI |
| [#4](../../issues/4) | `nsys-ai diff` — AI-narrated profile comparison | AI |
| [#5](../../issues/5) | TUI inline AI — press `?` to explain any kernel | AI+UI |
| [#6](../../issues/6) | Web UI chat widget — ask questions in the browser | AI+UI |

### 🟡 P2 — Medium

| # | Item | Pillar |
|---|------|--------|
| [#7](../../issues/7) | Custom web flame chart with NVTX-aware hierarchy | UI |
| [#8](../../issues/8) | Multi-model AI backend + caching layer | AI |
| [#9](../../issues/9) | TUI polish — multi-GPU stacked view, diff mode | UI |
| [#10](../../issues/10) | `nsys-ai suggest` — NVTX annotation suggestions | AI |

### 🟣 P3 — Nice to have (longer term)

| # | Item | Pillar |
|---|------|--------|
| [#11](../../issues/11) | VS Code extension — open `.sqlite` → launch viewer | UI |
| [#12](../../issues/12) | Jupyter widget for inline profile viewing | UI |
| [#13](../../issues/13) | CI integration — `nsys-ai check` for perf regression gating | AI |
| [#14](../../issues/14) | Anomaly detection across training iterations | AI |

---

## 🖥️ Pillar 1 — UI

> Zero-friction viewing of Nsight profiles across every surface — terminal, browser, VS Code.

**One-Click Perfetto (Server → Local)** — VSCode transport: remote SSH profile → local Perfetto in one click. Auto-detect `.sqlite` / `.nsys-rep`, convert + stream. Single command: `nsys-ai open profile.sqlite`.

**TUI** — Timeline polish (bookmarks, annotation overlay, multi-GPU stacked view). Tree improvements (sparklines, diff mode). Unified launcher that auto-selects timeline vs tree.

**Web UI** — Self-hosted viewer richer than Perfetto. NVTX-aware flame chart, side-by-side comparison, shareable links.

**Packaging** — VS Code extension stub, Jupyter widget, zero-config pip install.

---

## 🤖 Pillar 2 — AI

> AI that understands GPU profiles as a first-class concept — integrated everywhere, not bolted on.

**AI in every interface** — TUI: inline commentary panel. Web: chat widget. CLI: `nsys-ai ask "why is iteration 142 slow?"`.

**AI CLI** — `analyze` (auto-report), `diff` (narrated comparison), `suggest` (NVTX annotations), `explain` (kernel deep-dive).

**Backend** — Profile-aware RAG, multi-model support (Claude/GPT/Ollama), cost-gated, caching.

**Automation** — Iteration regression detection, anomaly flagging, CI pass/fail gating.

---

## 🧠 Pillar 3 — Agent & Skills

> An intelligent agent that uses standardized SQL skills to diagnose GPU performance problems from first principles.

**Skills Foundation** — 8 built-in SQL skills: `top_kernels`, `memory_transfers`, `nvtx_kernel_map`, `gpu_idle_gaps`, `nccl_breakdown`, `kernel_launch_overhead`, `thread_utilization`, `schema_inspect`. User-extensible skill registry.

**Agent Persona** — CUDA ML systems expert with deep knowledge of nsys, Megatron, SGLang, vLLM. Follows evidence-based analysis: orient → identify → hypothesize → investigate → diagnose → recommend → verify.

**Book of Root Causes** — Living document of GPU performance problems. Quick-reference table, 10 detailed root cause writeups, and 38 veteran diagnostic questions.

**Benchmarking Problems** (planned):
1. Identifying pipeline "bubbles" and stalls
2. Calculating Model Flops Utilization (MFU)
3. Determining if kernels achieve ideal arithmetic intensity for a given GPU
4. Analyzing network overlap and bandwidth vs. compute-communication balance
5. Investigating module loading or kernel compilation elongating forward/backward passes
6. Assessing how memory/GC affects performance

---

## ✅ Shipped

- [x] Timeline TUI (v0.1.0)
- [x] Tree TUI (v0.1.0)
- [x] HTML viewer export (v0.1.0)
- [x] Perfetto JSON export + `perfetto` command (v0.1.5)
- [x] Web UI server — `nsys-ai web` (v0.2.0)
- [x] AI module — auto-commentary, NVTX suggestions, bottleneck detection (v0.1.0)
- [x] PyPI package as `nsys-ai` (v0.2.1)
- [x] Agent skill system — 8 built-in SQL skills + registry + CLI (v0.3.0)
- [x] Agent persona + analysis loop — `nsys-ai agent analyze|ask` (v0.3.0)
- [x] Book of Root Causes — quick-ref, long-form, veteran questions (v0.3.0)
- [x] Modular packaging — `[agent]`, `[all]` extras (v0.3.0)
