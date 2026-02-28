<div align="center">

# 🔬 nsys-ai

**AI-powered analysis for NVIDIA Nsight Systems profiles**

Navigate GPU kernel timelines, diagnose performance bottlenecks with AI, and explore NVTX hierarchies — all from your terminal.

> **Mission:** Build an intelligent agent that truly understands GPU performance from first principles. An agent that can identify pipeline bubbles, calculate MFU, assess arithmetic intensity, and diagnose the root causes that cost millions of dollars in GPU hours — turning months of expert debugging into minutes.

[![CI](https://github.com/GindaChen/nsys-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/GindaChen/nsys-ai/actions)
[![PyPI](https://img.shields.io/pypi/v/nsys-ai)](https://pypi.org/project/nsys-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## ⚡ Install

```bash
pip install nsys-ai
```

That's it. No system dependencies, no CUDA required. Just Python 3.10+.

---

## 🎯 What It Does

nsys-ai reads `.sqlite` profile exports from [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) and gives you **three ways** to explore them:

<table>
<tr>
<td width="33%" align="center">

### 🖥️ Timeline TUI
Perfetto-style horizontal timeline in your terminal

</td>
<td width="33%" align="center">

### 🌲 Tree TUI
Interactive NVTX hierarchy browser with kernel details

</td>
<td width="33%" align="center">

### 🌐 HTML Viewer
Exportable interactive visualizations for sharing

</td>
</tr>
<tr>
<td>

```
S21 ████░██████░███
S56 ██████░░░███████
S60 ░░░██████░░░░░██
    |         │
    39.1s   39.5s
```

</td>
<td>

```
▼ Iteration (324ms)
  ▼ forward (180ms)
    ▼ Attention (89ms)
      ■ flash_fwd  26ms
      ■ flash_bwd  63ms
```

</td>
<td>

Interactive HTML exports:<br>
• NVTX stack viewer<br>
• SQLite schema explorer<br>
• Perfetto JSON traces

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 1. Get a profile

```bash
# Option A: Profile your own PyTorch training
nsys profile -o my_training python train.py
# → produces my_training.sqlite

# Option B: Download an example profile
cd examples/example-20-megatron-distca
python download_data.py
# → downloads output/megatron_distca.sqlite
```

### 2. Explore it

```bash
# Quick overview
nsys-ai info my_training.sqlite

# Interactive timeline (the main attraction)
nsys-ai timeline my_training.sqlite --gpu 0 --trim 39 42

# Interactive tree browser
nsys-ai tui my_training.sqlite --gpu 0 --trim 39 42

# GPU kernel summary
nsys-ai summary my_training.sqlite --gpu 0
```

### 3. Export & share

```bash
# Perfetto JSON (open in ui.perfetto.dev)
nsys-ai export my_training.sqlite -o traces/

# Interactive HTML viewer
nsys-ai viewer my_training.sqlite --gpu 0 --trim 39 42 -o report.html

# Flat CSV/JSON for scripting
nsys-ai export-csv my_training.sqlite --gpu 0 --trim 39 42 -o kernels.csv
```

---

## ⌨️ Timeline TUI

The timeline is a **Perfetto-style** horizontal viewer with per-stream kernel visualization, NVTX hierarchy bars, and a time-cursor navigation model.

### Navigation

| Key | Action |
|:---:|--------|
| `←` `→` | Pan through time |
| `Shift+←/→` | Page pan (1/4 viewport) |
| `↑` `↓` | Select stream |
| `Tab` | Snap to next kernel |
| `+` `-` | Zoom in / out |
| `a` | Toggle absolute ↔ relative time |

### Analysis

| Key | Action |
|:---:|--------|
| `/` | Filter kernels by name |
| `m` | Set minimum duration threshold |
| `d` | Toggle demangled kernel names |
| `C` | Open config panel |
| `h` | Full help overlay |

### Bookmarks

| Key | Action |
|:---:|--------|
| `B` | Save bookmark (with kernel + NVTX context) |
| `'` | Bookmark list — press 1-9 to jump |
| `,` `.` | Cycle through bookmarks |
| `` ` `` | Jump back to previous position |
| `[` `]` | Set range start / end |

### Config Panel (`C`)

Tweak settings live with ↑/↓ to select and ←/→ to adjust:

- Selected stream rows (1-6)
- Other stream rows (1-4)
- Time tick density (2-20)
- NVTX depth levels (0-8)
- Min kernel duration filter

---

## 📚 Documentation

The `docs/` directory includes comprehensive guides for Nsight Systems profiling:

| Guide | Topic |
|-------|-------|
| [CLI Reference](docs/01-cli-reference.md) | Full `nsys` command reference |
| [SQLite Schema](docs/02-sqlite-schema.md) | Database tables & relationships |
| [NVTX Annotations](docs/03-nvtx-annotations.md) | Adding markers to your code |
| [CUDA Trace](docs/04-cuda-trace.md) | GPU kernel tracing |
| [NCCL Tracing](docs/05-nccl-tracing.md) | Multi-GPU collective analysis |
| [Python/PyTorch](docs/06-python-pytorch.md) | Profiling PyTorch workloads |
| [Containers](docs/07-container-profiling.md) | Profiling inside Docker/Slurm |
| [Focused Profiling](docs/08-focused-profiling.md) | Targeted profiling strategies |

### 🔍 Interactive SQLite Schema Explorer

The [`docs/sqlite-explorer/`](docs/sqlite-explorer/) contains an **interactive HTML tool** for exploring the Nsight SQLite schema — tables, foreign keys, example queries, and key concepts. Open `docs/sqlite-explorer/index.html` in a browser:

- Browse all Nsight SQLite tables with column types
- See foreign key relationships visualized
- Copy-paste ready SQL query examples
- Cross-highlighted concept explanations

---

## 🛠️ All Commands

| Command | Description |
|---------|-------------|
| `info` | Profile metadata & GPU hardware |
| `summary` | Top kernels, stream breakdown, auto-commentary |
| `overlap` | Compute / NCCL overlap analysis |
| `nccl` | NCCL collective breakdown by type |
| `iters` | Auto-detect training iterations |
| `tree` | NVTX hierarchy as text |
| `tui` | **Interactive tree TUI** |
| `timeline` | **Interactive timeline TUI** |
| `search` | Search kernels / NVTX by name |
| `export` | Perfetto JSON traces |
| `export-csv` | Flat CSV for spreadsheets |
| `export-json` | Flat JSON for scripting |
| `viewer` | Interactive HTML report |
| `markdown` | NVTX hierarchy as markdown |

---

## 🧩 Skills (Analysis Building Blocks)

nsys-ai ships with 8 built-in SQL skills — self-contained analysis units that work without any LLM:

```bash
# List all available skills
nsys-ai skill list

# Run a specific skill
nsys-ai skill run top_kernels profile.sqlite
nsys-ai skill run gpu_idle_gaps profile.sqlite
nsys-ai skill run nccl_breakdown profile.sqlite
```

| Skill | What it does |
|-------|-------------|
| `top_kernels` | Heaviest GPU kernels by total time |
| `memory_transfers` | H2D/D2H/D2D transfer breakdown |
| `nvtx_kernel_map` | NVTX annotation → kernel mapping |
| `gpu_idle_gaps` | Pipeline bubbles between kernels |
| `nccl_breakdown` | NCCL collective operation summary |
| `kernel_launch_overhead` | CPU→GPU dispatch latency |
| `thread_utilization` | CPU thread bottleneck detection |
| `schema_inspect` | Database tables and columns |

Skills are extensible — add your own by creating a Python file that exports a `SKILL` constant.

---

## 🤖 AI Agent

The agent is a CUDA ML systems expert that runs skills automatically and diagnoses problems:

```bash
# Full auto-analysis
nsys-ai agent analyze profile.sqlite

# Ask a specific question
nsys-ai agent ask profile.sqlite "why are there bubbles in the pipeline?"
nsys-ai agent ask profile.sqlite "is NCCL overlapping with compute?"
```

With `pip install nsys-ai[agent]`, the agent can use an LLM to synthesize natural language analysis from skill results.

---

## 📦 Install Tiers

```bash
pip install nsys-ai          # Core: CLI + TUI + skills (no dependencies!)
pip install nsys-ai[agent]   # + LLM-backed agent analysis (requires anthropic)
pip install nsys-ai[all]     # Everything
```

---

## 🤖 AI Analysis (Optional)

nsys-ai includes an optional AI module that can analyze your profiles:

```bash
pip install nsys-ai[ai]
```

- **Auto-commentary** on kernel distributions and performance patterns
- **NVTX annotation suggestions** for un-annotated code regions
- **Performance bottleneck detection** with actionable recommendations

---

## 🧑‍💻 Development

```bash
git clone https://github.com/GindaChen/nsys-ai.git
cd nsys-ai
pip install -e '.[dev]'
pytest tests/ -v
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

<div align="center">
<sub>Built for GPU performance engineers who live in the terminal.</sub>
</div>
