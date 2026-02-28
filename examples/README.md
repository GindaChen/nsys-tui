# 🔬 nsys-ai Examples

Hands-on examples for getting started with nsys-ai. Each example includes a download script and a step-by-step quick start.

> **Low numbers = simpler.** Numbers 01–09 are reserved for introductory examples.

## Examples

| # | Example | Profile | GPUs | Difficulty |
|---|---------|---------|------|------------|
| 10 | [FastVideo Inference](example-10-fastvideo-inference/) | Inference (video generation) | 4× H100 | ⭐⭐ Intermediate |
| 20 | [Megatron-LM DistCA](example-20-megatron-distca/) | Training (Transformer Engine) | 8× H200 | ⭐⭐⭐ Advanced |

## Quick Start (Example 10)

```bash
pip install nsys-ai
cd example-10-fastvideo-inference
python download_data.py
nsys-ai timeline output/fastvideo-wan21-1.3b-4gpu.sqlite --gpu 0
```

## Adding New Examples

Follow the convention:

```
example-NN-short-description/
├── .gitignore         # Ignores output/ directory
├── README.md          # Step-by-step guide
├── download_data.py   # Data download script
└── output/            # Downloaded data (gitignored)
```

Reserve number ranges:
- `01–09` — Introductory / synthetic profiles
- `10–19` — Single-model inference profiling
- `20–29` — Distributed training profiling
