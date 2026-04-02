# 📖 Book of Root Causes — Quick Reference

> Performance problems that cost millions of dollars in GPU hours, distilled into a single table.

| # | Root Cause | Symptom | Detection Skill | Severity |
|---|-----------|---------|----------------|----------|
| 1 | **GPU Bubbles** | Idle gaps between kernels on a stream | `gpu_idle_gaps` | 🔴 High |
| 2 | **CPU Bottleneck** | Low GPU utilization despite available work | `thread_utilization` | 🔴 High |
| 3 | **NCCL Serialization** | AllReduce not overlapping with compute | `nccl_breakdown` | 🔴 High |
| 4 | **Excessive H2D Transfers** | Large memory copies in the critical path | `memory_transfers` | 🟠 Medium |
| 5 | **Small Kernel Overhead** | Thousands of tiny kernels with high launch cost | `kernel_launch_overhead` | 🟠 Medium |
| 6 | **Kernel Hotspot** | Single kernel dominates >50% of total time | `top_kernels` | 🔴 High |
| 7 | **Missing NVTX** | Cannot attribute kernels to source code | `nvtx_kernel_map` | 🟡 Low |
| 8 | **GC Pauses** | Python garbage collection stalls GPU pipeline | `gpu_idle_gaps` | 🟠 Medium |
| 9 | **Module Loading** | Import/JIT compilation during forward pass | `gpu_idle_gaps` | 🟡 Low |
| 10 | **Compute-Comm Imbalance** | Some ranks finish early, wait at barrier | `nccl_breakdown` | 🔴 High |
| 11 | **Stream Serialization** | Streams that should overlap but run sequentially | `gpu_idle_gaps` | 🟠 Medium |
| 12 | **Excessive Synchronization** | `cudaDeviceSynchronize` in the loop | `kernel_launch_overhead` | 🟠 Medium |
| 13 | **FP32 Fallback** | Tensor Core eligible kernels falling back to FP32/SIMT | `tensor_core_usage` | 🔴 High |

---

## How to Use This

1. **Run `nsys-ai agent analyze <profile>`** — the agent checks for all of these automatically
2. **Check the top hits** — focus on 🔴 High severity items first
3. **Drill down** — run the specific detection skill for more detail
4. **Read the [full writeup](book.md)** for remediation guidance

See also: [veteran-questions.md](veteran-questions.md) — diagnostic questions a performance expert would ask.
