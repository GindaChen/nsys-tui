#!/usr/bin/env python3
"""Regenerate tests/fixtures/mock.sqlite (the hermetic CI fixture).

Requirements (for a real build, not --dry-run):
  - `nsys` binary on PATH (NVIDIA Nsight Systems)
  - CUDA-capable GPU accessible (`nvidia-smi` works)
  - PyTorch with CUDA support installed in the **same interpreter** this script
    runs with — the inner capture is `python -c "import torch; torch.zeros(16, device='cuda') ..."`.
    On a multi-venv system run via the CUDA venv, e.g. `.venv-gpu/bin/python scripts/build_fixture.py`.

Produces a minimal profile (a couple of tiny kernels, no NCCL, no NVTX) — just
enough to exercise nsys-ai CLI wiring in CI without depending on user-local
~100 MB profiles. The exact kernel count depends on torch/CUDA version but has
historically been 1–3 kernels for `torch.zeros(16, device='cuda').sum().item()`.

CI (`.github/workflows/plugin-smoke.yml`) loads the committed fixture; this
script is run manually on a CUDA host when the nsys schema changes.

Usage:
    python scripts/build_fixture.py              # actually build (needs GPU + torch)
    python scripts/build_fixture.py --dry-run    # print commands only (no CUDA needed)
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures"
FIXTURE = FIXTURE_DIR / "mock.sqlite"
REP = FIXTURE_DIR / "mock.nsys-rep"

SCRIPT = "import torch; x = torch.zeros(16, device='cuda'); x.sum().item()"


def build_commands() -> list[list[str]]:
    # Use the current interpreter so CUDA-enabled venvs (e.g. .venv-gpu) carry through.
    # Minimal-flag capture keeps the fixture small (target: a few hundred KB):
    #   --sample=none       skip CPU IP sampling / symbol downloads
    #   --cpuctxsw=none     skip context-switch traces
    #   --trace=cuda        CUDA only; no OS, NVTX, or OpenGL tracing
    return [
        ["nsys", "profile",
         "-o", str(REP.with_suffix("")),
         "--force-overwrite=true",
         "--sample=none",
         "--cpuctxsw=none",
         "--trace=cuda",
         sys.executable, "-c", SCRIPT],
        ["nsys", "export", "--type=sqlite", "--force-overwrite=true",
         f"--output={FIXTURE}", str(REP)],
    ]


def preflight() -> None:
    """Fail fast with a clear message if prerequisites are missing.

    Only runs in real-build mode. --dry-run skips preflight so CI can
    syntax-check the script on hosts without CUDA or torch.
    """
    import shutil
    if shutil.which("nsys") is None:
        sys.exit("error: `nsys` not on PATH — install NVIDIA Nsight Systems "
                 "(https://developer.nvidia.com/nsight-systems) and retry.")
    try:
        import torch  # noqa: F401 — probe only
    except ImportError:
        sys.exit("error: torch not installed in this interpreter "
                 f"({sys.executable}). The inner capture runs `import torch`. "
                 "Install torch with CUDA support (e.g. `pip install torch`) "
                 "or invoke this script via a CUDA-enabled venv, "
                 "e.g. `.venv-gpu/bin/python scripts/build_fixture.py`.")
    if not torch.cuda.is_available():
        sys.exit("error: torch installed but CUDA not available. Ensure the "
                 "interpreter was built with CUDA and a GPU is visible "
                 "(check `nvidia-smi`).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands without executing")
    args = parser.parse_args()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    cmds = build_commands()

    if args.dry_run:
        print(f"# Would write fixture to {FIXTURE}")
        for cmd in cmds:
            print(shlex.join(cmd))
        return 0

    preflight()

    for cmd in cmds:
        print(f"$ {shlex.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"command failed with exit {result.returncode}", file=sys.stderr)
            return result.returncode

    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
