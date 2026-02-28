"""Integration smoke tests: require a real .sqlite profile (e.g. data/...)."""
import os
import subprocess
import sys
import tempfile

import pytest

PROFILE_ENV = "NSYS_TEST_PROFILE"
DEFAULT_PROFILE = "data/nsys-hero/distca-0/baseline.t128k.host-fs-mbz-gpu-899"


def _profile_path():
    path = os.environ.get(PROFILE_ENV, DEFAULT_PROFILE)
    return path if os.path.exists(path) else None


@pytest.mark.skipif(_profile_path() is None, reason="No test profile")
def test_info():
    path = _profile_path()
    r = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "info", path],
        capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert "GPU" in r.stdout or "Kernels" in r.stdout


@pytest.mark.skipif(_profile_path() is None, reason="No test profile")
def test_summary():
    path = _profile_path()
    r = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "summary", path],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0


@pytest.mark.skipif(_profile_path() is None, reason="No test profile")
def test_analyze():
    path = _profile_path()
    r = subprocess.run(
        [sys.executable, "-m", "nsys_tui", "analyze", path, "--gpu", "4", "--trim", "39", "42"],
        capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert "Span:" in r.stdout or "Kernels:" in r.stdout


@pytest.mark.skipif(_profile_path() is None, reason="No test profile")
def test_analyze_markdown_output():
    path = _profile_path()
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        out = f.name
    try:
        r = subprocess.run(
            [sys.executable, "-m", "nsys_tui", "analyze", path, "--gpu", "4", "--trim", "39", "42", "-o", out],
            capture_output=True, text=True, timeout=30)
        assert r.returncode == 0
        assert os.path.getsize(out) > 100
    finally:
        os.unlink(out)


@pytest.mark.skipif(_profile_path() is None, reason="No test profile")
def test_export_perfetto_json():
    path = _profile_path()
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(
            [sys.executable, "-m", "nsys_tui", "export", path, "--gpu", "4", "--trim", "39", "42", "-o", d],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0
        assert any(f.startswith("trace_gpu") and f.endswith(".json") for f in os.listdir(d))


@pytest.mark.skipif(_profile_path() is None, reason="No test profile")
def test_export_csv():
    path = _profile_path()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out = f.name
    try:
        r = subprocess.run(
            [sys.executable, "-m", "nsys_tui", "export-csv", path, "--gpu", "4", "--trim", "39", "42", "-o", out],
            capture_output=True, text=True, timeout=30)
        assert r.returncode == 0
        assert os.path.getsize(out) > 50
    finally:
        os.unlink(out)
