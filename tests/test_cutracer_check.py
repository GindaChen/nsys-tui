"""`cutracer check` must diagnose the environment, not fall over in it.

The command's whole job is to say why instrumentation is not ready. It used to
crash on one of the conditions it exists to report: ``importlib.util.find_spec``
proves a module can be *located*, not that it *imports*, and the ``import`` on
the next line was unguarded. cutracer 0.2.1 and 0.3.0 both import
``importlib_resources`` without declaring it, so on an environment lacking that
backport ``nsys-ai cutracer check`` exited with a ModuleNotFoundError traceback
pointing into handlers.py.

The second case here is a mismatch rather than a failure: the ``cutracer`` extra
is unpinned above 0.2.0, so pip installs whatever is current, while the bundled
installer builds one tag. Nothing compared them, and the .so is what writes the
traces the parser reads.
"""

import sys
import types

import pytest

from nsys_ai.cli import handlers
from nsys_ai.cutracer.installer import CUTRACER_TAG


@pytest.fixture
def fake_cutracer(monkeypatch):
    """Install a stand-in ``cutracer``, importable or not.

    ``find_spec`` consults ``sys.modules`` first and rejects a module with no
    ``__spec__``, so the importable stand-in needs a real one. The unimportable
    case cannot live in ``sys.modules`` at all -- that is the point of it -- so
    it patches the lookup and the import separately, which is exactly the split
    the bug lived in.
    """
    import builtins
    import importlib.machinery
    import importlib.util

    def _install(*, version=None, raises=None):
        if raises is not None:
            monkeypatch.setattr(
                importlib.util,
                "find_spec",
                lambda name, *a, **k: (
                    importlib.machinery.ModuleSpec("cutracer", loader=None)
                    if name == "cutracer"
                    else None
                ),
            )
            real_import = builtins.__import__

            def _fake_import(name, *args, **kwargs):
                if name == "cutracer":
                    raise raises
                return real_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", _fake_import)
            return

        module = types.ModuleType("cutracer")
        module.__spec__ = importlib.machinery.ModuleSpec("cutracer", loader=None)
        if version is not None:
            module.__version__ = version
        monkeypatch.setitem(sys.modules, "cutracer", module)

    return _install


def _run_check(capsys):
    """Run the check, returning (stdout, exited_nonzero)."""
    try:
        handlers._cutracer_check()
    except SystemExit as exit_status:
        return capsys.readouterr().out, bool(exit_status.code)
    return capsys.readouterr().out, False


def test_an_unimportable_package_is_reported_not_raised(fake_cutracer, capsys, monkeypatch):
    """The reported B200 failure: installed, locatable, and its import raises."""
    monkeypatch.setattr(handlers, "_find_cutracer_so", lambda: None)
    fake_cutracer(raises=ModuleNotFoundError("No module named 'importlib_resources'", name="importlib_resources"))

    out, failed = _run_check(capsys)

    assert "FOUND but not importable" in out
    assert "importlib_resources" in out
    # Actionable, not just descriptive.
    assert "pip install importlib_resources" in out
    assert failed, "an unusable cutracer must still be a failing check"


def test_a_missing_package_still_reads_as_missing(fake_cutracer, capsys, monkeypatch):
    """The pre-existing path keeps its wording; absent is not the same as broken."""
    import importlib.util

    monkeypatch.setattr(handlers, "_find_cutracer_so", lambda: None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None)

    out, failed = _run_check(capsys)

    assert "NOT FOUND" in out
    assert "not importable" not in out
    assert failed


def test_a_package_newer_than_the_installer_tag_is_flagged(fake_cutracer, capsys, monkeypatch):
    """pip takes the newest release; the installer builds one tag."""
    monkeypatch.setattr(handlers, "_find_cutracer_so", lambda: None)
    fake_cutracer(version="0.9.9")

    out, _ = _run_check(capsys)

    assert "version alignment" in out
    assert "0.9.9" in out and CUTRACER_TAG in out


def test_a_matching_version_says_nothing(fake_cutracer, capsys, monkeypatch):
    """The warning is for a real mismatch, not noise on every run."""
    monkeypatch.setattr(handlers, "_find_cutracer_so", lambda: None)
    fake_cutracer(version=CUTRACER_TAG.lstrip("v"))

    out, _ = _run_check(capsys)

    assert "version alignment" not in out
    assert f"OK (v{CUTRACER_TAG.lstrip('v')})" in out


def test_an_unknown_version_does_not_claim_a_mismatch(fake_cutracer, capsys, monkeypatch):
    """A package without __version__ is unknown, which is not evidence of drift."""
    monkeypatch.setattr(handlers, "_find_cutracer_so", lambda: None)
    fake_cutracer(version=None)

    out, _ = _run_check(capsys)

    assert "OK (vunknown)" in out
    assert "version alignment" not in out
