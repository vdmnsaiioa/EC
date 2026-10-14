"""Loading libqaqua.so (sseft.qaqua.load_library) where the compiler's Fortran runtime is not on the loader's path:
the runtime build.sh bundles in <build>/runtime, the order of its libraries, the error that says how to repair a
build.  ctypes.CDLL is replaced by a fake loader, so no q-AQUA build is needed."""
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sseft import qaqua as Q

REPO = Path(__file__).resolve().parents[1]


class FakeLoader:
    """ctypes.CDLL for libraries that load only once the libraries they need are loaded (by basename)."""

    def __init__(self, needs):
        self.needs, self.loaded = needs, []

    def __call__(self, path, *args, **kwargs):
        name = os.path.basename(path)
        missing = [d for d in self.needs.get(name, []) if d not in self.loaded]
        if missing:
            raise OSError(f"{missing[0]}: cannot open shared object file: No such file or directory")
        self.loaded.append(name)
        return SimpleNamespace(name=name)


def _build(tmp_path, runtime=()):
    (tmp_path / "libqaqua.so").write_bytes(b"")
    if runtime:
        (tmp_path / Q.RUNTIME_DIR).mkdir()
        for f in runtime:
            (tmp_path / Q.RUNTIME_DIR / f).write_bytes(b"")
    return str(tmp_path)


@pytest.fixture
def loader(monkeypatch):
    def install(needs):
        fake = FakeLoader(needs)
        monkeypatch.setattr(Q.ctypes, "CDLL", fake)
        monkeypatch.setattr(Q, "_preloaded", [])
        return fake
    return install


def test_a_library_that_loads_is_loaded_alone(tmp_path, loader):
    root = _build(tmp_path, runtime=["libgfortran.so.5"])
    fake = loader({})
    assert Q.load_library(root).name == "libqaqua.so" and fake.loaded == ["libqaqua.so"]


def test_the_bundled_runtime_is_preloaded_in_dependency_order(tmp_path, loader):
    """an older GCC's libgfortran needs libquadmath, which sorts after it: the second pass loads it."""
    root = _build(tmp_path, runtime=["libgfortran.so.5", "libquadmath.so.0"])
    fake = loader({"libqaqua.so": ["libgfortran.so.5"], "libgfortran.so.5": ["libquadmath.so.0"]})
    assert Q.load_library(root).name == "libqaqua.so"
    assert fake.loaded == ["libquadmath.so.0", "libgfortran.so.5", "libqaqua.so"]
    assert len(Q._preloaded) == 2


def test_a_build_without_its_runtime_says_how_to_repair_it(tmp_path, loader):
    root = _build(tmp_path)
    loader({"libqaqua.so": ["libgfortran.so.5"]})
    with pytest.raises(OSError) as err:
        Q.load_library(root)
    msg = str(err.value)
    assert "libgfortran.so.5: cannot open shared object file" in msg and "build.sh --relink" in msg
    assert isinstance(err.value.__cause__, OSError)


def test_runtime_names():
    for name in ("libgfortran.so.5", "libgfortran.so.5.0.0", "libquadmath.so.0", "libifcore.so.5", "libimf.so"):
        assert Q._RUNTIME_NAME.match(name), name
    for name in ("libgfortran-83c28eba-468e71e5.so.5.0.0", "libqaqua.so", "libm.so.6", "libgcc_s.so.1"):
        assert not Q._RUNTIME_NAME.match(name), name          # a wheel's renamed copy is not what libqaqua.so uses
    assert isinstance(Q.loaded_runtime(), list)


@pytest.mark.skipif(shutil.which("sh") is None, reason="no sh")
def test_build_script_parses():
    subprocess.run(["sh", "-n", str(REPO / "external" / "qaqua" / "build.sh")], check=True)
