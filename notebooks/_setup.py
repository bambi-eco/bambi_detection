# -*- coding: utf-8 -*-
"""Shared setup for the bambi-detection notebooks.

Every notebook starts with::

    from _setup import ensure_environment, get_flight
    ensure_environment()
    flight = get_flight("146")

``ensure_environment`` installs bambi-detection (editable when run from a
checkout, from the pinned tag otherwise) plus an alfspy backend, and clones the
public Dataset repository whose scripts do all the downloading. ``get_flight``
fetches one flight through ``download_from_zenodo.py`` into a cache and returns
its folder, so a re-run costs nothing.

The cache is ``BAMBI_DATA_DIR`` if set, else ``<repo>/.test-data`` - the same
folder the test suite's ``data_dir`` fixture uses, so CI's cache serves both.
On Colab everything lands under ``/content``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_REPO = "https://github.com/bambi-eco/Dataset.git"
BAMBI_TAG = "v1.0.0"                 # bumped by the release process
ALFS_TORCH_TAG = "v1.1.1"

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IN_COLAB = "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG") is not None
WORK = Path("/content") if IN_COLAB else REPO
DATA_DIR = Path(os.environ.get("BAMBI_DATA_DIR", WORK / ".test-data"))
DATASET_DIR = WORK / "Dataset"


def _run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    # The Dataset scripts print emoji progress markers; force UTF-8 so they do
    # not crash under a cp1252 console (Windows kernels) - Linux CI is UTF-8 anyway.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run([str(c) for c in cmd], check=True, env=env, **kw)


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def ensure_environment(backend: str = "torch") -> None:
    """Install bambi-detection + an alfspy backend, clone the Dataset tooling.

    :param backend: ``"torch"`` (default, needs no OpenGL) or ``"moderngl"``
    """
    pip = [sys.executable, "-m", "pip", "install", "-q"]

    if not _have("bambi"):
        if (REPO / "pyproject.toml").exists() and not IN_COLAB:
            _run(pip + ["-e", str(REPO)])
        else:
            _run(pip + [f"git+https://github.com/bambi-eco/bambi_detection.git@{BAMBI_TAG}"])

    if not _have("alfspy"):
        if backend == "torch":
            _run(pip + ["torch", "torchvision", "--index-url",
                        "https://download.pytorch.org/whl/cpu"])
            _run(pip + [f"git+https://github.com/bambi-eco/alfs_pytorch.git@{ALFS_TORCH_TAG}"])
        else:
            _run(pip + ["git+https://github.com/bambi-eco/alfs_py.git@v2.1.0", "moderngl"])

    if not DATASET_DIR.exists():
        _run(["git", "clone", "--depth", "1", DATASET_REPO, str(DATASET_DIR)])
    for extra in ("requests", "tqdm"):
        if not _have(extra):
            _run(pip + [extra])

    print(f"\nbambi-detection ready. data dir: {DATA_DIR}")


def get_flight(flight_id: str, version: str = "base") -> Path:
    """Download one public flight through the Dataset repo tooling and cache it.

    :param flight_id: flight prefix as used by ``download_from_zenodo.py -f``
    :param version: ``base`` (poses + MOT + processed video), ``raw`` (original
        DJI MP4 + SRT + AirData), ``matched`` or ``orthographic``
    :return: folder holding that flight's files
    """
    target = DATA_DIR / version / str(flight_id)
    if target.exists() and any(target.iterdir()):
        print(f"flight {flight_id} ({version}) cached at {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    script = DATASET_DIR / "download_from_zenodo.py"
    if not script.exists():
        raise RuntimeError("Dataset tooling missing - call ensure_environment() first")
    _run([sys.executable, str(script), "--version", version, "-f", str(flight_id),
          "--unzip", "-o", str(target)], cwd=str(DATASET_DIR))
    return target


def find(folder: Path, *patterns: str) -> list[Path]:
    """Files under *folder* matching any glob pattern, sorted."""
    out: set[Path] = set()
    for pat in patterns:
        out.update(folder.rglob(pat))
    return sorted(out)


def get_dem(flight_id: str, version: str = "base") -> Path:
    """The flight's DEM mesh (``DEM/<id>_matched_dem.glb`` + ``.json``), made if missing.

    Three sources, cheapest first:

    1. already in the flight folder (a previous run);
    2. a GeoTIFF clip shipped with the repo under ``tests/fixtures/dem`` -
       ~0.4 MB, meshed on the spot with :func:`bambi.io.dem.geotiff_to_dem`
       (this is what CI uses);
    3. the Dataset repo's ``dem_from_poses.py``, which downloads the BEV
       1 m ALS tile covering the flight (~10 GB for one tile - do it once).
    """
    folder = get_flight(flight_id, version) / "DEM"
    glb = folder / f"{flight_id}_matched_dem.glb"
    if glb.exists() and glb.with_suffix(".json").exists():
        return glb
    fixture = REPO / "tests" / "fixtures" / "dem" / f"{flight_id}_matched_dem.tif"
    if fixture.exists():
        from bambi.io.dem import geotiff_to_dem
        folder.mkdir(parents=True, exist_ok=True)
        print(f"meshing bundled DEM clip {fixture.name} -> {glb}")
        geotiff_to_dem(fixture, glb, simplify=2)
        return glb
    poses = next(get_flight(flight_id, version).glob("*_poses.json"))
    _run([sys.executable, str(DATASET_DIR / "dem_from_poses.py"), "--file", str(poses),
          "--output-dir", str(folder)], cwd=str(DATASET_DIR))
    return glb


def clean_cache() -> None:
    """Delete the flight cache (use sparingly - a flight is hundreds of MB)."""
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
