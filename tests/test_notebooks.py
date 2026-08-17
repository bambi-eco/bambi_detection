# -*- coding: utf-8 -*-
"""Execute every notebook headless.

The notebooks are the acceptance test for each capability family, so a
notebook that stops running is a red build - not a stale document nobody
opens. Marked ``notebook`` (and implicitly slow: the first run downloads a
public flight into ``data_dir``); the scheduled CI job runs them against a
cached flight.
"""
import os
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS = sorted(p for p in NOTEBOOKS_DIR.glob("*.ipynb") if not p.name.startswith("_"))


@pytest.mark.notebook
@pytest.mark.slow
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_executes(notebook, data_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("BAMBI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    nb = nbformat.read(notebook, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=1800, kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOKS_DIR)}},
    )
    client.execute()
    # Keep the executed copy as an artefact for inspection (not committed).
    out = tmp_path / notebook.name
    nbformat.write(nb, out)
    errors = [o for c in nb.cells if c.cell_type == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, errors[0].get("evalue")


def test_every_notebook_starts_from_the_shared_setup():
    """New notebooks must go through ``_setup`` so data access stays in one place."""
    missing = [p.name for p in NOTEBOOKS
               if "from _setup import" not in nbformat.read(p, as_version=4).cells[2].source
               and "from _setup import" not in "".join(c.source for c in nbformat.read(p, as_version=4).cells if c.cell_type == "code")]
    assert not missing, f"notebooks not using notebooks/_setup.py: {missing}"


def test_notebooks_are_committed_without_outputs():
    """Outputs are produced by CI, not committed; keeps diffs reviewable."""
    dirty = []
    for p in NOTEBOOKS:
        nb = nbformat.read(p, as_version=4)
        if any(c.get("outputs") for c in nb.cells if c.cell_type == "code"):
            dirty.append(p.name)
    assert not dirty, f"strip outputs before committing: {dirty}"
