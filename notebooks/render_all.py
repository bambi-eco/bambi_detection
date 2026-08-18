# -*- coding: utf-8 -*-
"""Execute every notebook and store the result under ``notebooks/rendered/``.

The notebooks themselves are committed without outputs (a test enforces it,
so diffs stay reviewable and CI proves they still run). The rendered copies
are what GitHub shows with plots and printed numbers - "frozen" results of
the public flights. Re-run this after changing a notebook::

    python notebooks/render_all.py            # all
    python notebooks/render_all.py 02 04      # by prefix
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import nbclient
import nbformat

HERE = Path(__file__).resolve().parent
OUT = HERE / "rendered"


def render(path: Path) -> Path:
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=3600, kernel_name="python3",
                                     resources={"metadata": {"path": str(HERE)}})
    client.execute()
    OUT.mkdir(exist_ok=True)
    target = OUT / path.name
    nbformat.write(nb, target)
    return target


def main(argv=None) -> int:
    os.environ.pop("MPLBACKEND", None)          # let ipykernel use its inline backend, so figures land in the file
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    prefixes = list(argv if argv is not None else sys.argv[1:])
    notebooks = sorted(p for p in HERE.glob("*.ipynb") if not p.name.startswith("_"))
    if prefixes:
        notebooks = [p for p in notebooks if any(p.name.startswith(x) for x in prefixes)]
    for nb in notebooks:
        print(f"rendering {nb.name} ...", flush=True)
        out = render(nb)
        print(f"  -> {out.relative_to(HERE.parent)} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
