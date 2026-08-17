# -*- coding: utf-8 -*-
"""Every console script must build its parser and answer ``--help`` cleanly.

This is the contract for the file-based edge: a CLI may read paths *after*
parsing, never before, and never with a hard-coded default that points at a
private drive. Two real bugs motivated the test - a leftover debug block that
overwrote ``sys.argv`` before ``parse_args()``, and a script that opened a
hard-coded file while building its defaults.
"""
import contextlib
import io
import re
import sys
from importlib.metadata import entry_points
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _declared_scripts():
    """``name -> module:func`` from ``[project.scripts]`` in pyproject.toml."""
    text = PYPROJECT.read_text(encoding="utf-8")
    block = re.search(r"^\[project\.scripts\]\n(.*?)(?=^\[|\Z)", text, re.M | re.S)
    assert block, "no [project.scripts] table"
    out = {}
    for line in block.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            name, target = (s.strip().strip('"') for s in line.split("=", 1))
            out[name] = target
    return out


SCRIPTS = _declared_scripts()


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_console_script_answers_help(name, monkeypatch, tmp_path):
    module, func = SCRIPTS[name].split(":")
    try:
        mod = __import__(module, fromlist=[func])
    except ImportError as exc:
        pytest.skip(f"{module}: optional dependency missing ({exc.name})")
    main = getattr(mod, func)

    # Run from an empty cwd so any pre-parse relative path access would fail
    # loudly instead of accidentally succeeding against the repo.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [name, "--help"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        with pytest.raises(SystemExit) as exc:
            try:
                main(["--help"])
            except TypeError:                # main() with no argv parameter
                main()
    assert exc.value.code == 0, f"{name} --help exited {exc.value.code}: {buf.getvalue()[-300:]}"
    assert "usage:" in buf.getvalue(), f"{name} produced no usage text"


def test_no_hard_coded_private_paths_in_argparse_defaults():
    """Defaults may point at example data, never at someone's network drive."""
    src = Path(__file__).resolve().parent.parent / "src" / "bambi"
    offenders = []
    for path in src.rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "add_argument" in line and re.search(r"default=r?['\"][A-Z]:\\", line):
                offenders.append(f"{path.relative_to(src)}:{i}")
    assert not offenders, "argparse defaults on drive-letter paths: " + ", ".join(offenders)


def test_scripts_table_matches_installed_entry_points():
    """What pyproject declares is what an install exposes."""
    installed = {ep.name: ep.value for ep in entry_points(group="console_scripts")
                 if ep.name.startswith("bambi-")}
    if not installed:
        pytest.skip("package not installed (pip install -e .)")
    assert installed == SCRIPTS
