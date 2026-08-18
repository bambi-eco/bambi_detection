# -*- coding: utf-8 -*-
"""Load the QGIS plugin's own code as a parity oracle (unit tier, no QGIS).

The plugin checkout is pointed at with ``BAMBI_PLUGIN_REPO``; its
``tests/qgis_stub.py`` is loaded by file path (the plugin's ``tests`` package
must not shadow ours) so ``bambi_wildlife_detection`` imports headless, exactly
as the plugin's own unit tier runs it.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = os.environ.get("BAMBI_PLUGIN_REPO")

requires_plugin = pytest.mark.skipif(not REPO, reason="set BAMBI_PLUGIN_REPO to compare against the plugin")


def load_plugin():
    """Import the plugin package with qgis stubbed; returns the package root name."""
    if not REPO:
        pytest.skip("set BAMBI_PLUGIN_REPO to compare against the plugin")
    repo = Path(REPO)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if "bambi_qgis_stub" not in sys.modules:
        spec = importlib.util.spec_from_file_location("bambi_qgis_stub", repo / "tests" / "qgis_stub.py")
        stub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stub)
        stub.install_qgis_stubs()
        sys.modules["bambi_qgis_stub"] = stub
    return "bambi_wildlife_detection"


def plugin_processor():
    """A ``BambiProcessor`` (its ``__init__`` is side-effect free)."""
    load_plugin()
    from bambi_wildlife_detection.bambi_processing import BambiProcessor
    return BambiProcessor()


def plugin_module(dotted: str):
    """``plugin_module("core.population")`` -> the plugin's module object."""
    load_plugin()
    import importlib
    return importlib.import_module(f"bambi_wildlife_detection.{dotted}")
