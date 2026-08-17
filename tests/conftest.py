# -*- coding: utf-8 -*-
"""Shared pytest configuration for the bambi-detection test suite.

The suite has two tiers, selected by marker:

* the default tier is pure unit tests - no network, no model weights, no
  flight data - and must stay fast enough to run on every commit;
* ``@pytest.mark.slow`` tests download a public BAMBI flight through the
  Dataset repository's tooling and cache it (see ``flight_dir``); they run in
  the scheduled CI job and on demand.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_report_header(config):
    return f"bambi-detection src: {SRC}"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Where downloaded flights are cached between runs.

    ``BAMBI_TEST_DATA`` lets CI point this at a persistent cache volume; the
    default lives under the repo so a developer's first ``pytest -m slow``
    populates it once.
    """
    root = Path(os.environ.get("BAMBI_TEST_DATA", REPO_ROOT / ".test-data"))
    root.mkdir(parents=True, exist_ok=True)
    return root
