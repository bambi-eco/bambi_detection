# -*- coding: utf-8 -*-
"""Architecture guards for the bambi engine.

These are the rules the layer split depends on. They are checked structurally
(by parsing source, not by running it) so they hold on every commit without
any data or heavy dependency being present.

1. **Every module imports.** No dangling imports, no ``src.bambi`` prefixes,
   no ``sys.exit`` at import time. This is what Phase 0 restored and what a
   library must never lose again.
2. **No QGIS anywhere.** The engine is the layer *below* the QGIS plugin.
3. **No backend-specific rendering imports.** ModernGL / torch are alfspy's
   business; the engine goes through ``bambi.util.render_context``.
4. **The numpy contract** on the public surface of engine modules: no file
   paths, no record dicts, no ``open()``; array-shaped results are arrays.
   Enforced on an allowlist that grows as Phase 1 lands each capability
   family, so the rule bites exactly the code that has been written to it.
"""
import ast
import importlib
import io
import contextlib
import pkgutil
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
PKG = SRC / "bambi"


def _py_files():
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


ALL_MODULES = [_module_name(p) for p in _py_files()]


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, node
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module, node


# ---------------------------------------------------------------------------
# 1. every module imports, silently
# ---------------------------------------------------------------------------

# Optional heavy stacks a developer's environment may lack; a module that
# needs one is skipped rather than failed so the suite stays runnable on a
# minimal install. CI installs everything and sees the full set.
_OPTIONAL = ("ultralytics", "torch", "torchvision", "moderngl", "alfspy",
             "geopandas", "contextily", "rasterio", "moviepy", "imageio_ffmpeg",
             "webcolors", "rdp", "tqdm", "requests", "hachoir", "piexif")


@pytest.mark.parametrize("module", ALL_MODULES)
def test_module_imports_cleanly(module):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            importlib.import_module(module)
    except SystemExit as exc:                     # a library must never do this
        pytest.fail(f"{module} called sys.exit({exc.code}) at import time")
    except ImportError as exc:
        # Some modules re-raise a plain ImportError with guidance; walk the
        # chain to the underlying ModuleNotFoundError to see what was missing.
        cause = exc
        while cause is not None and not isinstance(cause, ModuleNotFoundError):
            cause = cause.__cause__ or cause.__context__
        missing = (getattr(cause, "name", "") or "").split(".")[0]
        if missing in _OPTIONAL:
            pytest.skip(f"optional dependency not installed: {missing}")
        raise
    # ultralytics prints a banner on import (behind ANSI clear-line codes);
    # nothing of ours should write anything.
    noise = re.sub(r"\x1b\[[0-9;]*[A-Za-z]|\r", "", buf.getvalue()).strip()
    if noise and not noise.startswith("Ultralytics"):
        pytest.fail(f"{module} writes to stdout/stderr on import: {noise[:120]!r}")


def test_no_repo_root_relative_imports():
    """``from src.bambi ...`` only works when run from the repo root."""
    offenders = []
    for path in _py_files():
        for name, _ in _imports(ast.parse(path.read_text(encoding="utf-8"))):
            if name == "src" or name.startswith("src."):
                offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"repo-root-relative imports: {sorted(set(offenders))}"


def test_no_sys_exit_at_module_level():
    """Scripts may exit inside main(); modules may not exit while importing."""
    offenders = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # module-level statements and any try/except directly at module level
        for node in tree.body:
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    break
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "exit"
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "sys"):
                    # allowed only under `if __name__ == "__main__":`
                    if not _under_main_guard(tree, sub):
                        offenders.append(f"{path.relative_to(SRC)}:{sub.lineno}")
    assert not offenders, f"sys.exit at import time: {offenders}"


def _under_main_guard(tree, target):
    for node in tree.body:
        if isinstance(node, ast.If):
            test = ast.unparse(node.test).replace("'", '"')
            if test == '__name__ == "__main__"':
                if any(sub is target for sub in ast.walk(node)):
                    return True
    return False


# ---------------------------------------------------------------------------
# 2./3. layering
# ---------------------------------------------------------------------------

FORBIDDEN_PREFIXES = ("qgis", "PyQt5", "PyQt6", "moderngl")


@pytest.mark.parametrize("path", _py_files(), ids=lambda p: str(p.relative_to(PKG)))
def test_no_forbidden_imports(path):
    if path.name == "render_context.py":
        return                                    # the one place that names backends
    bad = [name for name, _ in _imports(ast.parse(path.read_text(encoding="utf-8")))
           if name.split(".")[0] in FORBIDDEN_PREFIXES]
    assert not bad, f"{path.relative_to(SRC)} imports {bad}"


def test_render_contexts_come_from_the_neutral_helper():
    """``make_mgl_context`` exists only on the ModernGL build of alfspy."""
    offenders = [str(p.relative_to(SRC)) for p in _py_files()
                 if p.name != "render_context.py"
                 and "make_mgl_context" in p.read_text(encoding="utf-8")]
    assert not offenders, f"{offenders} call make_mgl_context(); use bambi.util.render_context"


# ---------------------------------------------------------------------------
# 4. the numpy contract (engine surface only; allowlist grows with Phase 1)
# ---------------------------------------------------------------------------

#: Modules written to the contract. Add each family here as it lands; the
#: scanner then holds it to array-in / array-out on its public functions.
CONTRACT_MODULES = [
    "bambi.util.srt_air_data_converter",
    # Phase 1, family 1 - extraction & poses
    "bambi.geo.poses",
    "bambi.geo.calibration",
    # Phase 1, family 2 - geo-referencing
    "bambi.geo.camera",
    "bambi.geo.georef",
    "bambi.geo.dem",
    "bambi.testing.synthetic",
    # Phase 1, family 3 - tracking
    "bambi.tracking.iou",
    "bambi.tracking.matching",
    # Phase 1, family 4 - survey analytics
    "bambi.survey.transects",
    "bambi.survey.perpendicular",
    "bambi.survey.density",
    "bambi.survey.distance_sampling",
    "bambi.survey.population",
    # bambi.io.* is the file edge and is deliberately NOT here.
    # bambi.util.paths is deliberately NOT here: it is a path helper for the
    # io/cli edge, which is exactly what the contract keeps out of the engine.
]

_PATH_PARAM = re.compile(r"(^|_)(path|file|dir|folder|filename)s?$")


def _public_functions(tree):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            yield node
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and not sub.name.startswith("_"):
                    yield sub


def _annotation_text(node):
    return ast.unparse(node) if node is not None else ""


@pytest.mark.parametrize("module", CONTRACT_MODULES)
def test_contract_no_paths_or_records_in_public_signatures(module):
    path = SRC / (module.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    problems = []
    for fn in _public_functions(tree):
        for arg in fn.args.args + fn.args.kwonlyargs:
            if arg.arg in ("self", "cls"):
                continue
            ann = _annotation_text(arg.annotation)
            if _PATH_PARAM.search(arg.arg):
                problems.append(f"{fn.name}({arg.arg}): path-like parameter")
            if re.search(r"\b(dict|Dict|List\[dict\]|List\[Dict)\b", ann):
                problems.append(f"{fn.name}({arg.arg}: {ann}): record-dict parameter")
        ret = _annotation_text(fn.returns)
        if re.search(r"List\[List\[|List\[dict\]|List\[Dict\[", ret):
            problems.append(f"{fn.name} -> {ret}: array-shaped result not an array")
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == "open":
                problems.append(f"{fn.name}: opens a file (belongs in the io/cli edge)")
    assert not problems, "\n".join(problems)
