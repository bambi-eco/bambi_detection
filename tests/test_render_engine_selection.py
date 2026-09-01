# -*- coding: utf-8 -*-
"""The engine and the ray caster are the caller's choice, not the code's.

alfspy 3.0 carries three render engines and two ray casters in one package,
selected at run time by ``$ALFS_ENGINE`` / ``$ALFS_RAYCASTER``. Before 3.0 they
were two distributions, and ``bambi.util.render_context`` told them apart by
asking whether ``make_torch_context`` existed - a test that is now true on every
machine, so the wrapper answered "torch" whatever was asked for and made torch a
hard requirement of a package that deliberately has no default backend.

Nothing catches that by running the pipeline: a torch-only CI matrix passes
either way, and so do the notebooks - they just quietly render on an engine
nobody chose. So it is pinned here.
"""
import sys
import types

import pytest

from bambi.util import render_context


def _fake_backends(monkeypatch, *, engines=("moderngl", "torch"),
                   resolved="moderngl", fails=None, calls=None):
    """Install a stand-in ``alfspy.core.backends`` - the 3.0 context registry."""
    backends = types.ModuleType("alfspy.core.backends")

    def make_context(engine=None, device=None, **options):
        if calls is not None:
            calls.append({"engine": engine, "device": device, **options})
        if fails:
            raise RuntimeError(fails)
        return "CONTEXT:%s" % (engine or resolved)

    backends.make_context = make_context
    backends.available_engines = lambda: list(engines)
    backends.resolve_engine = lambda engine=None: engine or resolved

    alfspy = types.ModuleType("alfspy")
    alfspy.__path__ = []
    core = types.ModuleType("alfspy.core")
    core.__path__ = []
    core.backends = backends
    alfspy.core = core
    monkeypatch.setitem(sys.modules, "alfspy", alfspy)
    monkeypatch.setitem(sys.modules, "alfspy.core", core)
    monkeypatch.setitem(sys.modules, "alfspy.core.backends", backends)
    return backends


# ---------------------------------------------------------------------------
# The engine follows the environment
# ---------------------------------------------------------------------------

def test_the_backend_is_whatever_alfspy_resolves(monkeypatch):
    _fake_backends(monkeypatch, resolved="vulkan")
    assert render_context.render_backend() == "vulkan"


def test_no_engine_is_named_by_the_caller(monkeypatch):
    """The choice reaches alfspy through the environment, not an argument -
    passing one here is what pinned every run to torch before."""
    calls = []
    _fake_backends(monkeypatch, resolved="moderngl", calls=calls)
    render_context.make_render_context()
    assert calls == [{"engine": None, "device": None}]


def test_an_engine_can_still_be_forced(monkeypatch):
    calls = []
    _fake_backends(monkeypatch, calls=calls)
    assert render_context.make_render_context(engine="vulkan") == "CONTEXT:vulkan"
    assert calls == [{"engine": "vulkan", "device": None}]


def test_the_device_is_forwarded(monkeypatch):
    calls = []
    _fake_backends(monkeypatch, calls=calls)
    render_context.make_render_context(device="cuda")
    assert calls == [{"engine": None, "device": "cuda"}]


def test_a_missing_engine_names_the_ones_that_work(monkeypatch):
    """Selecting an engine whose extra was never installed is the common
    failure, so the message has to say what to install or pick instead."""
    _fake_backends(monkeypatch, engines=("moderngl",), resolved="vulkan",
                   fails="No module named 'wgpu'")
    with pytest.raises(RuntimeError) as excinfo:
        render_context.make_render_context()
    message = str(excinfo.value)
    assert "vulkan" in message and "moderngl" in message
    assert "AlfsPy[vulkan]" in message


def test_missing_alfspy_is_reported_as_such(monkeypatch):
    for name in ("alfspy", "alfspy.core", "alfspy.core.backends"):
        monkeypatch.setitem(sys.modules, name, None)
    assert render_context.render_backend() == "unavailable"
    assert render_context.available_engines() == []
    with pytest.raises(RuntimeError, match="alfspy is not available"):
        render_context.make_render_context()


# ---------------------------------------------------------------------------
# Against the installed alfspy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["moderngl", "torch", "vulkan"])
def test_the_env_var_selects_the_engine_for_real(engine, monkeypatch):
    """The seam only counts if the context that comes back belongs to the
    engine that was asked for."""
    pytest.importorskip("alfspy")
    from alfspy.core.backends import available_engines, backend_for_context, get_backend

    if engine not in available_engines():
        pytest.skip(f"{engine} backend not installed here")
    monkeypatch.setenv("ALFS_ENGINE", engine)

    assert render_context.render_backend() == engine
    ctx = render_context.make_render_context()
    assert backend_for_context(ctx) is get_backend(engine)


def test_the_raycaster_follows_the_environment(monkeypatch):
    pytest.importorskip("alfspy")
    monkeypatch.setenv("ALFS_RAYCASTER", "warp")
    assert render_context.raycast_backend() == "warp"
    monkeypatch.delenv("ALFS_RAYCASTER")
    assert render_context.raycast_backend() == "embree"


# ---------------------------------------------------------------------------
# Ray-caster reuse
# ---------------------------------------------------------------------------

def test_the_caster_is_built_once_per_mesh():
    """alfspy 3.0 builds the acceleration structure inside every call handed a
    bare mesh, and every georeferencing call site casts the same DEM once per
    frame - measured at 22x on a 24-frame footprint loop."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("alfspy")
    from alfspy.core.raycast import RayCaster

    mesh = trimesh.creation.box()
    first = render_context.ray_caster_for(mesh)
    assert isinstance(first, RayCaster)
    assert render_context.ray_caster_for(mesh) is first


def test_a_caster_passes_straight_through():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("alfspy")

    caster = render_context.make_ray_caster(trimesh.creation.box())
    assert render_context.ray_caster_for(caster) is caster


def test_two_meshes_get_two_casters():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("alfspy")

    one, two = trimesh.creation.box(), trimesh.creation.box()
    assert render_context.ray_caster_for(one) is not render_context.ray_caster_for(two)


def test_the_cache_does_not_keep_meshes_alive():
    """A cache keyed on a DEM must not be the reason the DEM stays in memory."""
    import gc
    import weakref

    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("alfspy")

    mesh = trimesh.creation.box()
    render_context.ray_caster_for(mesh)
    ref = weakref.ref(mesh)
    del mesh
    gc.collect()
    assert ref() is None
