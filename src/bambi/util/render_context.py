# -*- coding: utf-8 -*-
"""Construction of the alfspy render context and ray caster.

alfspy 3.0 rasterises through one of three interchangeable engines - ModernGL,
PyTorch or Vulkan - and casts rays through one of two casters. Both are chosen
at run time from the environment (``$ALFS_ENGINE``, ``$ALFS_RAYCASTER``) or by
argument, and a single ``make_context()`` serves every engine.

Before 3.0 the engines were two separate distributions that each installed a
package called ``alfspy`` and offered their own factory, so this module used to
sniff which one was present via ``hasattr(_render, "make_torch_context")``.
That test is now meaningless - one package provides all three factories - and
left unchanged it answers "torch" on every machine, which makes torch a hard
requirement of a package that deliberately has no default backend.

Nothing in :mod:`bambi` names an engine. Ask for a context here and alfspy
resolves the rest, so a caller selects one with an environment variable and
changes no code.

alfspy is imported lazily so this module stays importable without it.
"""
import weakref
from typing import Any, List, Optional

#: A render context belonging to whichever engine built it (a
#: ``moderngl.Context``, a ``TorchContext``, or the Vulkan backend's handle).
#: Used for annotation only - they share no common base class.
RenderContext = Any

#: A built ray-casting acceleration structure, or a mesh to build one from.
RayTarget = Any


def available_engines() -> List[str]:
    """The engines that can actually create a context on this machine.

    alfspy probes rather than merely importing, because ModernGL imports
    perfectly well where no usable GL driver exists and only fails when a
    context is created - which is the failure this predicts.

    :return: engine names, or ``[]`` when alfspy is not installed
    """
    try:
        from alfspy.core.backends import available_engines as _available
    except ImportError:
        return []
    try:
        return list(_available())
    except Exception:                      # pragma: no cover - a probe must not raise
        return []


def render_backend() -> str:
    """Name the engine a render would use right now.

    :return: the resolved engine name (``"moderngl"``, ``"torch"``,
        ``"vulkan"``, or whatever ``$ALFS_ENGINE`` names), or ``"unavailable"``
        when alfspy cannot be imported
    """
    try:
        from alfspy.core.backends import resolve_engine
    except ImportError:
        return "unavailable"
    return str(resolve_engine())


def raycast_backend() -> str:
    """Name the ray caster a projection would use right now.

    :return: ``"embree"``, ``"warp"``, or whatever ``$ALFS_RAYCASTER`` names;
        ``"unavailable"`` when alfspy cannot be imported
    """
    try:
        from alfspy.core.raycast import resolve_raycaster
    except ImportError:
        return "unavailable"
    return str(resolve_raycaster())


def make_render_context(device: Optional[str] = None,
                        engine: Optional[str] = None) -> RenderContext:
    """Create an alfspy render context.

    :param device: which device to render on (``"cuda"``, ``"cpu"``). PyTorch
        uses it directly, Vulkan maps ``"cpu"`` onto its software adapter, and
        ModernGL ignores it - OpenGL offers no device selection. Left ``None``,
        alfspy resolves ``$ALFS_DEVICE`` and then the backend's own choice.
    :param engine: force an engine, overriding ``$ALFS_ENGINE``. Left ``None``
        by everything in :mod:`bambi`: the choice belongs to whoever runs the
        pipeline, and naming one here would put it back in the code.
    :return: the engine's context, ready to pass to ``Renderer`` and ``CtxShot``
    :raises RuntimeError: when alfspy is missing, or the selected engine cannot
        create a context here
    """
    try:
        from alfspy.core.backends import make_context
    except ImportError as exc:
        raise RuntimeError(f"alfspy is not available: {exc}") from exc

    try:
        return make_context(engine, device=device)
    except Exception as exc:
        # By far the most common cause is an engine that was selected but never
        # installed - each is a separate pip extra - so name the ones that do
        # work rather than letting an ImportError from three frames down be the
        # whole explanation.
        wanted = engine or render_backend()
        usable = available_engines()
        working = ("Working engines here: " + ", ".join(usable) + "."
                   if usable else "No render engine works here.")
        raise RuntimeError(
            "The '{}' render engine could not start: {}\n\n{}\n"
            "Install it with `pip install \"AlfsPy[{}]\"`, or select one of the "
            "working engines with $ALFS_ENGINE.".format(
                wanted, exc, working, wanted)
        ) from exc


# ---------------------------------------------------------------------------
# Ray casting
# ---------------------------------------------------------------------------

# One built acceleration structure per mesh, keyed on identity. alfspy 3.0
# builds the structure inside every ``pixel_to_world_coord`` call that is handed
# a bare mesh, so a loop over frames - which is every georeferencing call site -
# pays for it once per frame instead of once per flight. Measured at 22x on a
# 24-frame footprint loop.
#
# Keyed on ``id`` with a weak reference alongside, because a Trimesh is not
# reliably hashable and ``id`` alone would hand a recycled address the previous
# mesh's structure. The entry is only used when the weak reference still
# resolves to the very object asked about.
_CASTERS = {}


def make_ray_caster(mesh: RayTarget) -> RayTarget:
    """Build a ray caster for *mesh*, or return it unchanged.

    Falls back to the mesh itself when alfspy is too old to expose a caster or
    one cannot be built: the result is then slower, not wrong.

    :param mesh: a ``trimesh.Trimesh``, a ``(vertices, faces)`` pair, or an
        already-built caster, which is returned as-is
    :return: something ``pixel_to_world_coord`` accepts as its mesh argument
    """
    try:
        from alfspy.core.raycast import RayCaster, create_raycaster
    except ImportError:
        return mesh
    if isinstance(mesh, RayCaster):
        return mesh
    try:
        return create_raycaster(mesh)
    except Exception:                      # pragma: no cover - slower, not broken
        return mesh


def ray_caster_for(mesh: RayTarget) -> RayTarget:
    """A cached ray caster for *mesh*, built on first use.

    This is what the georeferencing helpers pass to alfspy, so that casting the
    same DEM a thousand times builds its acceleration structure once. Callers
    that already hold a caster can pass it instead and skip the cache.

    Assumes a mesh is not mutated in place after it has been cast against -
    true of everything :mod:`bambi.io` produces, which reads a DEM and leaves it
    alone. A caller that does mutate one should pass :func:`make_ray_caster`'s
    result itself and manage its own lifetime.
    """
    try:
        from alfspy.core.raycast import RayCaster
    except ImportError:
        return mesh
    if isinstance(mesh, RayCaster):
        return mesh

    key = id(mesh)
    cached = _CASTERS.get(key)
    if cached is not None:
        ref, caster = cached
        if ref() is mesh:
            return caster
        del _CASTERS[key]                  # the address was recycled

    caster = make_ray_caster(mesh)
    if caster is mesh:
        return mesh                        # nothing was built; do not cache
    try:
        _CASTERS[key] = (weakref.ref(mesh, lambda _ref, k=key: _CASTERS.pop(k, None)),
                         caster)
    except TypeError:                      # pragma: no cover - not weak-referenceable
        pass
    return caster
