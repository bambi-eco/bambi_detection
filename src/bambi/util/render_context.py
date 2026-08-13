# -*- coding: utf-8 -*-
"""Backend-neutral construction of the alfspy render context.

alfspy ships in two implementations that install under the same package name:

* ``alfs_py`` rasterises through ModernGL and offers ``make_mgl_context()``;
* ``alfs_pytorch`` rasterises through PyTorch tensors and offers
  ``make_torch_context()``, keeping ``make_mgl_context()`` as a deprecated
  alias so ModernGL-era callers keep working.

Nothing in :mod:`bambi` names a backend.  Ask for a context here and the torch
factory is preferred when present — which also selects CUDA automatically —
falling back to the ModernGL one otherwise.  Swapping the installed alfspy then
needs no change anywhere else, and ``moderngl`` is never imported directly.

alfspy is imported lazily so this module stays importable without it.
"""
from typing import Any, Optional

#: A render context belonging to whichever alfspy backend is installed
#: (``moderngl.Context`` or ``alfspy.core.torchgl.context.TorchContext``).
#: Used for annotation only — the two share no common base class.
RenderContext = Any


def render_backend() -> str:
    """Name the rasteriser the installed alfspy uses.

    :return: ``"torch"``, ``"moderngl"``, or ``"unavailable"`` when alfspy
        cannot be imported at all
    """
    try:
        from alfspy.render import render as _render
    except ImportError:
        return "unavailable"
    return "torch" if hasattr(_render, "make_torch_context") else "moderngl"


def make_render_context(device: Optional[str] = None) -> RenderContext:
    """Create an alfspy render context using whichever backend is installed.

    :param device: torch device to render on (e.g. ``"cuda"``, ``"cpu"``).
        Only meaningful for the torch backend, which otherwise selects CUDA
        when it is available; ignored by the ModernGL backend.
    :return: the backend's context object, ready to pass to ``Renderer`` and
        ``CtxShot``
    :raises RuntimeError: when alfspy is not installed
    """
    try:
        from alfspy.render import render as _render
    except ImportError as exc:
        raise RuntimeError(f"alfspy is not available: {exc}") from exc

    factory = getattr(_render, "make_torch_context", None)
    if factory is not None:
        return factory(device=device) if device else factory()
    return _render.make_mgl_context()
