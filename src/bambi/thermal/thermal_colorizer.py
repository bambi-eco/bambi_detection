# -*- coding: utf-8 -*-
"""
Thermal colorizer for radiometric JPEG images.

Converts a per-pixel temperature array (°C) produced by an external
thermal parser into a colormapped BGR image suitable for OpenCV processing
(undistortion, encoding, etc.).

The actual parsing of the proprietary DJI / FLIR thermal format is delegated
to a caller-supplied *parse_fn* so that this module has no dependency on the
DJI Thermal SDK, exiftool, or any other proprietary library.

Typical usage (inside the QGIS plugin, where the SDK is available)::

    from bambi_thermal import Thermal
    from bambi.webgl.thermal_colorizer import ThermalColorizer

    thermal = Thermal(dtype=np.float32)
    colorizer = ThermalColorizer(
        parse_fn=thermal.parse,
        colormap="white-hotspot",
        lo_threshold=10.0,
        hi_threshold=45.0,
    )
    bgr = colorizer.load_as_bgr("/path/to/DJI_20260301_T.JPG")
    # bgr is a uint8 H×W×3 array in BGR order, ready for cv2.remap / cv2.imwrite
    thermal.close()
"""

from typing import Callable, Optional

import cv2
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Colormap name aliases (display name → matplotlib name)
# ---------------------------------------------------------------------------

_CMAP_ALIASES = {
    "white-hotspot": "gray",    # hot → white, cold → black
    "black-hotspot": "gray_r",  # hot → black, cold → white
}

COLORMAPS = [
    "white-hotspot",
    "black-hotspot",
    "plasma",
    "inferno",
    "magma",
    "viridis",
    "jet",
]


class ThermalColorizer:
    """Convert a radiometric thermal JPEG to a colormapped BGR image.

    The class is intentionally thin: it owns only the colormap configuration
    and delegates all thermal parsing to the *parse_fn* supplied by the caller.
    This keeps the DJI SDK (or any other proprietary dependency) outside this
    library.

    :param parse_fn: Callable that accepts a file path and returns a 2-D
        float32 numpy array of per-pixel temperatures in °C.  Typically
        ``Thermal(dtype=np.float32).parse`` from the QGIS plugin.
    :param colormap: Colormap name.  One of :data:`COLORMAPS` or any
        matplotlib colormap name.  Default: ``"white-hotspot"`` (grayscale,
        hot → white).
    :param lo_threshold: Lower clip temperature in °C.  Pixels whose
        temperature is below this value are rendered black.  ``None`` means
        no lower clip (use the image minimum).
    :param hi_threshold: Upper clip temperature in °C.  Pixels above this
        value are rendered black.  ``None`` means no upper clip (use the
        image maximum).
    """

    def __init__(
        self,
        parse_fn: Callable[[str], npt.NDArray[np.float32]],
        colormap: str = "white-hotspot",
        lo_threshold: Optional[float] = None,
        hi_threshold: Optional[float] = None,
    ) -> None:
        self._parse_fn = parse_fn
        self._colormap = colormap
        self._lo = lo_threshold
        self._hi = hi_threshold
        self._cmap = None   # resolved lazily on first use

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_as_bgr(self, path: str) -> npt.NDArray[np.uint8]:
        """Parse *path* as a radiometric thermal image and return a BGR frame.

        The returned array is uint8 H×W×3 in BGR channel order, directly
        compatible with :func:`cv2.remap` and :func:`cv2.imwrite`.

        :param path: Absolute path to the radiometric JPEG.
        :raises Exception: Propagates any exception raised by *parse_fn*.
        """
        temp = self._parse_fn(path)
        rgb = self._apply_colormap(temp)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cmap(self):
        """Resolve and cache the matplotlib colormap object."""
        if self._cmap is None:
            import matplotlib as mpl
            mpl_name = _CMAP_ALIASES.get(self._colormap, self._colormap)
            if hasattr(mpl, "colormaps"):
                self._cmap = mpl.colormaps[mpl_name]
            else:
                import matplotlib.cm as cm
                self._cmap = cm.get_cmap(mpl_name)
        return self._cmap

    def _apply_colormap(
        self, temp: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.uint8]:
        """Return a uint8 RGB H×W×3 array from *temp*."""
        arr = np.asarray(temp, dtype=np.float32)

        mask = np.zeros(arr.shape, dtype=bool)
        if self._lo is not None:
            mask |= arr < self._lo
        if self._hi is not None:
            mask |= arr > self._hi

        lo = self._lo if self._lo is not None else float(arr.min())
        hi = self._hi if self._hi is not None else float(arr.max())
        if hi <= lo:
            hi = lo + 1.0

        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        rgba = self._get_cmap()(norm)               # H×W×4 float64
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        rgb[mask] = 0
        return rgb
