# -*- coding: utf-8 -*-
"""Rendering frames onto the terrain: orthophotos, light-field integrals, and
what is needed around them (frame footprints from a mask, raster sizing,
tiling). Everything runs through alfspy - whichever backend is installed -
via :mod:`bambi.util.render_context`.

* :mod:`bambi.render.masks` - the valid-pixel polygon of a frame from its
  mask image, and where that polygon lands on the ground.
* :mod:`bambi.render.ortho` - shots, orthographic cameras, single-frame
  and integral (ALFS) renders, tiling for large extents, edge erosion.
"""
