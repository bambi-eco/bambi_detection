# -*- coding: utf-8 -*-
"""Linking detections across frames, and across cameras, as arrays.

* :mod:`bambi.tracking.iou` - the built-in tracker: greedy / Hungarian
  box-overlap matching with a centre-distance fallback, plus gap
  interpolation. Works on any box space - pixels or DEM-local metres.
* :mod:`bambi.tracking.matching` - which thermal track and which RGB track
  are the same animal (cross-modal matching by a bootstrapped affine and a
  one-to-one assignment).
"""
