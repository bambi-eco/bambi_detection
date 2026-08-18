# -*- coding: utf-8 -*-
"""Survey analytics: from ground positions to what a wildlife survey reports.

* :mod:`bambi.survey.transects` - the flight path as a polyline, its length,
  and how a flight is cut into transects (frame ranges).
* :mod:`bambi.survey.perpendicular` - perpendicular distances from
  detections or tracks to the flight route, and flat-earth frame footprints.
* :mod:`bambi.survey.density` - kernel-density surfaces (animals per
  hectare) and frame-overlap coverage grids.
* :mod:`bambi.survey.distance_sampling` - conventional line-transect
  distance sampling: detection function, effective strip width, density.
* :mod:`bambi.survey.population` - transect count/area tables and the
  naive / bootstrap / zero-inflated negative binomial density estimators.
"""
