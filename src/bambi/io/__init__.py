# -*- coding: utf-8 -*-
"""The file edge of the engine.

Everything under :mod:`bambi.io` reads files into arrays and writes arrays back
out. It is deliberately *outside* the numpy contract that governs the rest of
the engine: paths, dicts and open handles live here and only here, so that the
compute modules never have to know what a poses file looks like.
"""
