

try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("bambi-detection")
except Exception:  # pragma: no cover - not installed as a distribution
    __version__ = "1.0.0"
