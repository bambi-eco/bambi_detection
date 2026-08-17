# -*- coding: utf-8 -*-
"""Small path helpers shared by the geo-referencing scripts.

``deviating_folders`` was lost in the 2024 repository clean-up while its
callers survived, which left ``georeferenced_tracking``, ``tracks_to_geojson``
and ``visualize_tracks_global_and_image`` unimportable. Restored here, in a
module the callers can reach without the old ``src.bambi`` prefix.
"""
import os


def deviating_folders(parent_path: str, sub_path: str) -> str:
    """Return the sub-folder of *sub_path* relative to *parent_path*.

    If *sub_path* is a file its directory is used. Returns ``""`` when the two
    resolve to the same folder, so callers can join the result unconditionally.

    :param parent_path: the folder every input lives under
    :param sub_path: a file or folder somewhere below *parent_path*
    """
    folder = sub_path if os.path.isdir(sub_path) else os.path.dirname(sub_path)
    rel = os.path.relpath(folder, start=parent_path)
    return "" if rel == "." else rel
