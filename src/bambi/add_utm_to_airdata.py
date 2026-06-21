"""
airdata_to_utm.py
-----------------
Converts WGS 84 (longitude, latitude) coordinates in an AirData CSV to
projected UTM (x, y) coordinates and writes a new CSV alongside the input.

The script detects the longitude/latitude columns automatically by matching
column headers that start with "longitude" and "latitude" (AirData annotates
them with units in parentheses, e.g. "longitude(degrees)").

Output columns ``utm_x`` and ``utm_y`` (Easting / Northing in metres) are
appended after all original columns.  Rows where either coordinate is missing
are written with empty utm_x / utm_y cells.

Usage
-----
    python airdata_to_utm.py [CSV_PATH] [--epsg EPSG_CODE] [--output OUTPUT_PATH]

Examples
--------
    # Use all defaults
    python airdata_to_utm.py

    # Custom file, same default EPSG
    python airdata_to_utm.py "path/to/flight.csv"

    # Custom file and EPSG
    python airdata_to_utm.py "path/to/flight.csv" --epsg 32633
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pyproj import Transformer

# ─────────────────────────── USER CONFIGURATION ─────────────────────────────

DEFAULT_CSV    = r"C:\Users\P41743\Desktop\calib\angela\shark\logs\Mar-7th-2024-06-30AM-Flight-Airdata.csv"
DEFAULT_EPSG   = 32643   # target projected CRS (UTM zone); pass only the numeric part

# ─────────────────────────────────────────────────────────────────────────────


def _find_column(headers: List[str], prefix: str) -> str:
    """Return the first header whose lower-case name starts with *prefix*."""
    for h in headers:
        if h.lower().startswith(prefix):
            return h
    raise KeyError(
        f"No column starting with '{prefix}' found in: {headers}"
    )


def convert(csv_path: str, epsg: int, output_path: Optional[str] = None) -> str:
    """
    Read *csv_path*, add utm_x / utm_y columns, write *output_path*.

    Returns the path of the written file.
    """
    df = pd.read_csv(csv_path, dtype=str)

    headers = list(df.columns)
    lon_col = _find_column(headers, "longitude")
    lat_col = _find_column(headers, "latitude")

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    utm_x_vals: list[str] = []
    utm_y_vals: list[str] = []

    for lon_raw, lat_raw in zip(df[lon_col], df[lat_col]):
        try:
            lon = float(lon_raw)
            lat = float(lat_raw)
            x, y = transformer.transform(lon, lat)
            utm_x_vals.append(str(x))
            utm_y_vals.append(str(y))
        except (TypeError, ValueError):
            utm_x_vals.append("")
            utm_y_vals.append("")

    df[f"utm_x(m,EPSG:{epsg})"] = utm_x_vals
    df[f"utm_y(m,EPSG:{epsg})"] = utm_y_vals

    if output_path is None:
        p = Path(csv_path)
        output_path = str(p.with_stem(p.stem + f"_utm{epsg}"))

    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AirData CSV WGS84 coordinates to UTM."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=DEFAULT_CSV,
        help=f"Path to the AirData CSV file (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=DEFAULT_EPSG,
        help=f"Target EPSG code for the UTM projection (default: {DEFAULT_EPSG})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <input>_utm<EPSG>.csv next to input)",
    )
    args = parser.parse_args()

    print(f"Input  : {args.csv}")
    print(f"EPSG   : {args.epsg}")

    try:
        out = convert(args.csv, args.epsg, args.output)
    except FileNotFoundError:
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Output : {out}")


if __name__ == "__main__":
    main()
