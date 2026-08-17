# DEM fixtures

`146_matched_dem.tif` - a 350 x 350 m clip (EPSG:32633, 1 m) of the Austrian
Federal Office of Metrology and Surveying (BEV) ALS digital terrain model
around public BAMBI flight 146, produced by the Dataset repository's
`dem_from_poses.py`. Licence: CC BY 4.0, (c) BEV. Kept in the repo so the
notebooks and CI can mesh it with `bambi.io.dem.geotiff_to_dem` instead of
downloading the 10 GB source tile.
