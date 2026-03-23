#!/usr/bin/env python3
"""
Make a ~1x1° mask from GEBCO.

Definition:
- GEBCO z < 0 = ocean
- depth (m, positive) = -z
- A 1x1° cell is set to:
    1 if ANY pixel in the cell is ocean with depth <= 3000 m
    0 otherwise

Output:
- mask_1deg (0/1)
"""

import numpy as np
import xarray as xr

INFILE = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/bathymetry/GEBCO_2024.nc"
OUTFILE_MASK = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/bathymetry/GEBCO_2024_mask_1deg_any_ocean_le_3000m.nc"

TARGET_DEG = 1.0
DEPTH_MAX = 2000.0


def _pick_var(ds):
    for name in ["z", "elevation", "bathymetry", "Band1"]:
        if name in ds.data_vars:
            print(f"✔ Using bathymetry variable '{name}'")
            return name
    name = list(ds.data_vars)[0]
    print(f"⚠ Using first variable '{name}'")
    return name


def _pick_coord(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            print(f"✔ Using coordinate '{c}'")
            return c
    raise KeyError(f"Could not find any of {candidates}")


def main():
    print("===================================================")
    print(" GEBCO 1x1° MASK (ANY pixel <= 3000 m)")
    print("===================================================")

    print(f"→ Loading dataset:\n  {INFILE}")
    ds = xr.load_dataset(INFILE)

    zname = _pick_var(ds)
    lat_name = _pick_coord(ds, ["lat", "latitude", "y"])
    lon_name = _pick_coord(ds, ["lon", "longitude", "x"])

    z = ds[zname]
    print(f"→ Bathymetry shape: {z.sizes}")

    if (lat_name in z.dims) and (lon_name in z.dims):
        z = z.transpose(lat_name, lon_name)

    # --------------------------------------------------
    # Per-pixel condition
    # --------------------------------------------------
    print("→ Building per-pixel condition:")
    print("    (z < 0) & ((-z) <= 3000)")
    pixel_ok = (z < 0) & ((-z) <= DEPTH_MAX)

    # --------------------------------------------------
    # Infer coarsening factors
    # --------------------------------------------------
    lat = z[lat_name].values
    lon = z[lon_name].values
    dlat = float(np.nanmedian(np.abs(np.diff(lat))))
    dlon = float(np.nanmedian(np.abs(np.diff(lon))))

    f_lat = int(np.round(TARGET_DEG / dlat))
    f_lon = int(np.round(TARGET_DEG / dlon))

    print(f"→ Native resolution: {dlat:.6f}° × {dlon:.6f}°")
    print(f"→ Coarsening factors: lat={f_lat}, lon={f_lon}")

    # --------------------------------------------------
    # Coarsen: ANY pixel in the 1x1 cell satisfies condition
    # --------------------------------------------------
    print("→ Coarsening to 1x1° using .any() ...")
    keep_1deg = pixel_ok.coarsen(
        {lat_name: f_lat, lon_name: f_lon},
        boundary="trim"
    ).any()

    # Convert bool → 0/1
    mask = keep_1deg.astype(np.uint8)
    mask.name = "mask"

    n_keep = int(mask.sum())
    n_tot = mask.size
    print(f"✔ Cells kept: {n_keep} / {n_tot}")

    mask.attrs.update(
        {
            "description": "1 if ANY pixel in the 1x1° cell is ocean with depth <= 3000 m, else 0",
            "condition": "(z < 0) & ((-z) <= 3000)",
            "coarsened_to_degrees": TARGET_DEG,
            "units": "1",
        }
    )

    print(f"→ Writing mask:\n  {OUTFILE_MASK}")
    mask.to_dataset().to_netcdf(OUTFILE_MASK)

    print("===================================================")
    print(" DONE")
    print("===================================================")


if __name__ == "__main__":
    main()
