import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import time

GOAS_SHP = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/ocean_mask/goas_v01.shp"
OUT_CSV  = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/ocean_mask/ocean_area_south_of_30S.csv"

t0 = time.time()
print("▶ Starting ocean area computation south of 30°S")

# ------------------------------------------------------------
print("▶ Loading GOAS shapefile...")
goas = gpd.read_file(GOAS_SHP).to_crs("EPSG:4326")
print(f"   Loaded {len(goas)} polygons in {time.time()-t0:.1f} s")

# ------------------------------------------------------------
print("▶ Dissolving all ocean and sea polygons (this can be slow)...")
t_dissolve = time.time()
ocean_union = goas.dissolve()
print(f"   Dissolve complete in {time.time()-t_dissolve:.1f} s")

# ------------------------------------------------------------
print("▶ Building south-of-30°S clip polygon...")
south_of_30S = gpd.GeoDataFrame(
    geometry=[box(-180, -90, 180, -30)],
    crs="EPSG:4326"
)
print("   Clip polygon ready")

# ------------------------------------------------------------
print("▶ Clipping ocean geometry to south of 30°S...")
t_clip = time.time()
ocean_south = gpd.overlay(
    ocean_union.reset_index(drop=True),
    south_of_30S,
    how="intersection"
)
print(f"   Clip complete in {time.time()-t_clip:.1f} s")

# ------------------------------------------------------------
print("▶ Reprojecting to equal-area CRS (EPSG:6933)...")
t_proj = time.time()
ocean_south_eq = ocean_south.to_crs("EPSG:6933")
print(f"   Reprojection complete in {time.time()-t_proj:.1f} s")

# ------------------------------------------------------------
print("▶ Computing total area...")
OCEAN_AREA_M2 = float(ocean_south_eq.area.sum())
print("   Area computed")

# ------------------------------------------------------------
print("▶ Saving result to CSV...")
pd.DataFrame(
    {
        "region": ["Ocean south of 30S (GOAS v01)"],
        "ocean_area_m2": [OCEAN_AREA_M2],
        "ocean_area_1e13_m2": [OCEAN_AREA_M2 / 1e13]
    }
).to_csv(OUT_CSV, index=False)

# ------------------------------------------------------------
print("✔ DONE")
print(f"   Ocean area south of 30S = {OCEAN_AREA_M2:.3e} m²")
print(f"   ({OCEAN_AREA_M2/1e13:.2f} × 10^13 m²)")
print(f"   Total runtime: {time.time()-t0:.1f} s")
print(f"   Saved to: {OUT_CSV}")
