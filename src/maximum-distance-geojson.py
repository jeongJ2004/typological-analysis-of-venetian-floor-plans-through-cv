import os
import glob
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon

# ────────────────────────────────────────────────────
# 1) User configuration

EDIFICI_FILE = "/content/drive/MyDrive/segmentation_project/FOOTPRINTS/2024_Edifici.geojson"  # GeoJSON with building footprints
INPUT_DIR = "/content/drive/MyDrive/segmentation_project/vectorized_results/per_building"  # Per-building vectorized clips
OUTPUT_DIR = "/content/drive/MyDrive/segmentation_project/vectorized_results/per_building_annotated_maxdist"  # Output path for annotated clips

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────
# 2) Load master building footprints and index by EDIFI_ID

edf = gpd.read_file(EDIFICI_FILE).set_index("EDIFI_ID")

# ────────────────────────────────────────────────────
def compute_max_distance(footprint: Polygon, geom):
    """
    Compute the maximum Euclidean distance from any vertex of `geom` to the boundary of `footprint`.
    Handles Polygon, MultiPolygon, or other geometry types using bounding box fallback.
    """
    coords = []

    def collect(poly: Polygon):
        # Collect both outer and inner boundary coordinates
        coords.extend(poly.exterior.coords)
        for interior in poly.interiors:
            coords.extend(interior.coords)

    if isinstance(geom, Polygon):
        collect(geom)
    elif isinstance(geom, MultiPolygon):
        for part in geom.geoms:
            collect(part)
    else:
        # Fallback : use the corners of the bounding box
        xmin, ymin, xmax, ymax = geom.bounds
        coords.extend([
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymin),
            (xmax, ymax),
        ])

    # Calculate maximum distance to the footprint
    max_d = 0.0
    for x, y in coords:
        d = footprint.distance(Point(x, y))
        if d > max_d:
            max_d = d
    return max_d

# ────────────────────────────────────────────────────
# 3) Process each per-building GeoJSON clip

for path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.geojson"))):
    fname = os.path.basename(path)
    base, _ = os.path.splitext(fname)

    # Extract EDIFI_ID and source image name from filename
    try:
        eid = int(base.split("_ED")[-1])
        image_name = base.rsplit("_ED", 1)[0]
    except ValueError:
        print(f"Skipping unrecognized file: {fname}")
        continue

    # Load GeoJSON clip
    clip_gdf = gpd.read_file(path)
    if clip_gdf.empty:
        print(f"  -> empty, skip {fname}")
        continue

    # Align CRS with building footprints
    clip_gdf = clip_gdf.to_crs(edf.crs)
    footprint = edf.loc[eid].geometry

    # Compute max distance for each geometry in the clip
    clip_gdf["EDIFI_ID"] = eid
    clip_gdf["image_name"] = image_name
    clip_gdf["distance"] = clip_gdf.geometry.apply(lambda g: compute_max_distance(footprint, g))

    # Save the annotated GeoJSON
    out_path = os.path.join(OUTPUT_DIR, f"{base}_annotated_maxdist.geojson")
    clip_gdf.to_file(out_path, driver="GeoJSON")
    print(f"{fname} -> saved {os.path.basename(out_path)}")

print("All clips annotated with max-distance!")
