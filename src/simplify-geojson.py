import os
import glob
import geopandas as gpd

# ────────────────────────────────────────────────────
# 1) User configuration
#    - SIMPLIFY_TOLERANCE: tolerance value for simplification (in the same units as the CRS, e.g., meters)
INPUT_DIR = "/content/drive/MyDrive/segmentation_project/vectorized_results/per_building_annotated_maxdist"
OUTPUT_DIR = "/content/drive/MyDrive/segmentation_project/vectorized_results/per_building_simplified"
SIMPLIFY_TOLERANCE = 1.0  # Simplification tolerance (e.g., 1.0 meter)

# Ensure the output directory exists; if not, create it
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────
# 2) Iterate over all GeoJSON files in the input folder
for path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.geojson"))):
    # Extract just the filename (e.g., "hybrid_img0030_modified_ED610_annotated_maxdist.geojson")
    fname = os.path.basename(path)
    # Derive the base name without extension (e.g., "hybrid_img0030_modified_ED610_annotated_maxdist")
    base, _ = os.path.splitext(fname)

    # ────────────────────────────────────────────────────
    # 2.1) Load the annotated GeoJSON into a GeoDataFrame
    gdf = gpd.read_file(path)
    if gdf.empty:
        # If the file contains no features, skip it
        print(f"Empty file, skip: {fname}")
        continue

    # ────────────────────────────────────────────────────
    # 2.2) Simplify each geometry in the GeoDataFrame
    #      The `preserve_topology=True` flag ensures that no invalid geometries (e.g., self-intersections) are created.
    #      The `SIMPLIFY_TOLERANCE` parameter controls how aggressively vertices are removed:
    #        - Larger tolerance -> more vertices removed -> more generalized (coarser) shape
    #        - Smaller tolerance -> fewer vertices removed -> shape closer to the original
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: geom.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
    )

    # ────────────────────────────────────────────────────
    # 2.3) Save the simplified GeoDataFrame back to a new GeoJSON file
    out_path = os.path.join(OUTPUT_DIR, f"{base}_simplified.geojson")
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Saved simplified -> {os.path.basename(out_path)}")

print("All geometries in folder simplified !")
