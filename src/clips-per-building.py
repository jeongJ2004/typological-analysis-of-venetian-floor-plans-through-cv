import os
import glob
import geopandas as gpd

# ────────────────────────────────────────────────────
# Configuration parameters set by user
HYBRID_DIR = "/content/drive/MyDrive/segmentation_project/vectorized_results/geographic"       # Folder with GeoJSON files of vectorized hybrid results
EDIFICI_FILE = "/content/drive/MyDrive/segmentation_project/FOOTPRINTS/2024_Edifici.geojson"     # Master building footprints (with EDIFI_IDs)
OUTPUT_BASE = "/content/drive/MyDrive/segmentation_project/vectorized_results/per_building"     # Output folder for clipped results
BUFFER_METERS = 1       # Apply 1-meter buffer around each footprint polygon
MIN_COV = 0.05    # Keep clipped results only if coverage ≥ 5%
MAX_COV = 0.30    # And ≤ 30% coverage, to avoid full overlays or noise

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_BASE, exist_ok=True)

# ────────────────────────────────────────────────────
# Step 1 : Load all master building footprints as GeoDataFrame
edf = gpd.read_file(EDIFICI_FILE)  # Each row corresponds to one building (has unique EDIFI_ID)

# ────────────────────────────────────────────────────
# Step 2 : Iterate through each hybrid result file (.geojson format)
for hyb_path in sorted(glob.glob(os.path.join(HYBRID_DIR, "*.geojson"))):
    base = os.path.splitext(os.path.basename(hyb_path))[0]  # Extract filename without extension
    hyb_gdf = gpd.read_file(hyb_path)  # Load vectorized prediction (walls, stairs, windows)

    if hyb_gdf.empty:
        continue  # Skip empty files

    # Ensure CRS matches between hybrid layer and building footprints
    hyb_gdf = hyb_gdf.to_crs(edf.crs)

    # ────────────────────────────────────────────────────
    # Step 3 : For each building footprint in master set
    for _, foot in edf.iterrows():
        eid = foot["EDIFI_ID"]     # Unique building ID
        footprint = foot.geometry        # Polygon of the building

        # Apply a buffer around the footprint (e.g., 1 meter margin)
        clip_poly = footprint.buffer(BUFFER_METERS)

        # Compute intersection of prediction features with buffered footprint
        inters = hyb_gdf.geometry.intersection(clip_poly)

        # Filter out empty intersections
        mask = ~inters.is_empty
        clipped = hyb_gdf[mask].copy()
        clipped.geometry = inters[mask]

        if clipped.empty:
            continue  # No intersecting features found for this building

        # ────────────────────────────────────────────────────
        # Compute how much of the building footprint is covered by prediction
        cov = clipped.unary_union.area / footprint.area

        # Skip if coverage is too small (likely noise) or too large (possibly full overlap)
        if cov < MIN_COV or cov > MAX_COV:
            print(f"Skipped ED{eid} @ {base}: cov={cov:.1%}")
            continue

        # ────────────────────────────────────────────────────
        # Step 4 : Save clipped features into a separate GeoJSON per building
        out_name = f"{base}_ED{eid}.geojson"
        out_path = os.path.join(OUTPUT_BASE, out_name)
        clipped.to_file(out_path, driver="GeoJSON")  # Save only the relevant fragment
        print(f"Saved -> {out_name} (coverage {cov:.1%})")

print("All valid per-building clips completed successfully !")
