import os
import glob
import geopandas as gpd

# ────────────────────────────────────────────────────
# 1) User configuration

#    - PERIMETER_PERCENT: use 0.5% of each polygon’s perimeter as the simplification tolerance
INPUT_DIR = "/content/drive/MyDrive/segmentation_project/vectorized_results/per_building_annotated_maxdist"
OUTPUT_DIR = "/content/drive/MyDrive/segmentation_project/vectorized_results/per_building_simplified"
PERIMETER_PERCENT = 0.005  # Use 0.5% of each polygon’s perimeter as tolerance

# Ensure that the output directory exists (create if necessary)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────
# 2) Iterate over all GeoJSON files in the input folder
for geojson_path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.geojson"))):
    # Extract filename (e.g., "hybrid_img0030_modified_ED610_annotated_maxdist.geojson")
    filename = os.path.basename(geojson_path)
    # Derive the base name without extension (e.g., "hybrid_img0030_modified_ED610_annotated_maxdist")
    base_name, _ = os.path.splitext(filename)

    # ────────────────────────────────────────────────────
    # 2.1) Load the annotated GeoJSON into a GeoDataFrame
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        # If the file contains no features, skip it
        print(f"Empty file, skipping: {filename}")
        continue

    # ────────────────────────────────────────────────────
    # 2.2) Simplify each geometry in the GeoDataFrame
    #      - We calculate the polygon’s perimeter and multiply by PERIMETER_PERCENT
    #        to obtain a dynamic tolerance for simplification.
    #      - Setting preserve_topology=True ensures that no invalid geometries are created.
    def dynamic_simplify(geom):
        """
        Compute a dynamic simplification tolerance based on the polygon’s perimeter.
        Tolerance = (perimeter of geom) * PERIMETER_PERCENT.
        """
        # Calculate the geometry’s perimeter (geom.length returns the total boundary length,
        # even for MultiPolygon, GeoPandas sums each part’s length automatically).
        perimeter = geom.length

        # Compute tolerance as a fraction of the perimeter
        tolerance = perimeter * PERIMETER_PERCENT

        # Perform the simplification
        return geom.simplify(tolerance, preserve_topology=True)

    # Apply dynamic_simplify to every geometry
    gdf["geometry"] = gdf["geometry"].apply(dynamic_simplify)

    # ────────────────────────────────────────────────────
    # 2.3) Save the simplified GeoDataFrame back to a new GeoJSON file
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_simplified.geojson")
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"Saved simplified -> {os.path.basename(output_path)}")

print("All geometries in folder have been simplified !")
