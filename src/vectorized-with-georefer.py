import os
import cv2
import numpy as np
from shapely.geometry import Polygon, mapping
import json

# 1. Define input and output directory paths
INPUT_DIR = "/content/drive/MyDrive/segmentation_project/QGIS_res/vectorized_hybrid_qgis"    # Folder containing .tif hybrid images
OUTPUT_DIR = "/content/drive/MyDrive/segmentation_project/vectorized_results/geographic"       # Folder to save GeoJSON vectorized outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Define class-wise BGR color thresholds (lower and upper bounds)
class_colors_bgr = {
    1: ((0,   0,   0),   (10,  10,  10)),   # Wall    : Black ±10 tolerance
    2: ((0, 255,   0),   (0,  255,   0)),   # Stairs  : Pure Green
    3: ((255,  0,   0),   (255, 0,    0)),  # Window  : Pure Blue
}
class_names = {1: "wall", 2: "stairs", 3: "window"}

# 3. Iterate through all .tif files in the input directory
for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith(".tif"):
        continue

    print(f"Processing {fname}")
    img_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        continue

    features = []  # List to collect GeoJSON features

    # 4. For each class, create binary mask -> extract contours -> convert to GeoJSON polygons
    for class_id, (lo, hi) in class_colors_bgr.items():
        lower = np.array(lo, dtype=np.uint8)
        upper = np.array(hi, dtype=np.uint8)
        mask = cv2.inRange(img, lower, upper)  # Generate binary mask for current class

        # Detect external contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cnt.shape[0] < 3:
                continue  # Skip if not enough points to form a polygon

            pts = cnt.squeeze()
            if pts.ndim != 2:
                continue  # Skip if shape is invalid

            poly = Polygon(pts)  # Convert to Shapely polygon

            if not poly.is_valid:
                poly = poly.buffer(0)  # Attempt to fix invalid geometry

            # Append feature to list
            features.append({
                "type": "Feature",
                "properties": {
                    "class": class_id,
                    "label": class_names[class_id]
                },
                "geometry": mapping(poly)
            })

    # 5. Save all features as a GeoJSON file
    base, _ = os.path.splitext(fname)
    out_json = os.path.join(OUTPUT_DIR, f"{base}.geojson")
    with open(out_json, "w") as f:
        json.dump({
            "type": "FeatureCollection",
            "features": features
        }, f, indent=2)

    print(f"Saved -> {out_json}")

print("All vectorized files generated successfully!")
