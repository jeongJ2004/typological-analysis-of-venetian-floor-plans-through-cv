import os
import cv2
import numpy as np
from shapely.geometry import Polygon, mapping
import json

# 1. Define input and output directory paths
INPUT_DIR = "/content/drive/MyDrive/segmentation_project/hybrid_results_final"   # Folder containing hybrid images
OUTPUT_DIR = "/content/drive/MyDrive/segmentation_project/vectorized_results"     # Output folder for GeoJSON results
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Define class-wise color boundaries in BGR (lower, upper bounds)
class_colors_bgr = {
    1: ((0,   0,   0),   (10,  10,  10)),   # Wall : Black ±10 margin
    2: ((0, 255,   0),   (0,  255,   0)),   # Stairs : Pure Green
    3: ((255,  0,   0),   (255, 0,    0)),  # Window : Pure Blue
}
class_names = {1: "wall", 2: "stairs", 3: "window"}

# 3. Iterate over all image files in the input directory
for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    print(f"▶ Processing {fname}")
    img_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load: {img_path}")
        continue

    features = []  # List to store extracted GeoJSON features

    # 4. For each class, extract contours -> Polygons -> GeoJSON Features
    for class_id, (lower_bgr, upper_bgr) in class_colors_bgr.items():
        lower = np.array(lower_bgr, dtype=np.uint8)
        upper = np.array(upper_bgr, dtype=np.uint8)
        mask = cv2.inRange(img, lower, upper)  # Binary mask for the class

        # Extract external contours for the given class
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cnt.shape[0] < 3:
                continue  # Ignore small or invalid shapes

            pts = cnt.squeeze()
            if pts.ndim != 2:
                continue  # Skip if not valid 2D shape

            poly = Polygon(pts)  # Convert contour to Shapely Polygon

            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix self-intersections or invalid shapes

            # Create GeoJSON feature
            features.append({
                "type": "Feature",
                "properties": {
                    "class": class_id,
                    "label": class_names[class_id]
                },
                "geometry": mapping(poly)
            })

    # 5. Write extracted features to a GeoJSON file
    base, _ = os.path.splitext(fname)
    out_json = os.path.join(OUTPUT_DIR, f"{base}.geojson")

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(out_json, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"Saved -> {out_json}")

print("All vectorization completed successfully !")
