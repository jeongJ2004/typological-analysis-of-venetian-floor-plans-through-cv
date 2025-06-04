import os
import glob
import shutil
import geopandas as gpd

# ─────────────────────────────────────────────────────
# 1) User configuration

EDIFICI_FILE = "/content/drive/MyDrive/segmentation_project/FOOTPRINTS/2024_Edifici.geojson"   # GeoJSON containing all building footprints
INPUT_DIR = "/content/drive/MyDrive/segmentation_project/vectorized_results/per_building"   # Directory with per-building vectorized GeoJSON clips
OUTPUT_DIR = "/content/drive/MyDrive/segmentation_project/vectorized_results/best_per_building"  # Output directory for best-selected clips
MIN_COV = 0.05   # Minimum coverage ratio (5%)
MAX_COV = 0.30   # Maximum coverage ratio (30%)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────
# 2) Load building footprints and compute their area

edf = gpd.read_file(EDIFICI_FILE).set_index("EDIFI_ID")   # Load GeoDataFrame with building footprints
footprint_area = edf.geometry.area  # Series: {EDIFI_ID: area}

# ─────────────────────────────────────────────────────
# 3) Filter valid per-building clips based on coverage ratio

records = {}  # Dictionary: {edifi_id -> list of (coverage_ratio, geojson_path)}

for path in glob.glob(os.path.join(INPUT_DIR, "*.geojson")):
    fname = os.path.basename(path)

    # Extract building ID from filename (e.g., *_ED123.geojson)
    try:
        edifi_id = int(fname.split("_ED")[-1].split(".geojson")[0])
    except ValueError:
        continue  # Skip if filename doesn't contain a valid ID

    clip = gpd.read_file(path)
    if clip.empty:
        continue  # Skip empty files

    # Compute intersection area between clip and true footprint (no buffer)
    footprint = edf.loc[edifi_id].geometry
    inside = clip.geometry.intersection(footprint)
    area_clip = inside.unary_union.area
    area_tot = footprint_area.get(edifi_id, 0)

    if area_tot <= 0:
        continue  # Avoid division by zero

    cov_ratio = area_clip / area_tot  # Calculate coverage ratio

    # Filter out clips that are too small or too large
    if cov_ratio < MIN_COV or cov_ratio > MAX_COV:
        print(f"Skipping {fname}: coverage {cov_ratio:.1%} outside [{MIN_COV:.0%},{MAX_COV:.0%}]")
        continue

    # Store valid clips
    records.setdefault(edifi_id, []).append((cov_ratio, path))

# ─────────────────────────────────────────────────────
# 4) Report buildings with multiple valid clips

print("\n=== Duplicate clips per building ===")
for eid, lst in records.items():
    if len(lst) > 1:
        print(f"ED{eid} has {len(lst)} valid clips:")
        for cov, p in lst:
            print(f"  • {os.path.basename(p)} → {cov:.1%}")
print("=== End duplicate report ===\n")

# ─────────────────────────────────────────────────────
# 5) Select the best clip (highest coverage) for each building

for eid, lst in records.items():
    best_cov, best_path = max(lst, key=lambda x: x[0])  # Choose clip with max coverage
    out_name = os.path.basename(best_path)
    dst = os.path.join(OUTPUT_DIR, out_name)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(best_path, dst)  # Copy best clip to output directory

    print(f"ED{eid}: copied {out_name} (coverage {best_cov:.1%})")

print(f"\n All done ! Best per-building clips saved to: {OUTPUT_DIR}")
