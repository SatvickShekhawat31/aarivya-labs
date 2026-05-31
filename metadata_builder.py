import os
import json
import csv

CURATED_PATH = os.path.join(".", "curated_dataset")
OUTPUT_JSON = os.path.join(".", "metadata.json")
OUTPUT_CSV = os.path.join(".", "metadata_summary.csv")

SHAPE_REGIONS = [
    "upper_body", "lower_body", "sleeve", "neckline",
    "collar", "lapel", "placket", "pocket",
    "hem", "waistline", "cuff", "opening"
]

SHAPE_LABELS = {
    "0": "N/A",
    "1": "sleeveless",
    "2": "short_sleeve",
    "3": "medium_sleeve",
    "4": "long_sleeve",
    "5": "long_pants",
    "6": "short_pants",
    "7": "skirt",
    "8": "dress",
}

FABRIC_REGIONS = ["upper_body", "lower_body", "full_body"]

FABRIC_LABELS = {
    "0": "N/A",
    "1": "denim",
    "2": "cotton",
    "3": "leather",
    "4": "furry",
    "5": "knitted",
    "6": "chiffon",
    "7": "other",
}

PATTERN_REGIONS = ["upper_body", "lower_body", "full_body"]

PATTERN_LABELS = {
    "0": "N/A",
    "1": "solid",
    "2": "floral",
    "3": "plaid",
    "4": "stripe",
    "5": "print",
    "6": "graphic",
    "7": "other",
}

def parse_shape(values):
    result = {}

    for i, v in enumerate(values):
        region = SHAPE_REGIONS[i] if i < len(SHAPE_REGIONS) else f"region_{i}"
        result[region] = SHAPE_LABELS.get(v, v)

    return result

def parse_fabric(values):
    result = {}

    for i, v in enumerate(values):
        region = FABRIC_REGIONS[i] if i < len(FABRIC_REGIONS) else f"region_{i}"
        result[region] = FABRIC_LABELS.get(v, v)

    return result

def parse_pattern(values):
    result = {}

    for i, v in enumerate(values):
        region = PATTERN_REGIONS[i] if i < len(PATTERN_REGIONS) else f"region_{i}"
        result[region] = PATTERN_LABELS.get(v, v)

    return result

def parse_keypoints_loc(values):
    KEYPOINT_NAMES = [
        "head",
        "neck",
        "left_shoulder",
        "right_shoulder",
        "left_sleeve",
        "right_sleeve",
        "left_waistline",
        "right_waistline",
        "left_hem",
        "right_hem",
        "kp_10",
        "kp_11",
        "kp_12",
        "kp_13",
        "kp_14",
        "kp_15",
        "kp_16",
        "kp_17",
        "kp_18",
        "kp_19",
        "kp_20"
    ]

    coords = []

    for i in range(0, len(values) - 1, 2):
        kp_idx = i // 2

        name = (
            KEYPOINT_NAMES[kp_idx]
            if kp_idx < len(KEYPOINT_NAMES)
            else f"kp_{kp_idx}"
        )

        coords.append({
            "name": name,
            "x": int(values[i]),
            "y": int(values[i + 1])
        })

    return coords

def parse_keypoints_vis(values):
    return [int(v) for v in values]

print(f"\nScanning: {CURATED_PATH}")

sample_folders = sorted([
    d for d in os.listdir(CURATED_PATH)
    if os.path.isdir(os.path.join(CURATED_PATH, d))
])

print(f"Samples found: {len(sample_folders)}")

all_metadata = {}
csv_rows = []

for idx, sample_id in enumerate(sample_folders):

    folder = os.path.join(CURATED_PATH, sample_id)

    def read_txt(filename):
        path = os.path.join(folder, filename)

        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    caption = read_txt("caption.txt")

    shape_raw = read_txt("shape.txt").split()
    fabric_raw = read_txt("fabric.txt").split()
    pattern_raw = read_txt("pattern.txt").split()

    kp_loc_raw = read_txt("keypoints_loc.txt").split()
    kp_vis_raw = read_txt("keypoints_vis.txt").split()

    record = {
        "sample_id": sample_id,
        "files": {
            "image": "image.jpg",
            "densepose": "densepose.png",
            "segmentation": "segm.png",
            "caption": "caption.txt",
            "shape": "shape.txt",
            "fabric": "fabric.txt",
            "pattern": "pattern.txt",
            "keypoints_loc": "keypoints_loc.txt",
            "keypoints_vis": "keypoints_vis.txt"
        },
        "caption": caption,
        "style_attributes": {
            "shape": parse_shape(shape_raw),
            "fabric": parse_fabric(fabric_raw),
            "pattern": parse_pattern(pattern_raw)
        },
        "keypoints": {
            "locations": parse_keypoints_loc(kp_loc_raw),
            "visibility": parse_keypoints_vis(kp_vis_raw),
            "total": len(kp_vis_raw),
            "visible_count": sum(int(v) for v in kp_vis_raw)
        },
        "conditioning_signals": {
            "text": "caption",
            "pose": "densepose + keypoints",
            "garment_mask": "segmentation",
            "shape_class": "shape",
            "material": "fabric + pattern"
        }
    }

    all_metadata[sample_id] = record

    csv_rows.append({
        "sample_id": sample_id,
        "caption": caption,
        "upper_shape": parse_shape(shape_raw).get("upper_body", "N/A"),
        "lower_shape": parse_shape(shape_raw).get("lower_body", "N/A"),
        "upper_fabric": parse_fabric(fabric_raw).get("upper_body", "N/A"),
        "lower_fabric": parse_fabric(fabric_raw).get("lower_body", "N/A"),
        "upper_pattern": parse_pattern(pattern_raw).get("upper_body", "N/A"),
        "lower_pattern": parse_pattern(pattern_raw).get("lower_body", "N/A"),
        "visible_keypoints": sum(int(v) for v in kp_vis_raw)
    })

    if (idx + 1) % 1000 == 0:
        print(f"Processed {idx + 1}/{len(sample_folders)}...")

print(f"\nWriting {OUTPUT_JSON}...")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        all_metadata,
        f,
        indent=2,
        ensure_ascii=False
    )

print(f"Writing {OUTPUT_CSV}...")

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(csv_rows)

print("\n" + "=" * 60)
print("METADATA BUILD COMPLETE")
print("=" * 60)
print(f"metadata.json        → {len(all_metadata)} records")
print(f"metadata_summary.csv → {len(csv_rows)} rows")
print("=" * 60)
