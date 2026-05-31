import os
import json
import shutil

BASE_PATH = "."

IMAGES_PATH = os.path.join(BASE_PATH, "images")
SEGM_PATH = os.path.join(BASE_PATH, "segm")
DENSEPOSE_PATH = os.path.join(BASE_PATH, "densepose")

CAPTIONS_FILE = os.path.join(BASE_PATH, "captions.json")

SHAPE_FILE = os.path.join(BASE_PATH, "labels", "shape", "shape_anno_all.txt")
FABRIC_FILE = os.path.join(BASE_PATH, "labels", "texture", "fabric_ann.txt")
PATTERN_FILE = os.path.join(BASE_PATH, "labels", "texture", "pattern_ann.txt")

KEYPOINTS_LOC_FILE = os.path.join(BASE_PATH, "keypoints", "keypoints_loc.txt")
KEYPOINTS_VIS_FILE = os.path.join(BASE_PATH, "keypoints", "keypoints_vis.txt")

OUTPUT_PATH = os.path.join(BASE_PATH, "curated_dataset")
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_txt_dict(filepath):
    result = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) >= 2:
                sample_name = os.path.splitext(parts[0])[0]
                result[sample_name] = parts[1:]

    return result

print("\nLoading captions...")

with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    captions = json.load(f)

print(f"Captions loaded: {len(captions)}")

shape_dict = load_txt_dict(SHAPE_FILE)
fabric_dict = load_txt_dict(FABRIC_FILE)
pattern_dict = load_txt_dict(PATTERN_FILE)

keypoints_loc_dict = load_txt_dict(KEYPOINTS_LOC_FILE)
keypoints_vis_dict = load_txt_dict(KEYPOINTS_VIS_FILE)

print(f"Shape labels loaded: {len(shape_dict)}")
print(f"Fabric labels loaded: {len(fabric_dict)}")
print(f"Pattern labels loaded: {len(pattern_dict)}")
print(f"Keypoints loc loaded: {len(keypoints_loc_dict)}")
print(f"Keypoints vis loaded: {len(keypoints_vis_dict)}")

image_extensions = [".jpg", ".png", ".jpeg"]

image_map = {}

for fname in os.listdir(IMAGES_PATH):
    name, ext = os.path.splitext(fname)

    if ext.lower() in image_extensions:
        image_map[name] = os.path.join(IMAGES_PATH, fname)

image_ids = sorted(image_map.keys())

print(f"\nTotal images found: {len(image_ids)}")

valid_samples = 0
skipped_samples = 0

image_ids = image_ids[:20]

for idx, sample_id in enumerate(image_ids):

    print(f"\n[{idx + 1}/{len(image_ids)}] Checking: {sample_id}")

    caption_text = None

    for ext in image_extensions:
        key = sample_id + ext

        if key in captions:
            caption_text = captions[key]
            break

    segm_path = os.path.join(
        SEGM_PATH,
        sample_id + "_segm.png"
    )

    densepose_path = os.path.join(
        DENSEPOSE_PATH,
        sample_id + "_densepose.png"
    )

    all_exist = (
        caption_text is not None
        and os.path.exists(segm_path)
        and os.path.exists(densepose_path)
        and sample_id in shape_dict
        and sample_id in fabric_dict
        and sample_id in pattern_dict
        and sample_id in keypoints_loc_dict
        and sample_id in keypoints_vis_dict
    )

    if not all_exist:
        skipped_samples += 1
        continue

    sample_folder = os.path.join(
        OUTPUT_PATH,
        sample_id
    )

    os.makedirs(
        sample_folder,
        exist_ok=True
    )

    shutil.copy(
        image_map[sample_id],
        os.path.join(
            sample_folder,
            "image" + os.path.splitext(image_map[sample_id])[1]
        )
    )

    shutil.copy(
        segm_path,
        os.path.join(sample_folder, "segm.png")
    )

    shutil.copy(
        densepose_path,
        os.path.join(sample_folder, "densepose.png")
    )

    with open(
        os.path.join(sample_folder, "caption.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(caption_text)

    with open(
        os.path.join(sample_folder, "shape.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(" ".join(shape_dict[sample_id]))

    with open(
        os.path.join(sample_folder, "fabric.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(" ".join(fabric_dict[sample_id]))

    with open(
        os.path.join(sample_folder, "pattern.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(" ".join(pattern_dict[sample_id]))

    with open(
        os.path.join(sample_folder, "keypoints_loc.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(" ".join(keypoints_loc_dict[sample_id]))

    with open(
        os.path.join(sample_folder, "keypoints_vis.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(" ".join(keypoints_vis_dict[sample_id]))

    valid_samples += 1

    print(f"VALID: {sample_id}")

print("\n" + "=" * 60)
print("CURATION COMPLETE")
print("=" * 60)
print(f"Valid samples:   {valid_samples}")
print(f"Skipped samples: {skipped_samples}")
print(f"Output folder:   {os.path.abspath(OUTPUT_PATH)}")
print("=" * 60)