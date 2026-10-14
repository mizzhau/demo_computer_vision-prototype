import os
import sys
import shutil
import hashlib
import random
from pathlib import Path
import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_COLOR_DIR,
    RAW_SEG_DIR,
    SUBSET_DIR,
    SAMPLE_DIR,
    MANIFEST_PATH,
    RANDOM_SEED,
    SAMPLES_PER_CLASS,
    MIN_SAMPLES_PER_CLASS,
    TARGET_CLASSES
)


def compute_sha1(file_path: Path) -> str:
    hasher = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_group_id(file_name: str) -> str:
    if "___" in file_name:
        return file_name.split("___")[0]
    return file_name.split(".")[0]


def prepare_dataset():
    random.seed(RANDOM_SEED)

    print("Checking dataset availability...")
    if not RAW_COLOR_DIR.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {RAW_COLOR_DIR}")

    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    seen_hashes = set()

    for label_id, class_name in enumerate(TARGET_CLASSES):
        color_class_dir = RAW_COLOR_DIR / class_name
        seg_class_dir = RAW_SEG_DIR / class_name

        if not color_class_dir.exists():
            raise FileNotFoundError(f"Lớp {class_name} không tồn tại trong {RAW_COLOR_DIR}")

        image_files = sorted([
            f for f in color_class_dir.glob("*.*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        total_available = len(image_files)
        print(f"Class '{class_name}': {total_available} images found.")
        if total_available < MIN_SAMPLES_PER_CLASS:
            raise ValueError(
                f"Lớp '{class_name}' chỉ có {total_available} ảnh, không đạt yêu cầu tối thiểu {MIN_SAMPLES_PER_CLASS}!"
            )

        sampled_files = random.sample(
            image_files, min(SAMPLES_PER_CLASS, total_available)
        )

        dest_color_dir = SUBSET_DIR / "color" / class_name
        dest_seg_dir = SUBSET_DIR / "segmented" / class_name
        dest_color_dir.mkdir(parents=True, exist_ok=True)
        dest_seg_dir.mkdir(parents=True, exist_ok=True)

        dest_sample_dir = SAMPLE_DIR / class_name
        dest_sample_dir.mkdir(parents=True, exist_ok=True)

        groups = {}
        for idx, src_img in enumerate(sampled_files):
            grp = extract_group_id(src_img.name)
            groups.setdefault(grp, []).append(src_img)

        group_keys = list(groups.keys())
        random.shuffle(group_keys)

        n_total = len(sampled_files)
        n_train_target = int(n_total * 0.70)
        n_val_target = int(n_total * 0.15)

        split_assignments = {}
        curr_train, curr_val, curr_test = 0, 0, 0

        for grp in group_keys:
            grp_len = len(groups[grp])
            if curr_train + grp_len <= n_train_target or (curr_val >= n_val_target and curr_test >= (n_total - n_train_target - n_val_target)):
                for img_file in groups[grp]:
                    split_assignments[img_file.name] = "train"
                curr_train += grp_len
            elif curr_val + grp_len <= n_val_target:
                for img_file in groups[grp]:
                    split_assignments[img_file.name] = "val"
                curr_val += grp_len
            else:
                for img_file in groups[grp]:
                    split_assignments[img_file.name] = "test"
                curr_test += grp_len

        for idx, src_img in enumerate(sampled_files):
            dst_color = dest_color_dir / src_img.name
            shutil.copy2(src_img, dst_color)

            if idx < 20:
                shutil.copy2(src_img, dest_sample_dir / src_img.name)

            src_seg = seg_class_dir / src_img.name
            dst_seg = None
            if src_seg.exists():
                dst_seg = dest_seg_dir / src_img.name
                shutil.copy2(src_seg, dst_seg)

            with Image.open(src_img) as pil_img:
                w, h = pil_img.size

            file_hash = compute_sha1(src_img)
            seen_hashes.add(file_hash)

            rel_color_path = f"data/plantvillage_subset/color/{class_name}/{src_img.name}"
            rel_seg_path = (
                f"data/plantvillage_subset/segmented/{class_name}/{src_img.name}"
                if dst_seg else None
            )

            records.append({
                "image_id": f"{class_name}_{src_img.stem}",
                "path": rel_color_path,
                "seg_path": rel_seg_path,
                "label": class_name,
                "label_id": label_id,
                "split": split_assignments.get(src_img.name, "train"),
                "group_id": extract_group_id(src_img.name),
                "width": w,
                "height": h,
                "sha1": file_hash
            })

    manifest_df = pd.DataFrame(records)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")

    print(f"\nManifest generated successfully at {MANIFEST_PATH}")
    print(f"Total images in subset: {len(manifest_df)}")
    print("\nSplit distribution:")
    print(manifest_df.groupby(["label", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    prepare_dataset()
