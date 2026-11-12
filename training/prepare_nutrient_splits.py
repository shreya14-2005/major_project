"""Decode, group, and stratify the nutrient dataset without copying its images."""

import argparse
import csv
import hashlib
import itertools
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from training.nutrient_config import (
    CLASS_FOLDER_TO_LABEL,
    CLASS_ORDER,
    DATASET_ROOT,
    EXPECTED_TOTAL_IMAGES,
    PHASH_MAX_HAMMING_DISTANCE,
    PROJECT_ROOT,
    RANDOM_SEED,
    SPLITS_ROOT,
    SPLIT_PROPORTIONS,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
OUTPUT_FILENAMES = {
    "all.csv",
    "train.csv",
    "val.csv",
    "test.csv",
    "split_summary.csv",
    "duplicate_groups.csv",
    "near_duplicate_pairs.csv",
    "split_report.txt",
}


class DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value):
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_hash(image):
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{normalized.width}x{normalized.height}:".encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def perceptual_hash(image):
    grayscale = ImageOps.exif_transpose(image).convert("L").resize(
        (32, 32), Image.Resampling.LANCZOS
    )
    coefficients = cv2.dct(np.asarray(grayscale, dtype=np.float32))[:8, :8].flatten()
    threshold = float(np.median(coefficients[1:]))
    value = 0
    for bit, coefficient in enumerate(coefficients):
        if coefficient > threshold:
            value |= 1 << bit
    return value


def scan_dataset():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Nutrient dataset not found: {DATASET_ROOT}")

    expected_folders = set(CLASS_FOLDER_TO_LABEL)
    observed_folders = {
        path.name for path in DATASET_ROOT.iterdir() if path.is_dir()
    }
    if observed_folders != expected_folders:
        raise ValueError(
            f"Expected class folders {sorted(expected_folders)}, "
            f"found {sorted(observed_folders)}."
        )

    records = []
    for folder in CLASS_FOLDER_TO_LABEL:
        folder_path = DATASET_ROOT / folder
        files = sorted(path for path in folder_path.rglob("*") if path.is_file())
        if not files:
            raise ValueError(f"Class folder is empty: {folder_path}")
        for path in files:
            if path.suffix.casefold() not in IMAGE_EXTENSIONS:
                raise ValueError(
                    f"Unexpected non-JPEG file in nutrient class folder: {path}"
                )
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    oriented = ImageOps.exif_transpose(image)
                    width, height = oriented.size
                    if width <= 0 or height <= 0:
                        raise ValueError(f"Invalid image dimensions: {width}x{height}")
                    record = {
                        "image_path": str(path.resolve()),
                        "original_folder": folder,
                        "label": CLASS_FOLDER_TO_LABEL[folder],
                        "file_sha256": sha256_file(path),
                        "pixel_sha256": decoded_pixel_hash(image),
                        "phash": perceptual_hash(image),
                    }
                    records.append(record)
            except (OSError, SyntaxError) as error:
                raise ValueError(f"Cannot decode image {path}: {error}") from error

    if len(records) != EXPECTED_TOTAL_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_IMAGES} images, found {len(records)}."
        )
    counts = Counter(record["original_folder"] for record in records)
    if counts != Counter(
        {"Helathy": 72, "Magnessium": 123, "Nitrogen": 99,
         "Phosphorous": 113, "Pottasium": 56}
    ):
        raise ValueError(f"Class counts differ from the inspected dataset: {dict(counts)}")
    return records


def build_groups(records):
    """Connect exact and pHash-similar images so components cannot cross splits."""
    disjoint_set = DisjointSet(len(records))
    relation_rows = []
    hash_members = defaultdict(list)

    for index, record in enumerate(records):
        for hash_type, hash_value in (
            ("file_sha256", record["file_sha256"]),
            ("decoded_pixel_sha256", record["pixel_sha256"]),
        ):
            for other_index in hash_members[(hash_type, hash_value)]:
                disjoint_set.union(index, other_index)
                relation_rows.append(
                    {
                        "relation": hash_type,
                        "index_a": other_index,
                        "index_b": index,
                        "hamming_distance": "",
                    }
                )
            hash_members[(hash_type, hash_value)].append(index)

    near_pair_rows = []
    for left_index, left in enumerate(records):
        for right_index in range(left_index + 1, len(records)):
            right = records[right_index]
            distance = (left["phash"] ^ right["phash"]).bit_count()
            if distance <= PHASH_MAX_HAMMING_DISTANCE:
                disjoint_set.union(left_index, right_index)
                relation = {
                    "relation": "perceptual_hash",
                    "index_a": left_index,
                    "index_b": right_index,
                    "hamming_distance": distance,
                }
                relation_rows.append(relation)
                near_pair_rows.append(
                    {
                        "image_path_a": left["image_path"],
                        "label_a": left["label"],
                        "image_path_b": right["image_path"],
                        "label_b": right["label"],
                        "hamming_distance": distance,
                        "cross_class": left["label"] != right["label"],
                    }
                )

    components = defaultdict(list)
    for index, record in enumerate(records):
        components[disjoint_set.find(index)].append((index, record))
    groups = sorted(
        components.values(),
        key=lambda group: min(item[1]["image_path"].casefold() for item in group),
    )
    relation_by_root = defaultdict(set)
    for row in relation_rows:
        root = disjoint_set.find(row["index_a"])
        relation_by_root[root].add(row["relation"])

    group_id_by_path = {}
    group_rows = []
    record_group_rows = []
    for group_number, group in enumerate(groups, 1):
        group_id = f"related_{group_number:04d}"
        labels = {record["label"] for _, record in group}
        if len(labels) != 1:
            paths = [record["image_path"] for _, record in group]
            raise ValueError(
                "A duplicate/near-duplicate component contains conflicting class "
                f"labels and cannot be safely stratified: {paths}"
            )
        for _, record in group:
            group_id_by_path[record["image_path"]] = group_id
            record_group_rows.append((group_id, record))
        if len(group) > 1:
            group_rows.append(
                {
                    "group_id": group_id,
                    "label": next(iter(labels)),
                    "member_count": len(group),
                    "relations": ";".join(sorted(relation_by_root[disjoint_set.find(group[0][0])])),
                    "image_paths": " | ".join(
                        sorted(record["image_path"] for _, record in group)
                    ),
                }
            )

    return groups, group_id_by_path, group_rows, near_pair_rows


def rounded_targets(count):
    raw = [count * SPLIT_PROPORTIONS[name] for name in ("train", "val", "test")]
    targets = [int(value) for value in raw]
    remainder = count - sum(targets)
    order = sorted(
        range(3), key=lambda index: (raw[index] - targets[index], -index), reverse=True
    )
    for index in order[:remainder]:
        targets[index] += 1
    return dict(zip(("train", "val", "test"), targets))


def assign_groups(records, groups, group_id_by_path, seed):
    rng = random.Random(seed)
    groups_by_label = defaultdict(list)
    lookup = {record["image_path"]: record for record in records}
    for group in groups:
        label = lookup[group[0][1]["image_path"]]["label"]
        groups_by_label[label].append(group)

    path_to_split = {}
    split_names = ("train", "val", "test")
    for label in CLASS_ORDER:
        class_groups = list(groups_by_label[label])
        class_records = [
            record for record in records if record["label"] == label
        ]
        targets = rounded_targets(len(class_records))
        rng.shuffle(class_groups)
        class_groups.sort(key=len, reverse=True)
        counts = {name: 0 for name in split_names}

        for group in class_groups:
            group_size = len(group)
            scores = []
            for split_name in split_names:
                candidate_count = counts[split_name] + group_size
                score = sum(
                    ((candidate_count if name == split_name else counts[name]) - targets[name])
                    ** 2
                    for name in split_names
                )
                scores.append((score, split_name))
            minimum = min(score for score, _ in scores)
            choices = [name for score, name in scores if score == minimum]
            selected = rng.choice(choices)
            counts[selected] += group_size
            for _, record in group:
                path_to_split[record["image_path"]] = selected

        if any(counts[name] == 0 for name in split_names):
            raise ValueError(f"Class {label} did not populate every split: {counts}")

    if len(path_to_split) != len(records):
        raise ValueError("Not every source image received one split assignment.")
    for group in groups:
        splits = {path_to_split[record["image_path"]] for _, record in group}
        if len(splits) != 1:
            raise ValueError("A duplicate/related-image group crosses data splits.")
    return path_to_split


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    existing = [SPLITS_ROOT / name for name in OUTPUT_FILENAMES if (SPLITS_ROOT / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Split outputs already exist; refusing to overwrite. Use --overwrite "
            "only when intentionally regenerating them."
        )

    records = scan_dataset()
    groups, group_id_by_path, duplicate_groups, near_pairs = build_groups(records)
    path_to_split = assign_groups(records, groups, group_id_by_path, args.seed)
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for record in records:
        manifest_rows.append(
            {
                "image_path": record["image_path"],
                "label": record["label"],
                "original_folder": record["original_folder"],
                "split": path_to_split[record["image_path"]],
                "related_group_id": group_id_by_path[record["image_path"]],
            }
        )
    manifest_rows.sort(key=lambda row: (row["split"], row["label"], row["image_path"]))
    manifest_columns = [
        "image_path",
        "label",
        "original_folder",
        "split",
        "related_group_id",
    ]
    write_csv(SPLITS_ROOT / "all.csv", manifest_columns, manifest_rows)
    for split_name in ("train", "val", "test"):
        write_csv(
            SPLITS_ROOT / f"{split_name}.csv",
            manifest_columns,
            [row for row in manifest_rows if row["split"] == split_name],
        )

    summary_rows = []
    for label in CLASS_ORDER:
        for split_name in ("train", "val", "test"):
            count = sum(
                row["label"] == label and row["split"] == split_name
                for row in manifest_rows
            )
            summary_rows.append(
                {
                    "label": label,
                    "split": split_name,
                    "image_count": count,
                    "class_total": sum(row["label"] == label for row in manifest_rows),
                    "split_percent_of_class": f"{count / sum(row['label'] == label for row in manifest_rows) * 100:.2f}",
                }
            )
    write_csv(
        SPLITS_ROOT / "split_summary.csv",
        ["label", "split", "image_count", "class_total", "split_percent_of_class"],
        summary_rows,
    )

    duplicate_rows = [
        {
            "group_id": row["group_id"],
            "label": row["label"],
            "member_count": row["member_count"],
            "relations": row["relations"],
            "image_paths": row["image_paths"],
        }
        for row in duplicate_groups
    ]
    write_csv(
        SPLITS_ROOT / "duplicate_groups.csv",
        ["group_id", "label", "member_count", "relations", "image_paths"],
        duplicate_rows,
    )
    write_csv(
        SPLITS_ROOT / "near_duplicate_pairs.csv",
        [
            "image_path_a",
            "label_a",
            "image_path_b",
            "label_b",
            "hamming_distance",
            "cross_class",
        ],
        near_pairs,
    )

    split_counts = Counter(row["split"] for row in manifest_rows)
    report_lines = [
        "NUTRIENT DATASET SPLIT REPORT",
        "=============================",
        f"Source: {DATASET_ROOT}",
        f"Manifest directory: {SPLITS_ROOT}",
        f"Image total: {len(records)}",
        f"Random seed: {args.seed}",
        "Target ratios: train 70%, validation 15%, test 15%",
        "Images are referenced by absolute path; source images were not copied or modified.",
        "Related groups use exact byte SHA-256, decoded-pixel SHA-256, and 64-bit DCT pHash "
        f"with Hamming distance <= {PHASH_MAX_HAMMING_DISTANCE}.",
        f"Near-duplicate candidate pairs: {len(near_pairs)}",
        f"Multi-image related groups: {len(duplicate_rows)}",
        "Every related group was verified to occur in exactly one split.",
        "",
        "Split totals:",
    ]
    for name in ("train", "val", "test"):
        report_lines.append(f"  {name}: {split_counts[name]}")
    report_lines.extend(["", "Class-wise split counts:"])
    for label in CLASS_ORDER:
        values = {
            name: sum(
                row["label"] == label and row["split"] == name for row in manifest_rows
            )
            for name in ("train", "val", "test")
        }
        report_lines.append(
            f"  {label}: train={values['train']}, val={values['val']}, test={values['test']}"
        )
    report_lines.extend(
        [
            "",
            "Labels are the supplied dataset folder labels with spelling-only normalization. "
            "They have not been independently scientifically verified.",
            "No test images or labels were used during split assignment beyond stratified "
            "count allocation; model selection must use validation data only.",
            "",
        ]
    )
    (SPLITS_ROOT / "split_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"Dataset: {DATASET_ROOT}")
    print(f"Seed: {args.seed}; images: {len(records)}; related groups: {len(duplicate_rows)}")
    for name in ("train", "val", "test"):
        print(f"{name}: {split_counts[name]}")
    print(f"Manifests: {SPLITS_ROOT}")


if __name__ == "__main__":
    main()
