"""Create reproducible, duplicate-safe train/validation/test image splits."""

import argparse
import csv
import hashlib
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image


EXPECTED_COUNTS = {
    "Blight": 1146,
    "Common_Rust": 1306,
    "Gray_Leaf_Spot": 574,
    "Healthy": 1162,
}
SPLIT_PROPORTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
INSPECTION_REPORT_NAME = "dataset_inspection_report.txt"
MANIFEST_NAME = "image_split_manifest.csv"
SPLIT_REPORT_NAME = "dataset_split_report.txt"


def file_sha256(file_path):
    """Hash a file without loading the complete file into memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_sha256(image):
    """Hash decoded pixels using the same method as inspect_dataset.py."""
    rgba_image = image.convert("RGBA")
    digest = hashlib.sha256()
    digest.update(f"{rgba_image.width}x{rgba_image.height}:".encode("ascii"))
    digest.update(rgba_image.tobytes())
    return digest.hexdigest()


def read_report_duplicate_groups(report_path, source_directory):
    """Read duplicate groups and verify the inspection report matches this dataset."""
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Required duplicate inspection report is missing: {report_path}"
        )

    report_text = report_path.read_text(encoding="utf-8")
    expected_source_line = f"Dataset directory inspected: {source_directory.resolve()}"
    if expected_source_line not in report_text:
        raise ValueError(
            "The inspection report does not identify the selected source directory. "
            "Re-run training/inspect_dataset.py for this dataset before preparing splits."
        )

    marker = "Duplicate groups (decoded-pixel matches):"
    if marker not in report_text:
        raise ValueError(
            "The inspection report does not contain decoded-pixel duplicate groups. "
            "Re-run training/inspect_dataset.py before preparing splits."
        )

    section = report_text.split(marker, maxsplit=1)[1].split(
        "\nPer-class image properties", maxsplit=1
    )[0]
    duplicate_groups = []
    for line in section.splitlines():
        stripped_line = line.strip()
        if not stripped_line.startswith("- "):
            continue
        group_paths = [
            source_directory / Path(path_text.strip())
            for path_text in stripped_line[2:].split("|")
        ]
        if len(group_paths) < 2:
            raise ValueError(
                f"Malformed duplicate entry in inspection report: {line}"
            )
        missing_paths = [path for path in group_paths if not path.is_file()]
        if missing_paths:
            raise ValueError(
                "The inspection report lists duplicate image paths that are missing "
                f"from the source dataset: {missing_paths}"
            )
        duplicate_groups.append(frozenset(group_paths))

    count_line_prefix = "Decoded-pixel duplicate groups: "
    count_line = next(
        (
            line
            for line in report_text.splitlines()
            if line.startswith(count_line_prefix)
        ),
        None,
    )
    if count_line is None:
        raise ValueError(
            "The inspection report is missing the decoded-pixel duplicate count."
        )
    try:
        reported_group_count = int(count_line[len(count_line_prefix):])
    except ValueError as error:
        raise ValueError(
            "The inspection report has an invalid decoded-pixel duplicate count."
        ) from error
    if reported_group_count != len(duplicate_groups):
        raise ValueError(
            "The inspection report's duplicate group count does not match its "
            "duplicate path list. Re-run training/inspect_dataset.py."
        )

    return duplicate_groups


def scan_source_images(source_directory, inspection_groups):
    """Read and hash all source images, reporting unreadable files as errors."""
    class_directories = sorted(
        (path for path in source_directory.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    observed_class_names = {path.name for path in class_directories}
    expected_class_names = set(EXPECTED_COUNTS)
    if observed_class_names != expected_class_names:
        raise ValueError(
            "Source class directories differ from the inspected classes. "
            f"Expected {sorted(expected_class_names)}, found {sorted(observed_class_names)}."
        )

    image_records = []
    unreadable_images = []
    pixel_groups = defaultdict(list)

    for class_directory in class_directories:
        image_paths = sorted(
            (path for path in class_directory.rglob("*") if path.is_file()),
            key=lambda path: str(path).casefold(),
        )
        if not image_paths:
            raise ValueError(f"Class directory contains no files: {class_directory}")

        for image_path in image_paths:
            try:
                # Verify the full file, then load it to calculate its decoded-pixel hash.
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.load()
                    pixel_hash = decoded_pixel_sha256(image)
                record = {
                    "path": image_path,
                    "class_name": class_directory.name,
                    "pixel_hash": pixel_hash,
                }
                image_records.append(record)
                pixel_groups[pixel_hash].append(image_path)
            except (OSError, ValueError, SyntaxError) as error:
                unreadable_images.append((image_path, str(error)))

    if unreadable_images:
        details = "\n".join(
            f"  - {path}: {error}" for path, error in unreadable_images
        )
        raise ValueError(
            f"Found {len(unreadable_images)} unreadable image(s); none were skipped. "
            "Resolve them, then rerun preparation:\n" + details
        )

    actual_counts = Counter(record["class_name"] for record in image_records)
    if dict(actual_counts) != EXPECTED_COUNTS:
        raise ValueError(
            "Source image counts do not match the confirmed inspection counts. "
            f"Expected {EXPECTED_COUNTS}, found {dict(actual_counts)}."
        )

    detected_groups = {
        frozenset(paths)
        for paths in pixel_groups.values()
        if len(paths) > 1
    }
    if detected_groups != set(inspection_groups):
        report_only = set(inspection_groups) - detected_groups
        new_since_report = detected_groups - set(inspection_groups)
        raise ValueError(
            "Duplicate detection differs from the existing inspection report. "
            f"Groups no longer detected: {len(report_only)}; "
            f"new groups not in report: {len(new_since_report)}. "
            "Re-run training/inspect_dataset.py and review the updated report."
        )

    return image_records, detected_groups


def group_images(image_records):
    """Keep images with identical decoded pixels together as one assignment unit."""
    grouped = defaultdict(list)
    for record in image_records:
        grouped[record["pixel_hash"]].append(record)
    return list(grouped.values())


def split_cost(counts_by_class, total_counts, class_targets, total_targets):
    """Measure deviation from the target class and total split proportions."""
    cost = 0.0
    for class_name, targets in class_targets.items():
        class_total = sum(targets.values())
        for split_name, target_count in targets.items():
            difference = counts_by_class[class_name][split_name] - target_count
            cost += (difference / class_total) ** 2
    overall_total = sum(total_targets.values())
    for split_name, target_count in total_targets.items():
        difference = total_counts[split_name] - target_count
        cost += (difference / overall_total) ** 2
    return cost


def assign_groups_to_splits(groups):
    """Assign duplicate groups using a seeded greedy stratification objective."""
    rng = random.Random(RANDOM_SEED)
    shuffled_groups = list(groups)
    rng.shuffle(shuffled_groups)
    shuffled_groups.sort(key=len, reverse=True)

    class_totals = Counter(
        record["class_name"] for group in groups for record in group
    )
    total_images = sum(class_totals.values())
    class_targets = {
        class_name: {
            split_name: class_total * proportion
            for split_name, proportion in SPLIT_PROPORTIONS.items()
        }
        for class_name, class_total in class_totals.items()
    }
    total_targets = {
        split_name: total_images * proportion
        for split_name, proportion in SPLIT_PROPORTIONS.items()
    }
    counts_by_class = {
        class_name: {split_name: 0 for split_name in SPLIT_PROPORTIONS}
        for class_name in class_totals
    }
    total_counts = {split_name: 0 for split_name in SPLIT_PROPORTIONS}
    group_assignments = {}

    for group in shuffled_groups:
        group_counts = Counter(record["class_name"] for record in group)
        candidate_scores = []
        for split_name in SPLIT_PROPORTIONS:
            for class_name, count in group_counts.items():
                counts_by_class[class_name][split_name] += count
            total_counts[split_name] += len(group)

            candidate_scores.append(
                (split_cost(counts_by_class, total_counts, class_targets, total_targets),
                 split_name)
            )

            for class_name, count in group_counts.items():
                counts_by_class[class_name][split_name] -= count
            total_counts[split_name] -= len(group)

        best_score = min(score for score, _ in candidate_scores)
        best_splits = [
            split_name
            for score, split_name in candidate_scores
            if abs(score - best_score) < 1e-12
        ]
        chosen_split = rng.choice(best_splits)
        group_assignments[id(group)] = chosen_split
        for class_name, count in group_counts.items():
            counts_by_class[class_name][chosen_split] += count
        total_counts[chosen_split] += len(group)

    return group_assignments


def verify_split_assignments(image_records, path_to_split, duplicate_groups):
    """Ensure every source image appears once and duplicate groups stay together."""
    record_paths = [record["path"].resolve() for record in image_records]
    if len(path_to_split) != len(record_paths):
        raise ValueError("An image was assigned more than once or was not assigned.")
    if set(path_to_split) != set(record_paths):
        raise ValueError("One or more readable images were not assigned to a split.")

    for duplicate_group in duplicate_groups:
        group_splits = {
            path_to_split[path.resolve()] for path in duplicate_group
        }
        if len(group_splits) != 1:
            raise ValueError(
                "A detected duplicate group crosses dataset splits: "
                + ", ".join(str(path) for path in sorted(duplicate_group))
            )
    return path_to_split


def write_manifest(manifest_path, image_records, path_to_split):
    """Write one CSV row per image, with its source and assigned split."""
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_filename", "original_path", "class_name", "split"],
        )
        writer.writeheader()
        for record in sorted(image_records, key=lambda item: str(item["path"]).casefold()):
            writer.writerow(
                {
                    "image_filename": record["path"].name,
                    "original_path": str(record["path"].resolve()),
                    "class_name": record["class_name"],
                    "split": path_to_split[record["path"].resolve()],
                }
            )


def verify_copies(image_records, path_to_split, source_directory, staging_directory):
    """Confirm copied bytes match sources and every image exists exactly once."""
    copied_source_paths = []
    for record in image_records:
        source_path = record["path"]
        split_name = path_to_split[source_path.resolve()]
        class_name = record["class_name"]
        relative_path = source_path.relative_to(source_directory / class_name)
        copied_path = staging_directory / split_name / class_name / relative_path
        if not copied_path.is_file():
            raise ValueError(f"Copied image is missing: {copied_path}")
        if file_sha256(source_path) != file_sha256(copied_path):
            raise ValueError(f"Copied image bytes differ from the source: {copied_path}")
        copied_source_paths.append(source_path.resolve())

    if len(copied_source_paths) != len(image_records):
        raise ValueError("Copied-image count does not match the assigned image count.")
    if len(copied_source_paths) != len(set(copied_source_paths)):
        raise ValueError("A source image was copied more than once.")


def write_split_report(
    report_path,
    source_directory,
    output_directory,
    image_records,
    path_to_split,
    duplicate_groups,
):
    """Write split counts, percentages, and pre-training review notes."""
    class_counts = Counter(record["class_name"] for record in image_records)
    split_counts = Counter(path_to_split.values())
    per_class_split = {
        class_name: Counter(
            path_to_split[record["path"].resolve()]
            for record in image_records
            if record["class_name"] == class_name
        )
        for class_name in EXPECTED_COUNTS
    }
    cross_class_groups = [
        group
        for group in duplicate_groups
        if len({path.relative_to(source_directory).parts[0] for path in group}) > 1
    ]
    total_images = len(image_records)
    proportions = ", ".join(
        f"{name} {share * 100:.0f}%" for name, share in SPLIT_PROPORTIONS.items()
    )

    lines = [
        "MAIZE DISEASE DATASET SPLIT REPORT",
        "=" * 36,
        "",
        f"Preparation date: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Source directory: {source_directory.resolve()}",
        f"Prepared dataset directory: {output_directory.resolve()}",
        f"Random seed: {RANDOM_SEED}",
        f"Target proportions: {proportions}",
        "All original images were copied without resizing or augmentation.",
        "Source image files were not moved, modified, or deleted.",
        "",
        "Image counts by split",
        "---------------------",
    ]
    for split_name, proportion in SPLIT_PROPORTIONS.items():
        count = split_counts[split_name]
        percentage = count / total_images * 100 if total_images else 0
        lines.append(
            f"{split_name}: {count} images ({percentage:.2f}%; target {proportion * 100:.0f}%)"
        )
    lines.extend(["", "Class distribution by split", "----------------------------"])
    for class_name in EXPECTED_COUNTS:
        class_total = class_counts[class_name]
        lines.append(f"{class_name}: {class_total} total")
        for split_name, proportion in SPLIT_PROPORTIONS.items():
            count = per_class_split[class_name][split_name]
            percentage = count / class_total * 100 if class_total else 0
            lines.append(
                f"  {split_name}: {count} images ({percentage:.2f}%; "
                f"target {proportion * 100:.0f}%)"
            )
    lines.extend(
        [
            "",
            "Validation checks",
            "-----------------",
            f"Source readable images assigned: {total_images}",
            "Unreadable images found: 0",
            f"Total images across splits: {sum(split_counts.values())}",
            "Every source image assigned exactly once: yes",
            "Every copied file byte-for-byte matches its source: yes",
            "Duplicate pixel groups cross splits: no",
            f"Duplicate groups checked against inspection report: {len(duplicate_groups)}",
            f"Duplicate groups spanning different class labels: {len(cross_class_groups)}",
            "",
            "Issues to review before training",
            "--------------------------------",
        ]
    )
    if cross_class_groups:
        lines.append(
            "- The inspection report identifies duplicate image content under "
            "different class labels. These groups were kept in one split each, but "
            "their conflicting labels should be reviewed before training."
        )
    else:
        lines.append("- No cross-class duplicate groups were reported.")
    lines.append(
        "- The folder labels were preserved as supplied; verify the meaning of "
        "'Blight' from authoritative dataset metadata before disease-specific claims."
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_dataset(source_directory, output_directory, evaluation_directory):
    """Validate, split, copy, and report without modifying source files."""
    source_directory = source_directory.resolve()
    output_directory = output_directory.resolve()
    evaluation_directory = evaluation_directory.resolve()
    evaluation_directory.mkdir(parents=True, exist_ok=True)

    if not source_directory.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {source_directory}")
    if output_directory.exists():
        raise FileExistsError(
            f"Processed dataset output already exists: {output_directory}. "
            "It was not overwritten. Choose a new output directory or move the "
            "existing output yourself before rerunning."
        )
    if source_directory == output_directory or source_directory in output_directory.parents:
        raise ValueError("Processed output must not be inside the original source dataset.")
    manifest_path = evaluation_directory / MANIFEST_NAME
    split_report = evaluation_directory / SPLIT_REPORT_NAME
    existing_reports = [
        path for path in (manifest_path, split_report) if path.exists()
    ]
    if existing_reports:
        raise FileExistsError(
            "Preparation report output already exists and will not be overwritten: "
            + ", ".join(str(path) for path in existing_reports)
        )

    inspection_report = evaluation_directory / INSPECTION_REPORT_NAME
    inspection_groups = read_report_duplicate_groups(
        inspection_report, source_directory
    )
    image_records, duplicate_groups = scan_source_images(
        source_directory, inspection_groups
    )
    groups = group_images(image_records)
    group_split_names = assign_groups_to_splits(groups)
    path_to_split = {
        record["path"].resolve(): group_split_names[id(group)]
        for group in groups
        for record in group
    }
    verify_split_assignments(image_records, path_to_split, duplicate_groups)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".maize_disease_processed_staging_",
        dir=output_directory.parent,
    ) as temporary_parent:
        staging_directory = Path(temporary_parent) / "dataset"
        staging_directory.mkdir()
        for record in image_records:
            source_path = record["path"]
            split_name = path_to_split[source_path.resolve()]
            relative_path = source_path.relative_to(
                source_directory / record["class_name"]
            )
            destination_path = (
                staging_directory
                / split_name
                / record["class_name"]
                / relative_path
            )
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)

        verify_copies(
            image_records,
            path_to_split,
            source_directory,
            staging_directory,
        )

        write_manifest(
            manifest_path,
            image_records,
            path_to_split,
        )

        # Verify the manifest describes each source image once before publishing copies.
        with manifest_path.open(newline="", encoding="utf-8") as csv_file:
            manifest_rows = list(csv.DictReader(csv_file))
        manifest_sources = [row["original_path"] for row in manifest_rows]
        if len(manifest_rows) != len(image_records):
            manifest_path.unlink()
            raise ValueError("Manifest row count does not match the source image count.")
        if len(manifest_sources) != len(set(manifest_sources)):
            manifest_path.unlink()
            raise ValueError("Manifest assigns a source image more than once.")

        write_split_report(
            split_report,
            source_directory,
            output_directory,
            image_records,
            path_to_split,
            duplicate_groups,
        )
        staging_directory.rename(output_directory)

    return split_report, manifest_path, path_to_split


def main():
    """Run dataset preparation with paths rooted at the project directory."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create reproducible, duplicate-safe maize disease data splits."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=project_root / "datasets" / "archive (3)" / "data",
        help="Original class-folder dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "datasets" / "maize_disease_processed",
        help="New directory for train/val/test copies.",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=project_root / "evaluation",
        help="Directory containing the inspection report and receiving split reports.",
    )
    arguments = parser.parse_args()

    try:
        split_report, manifest_path, path_to_split = prepare_dataset(
            source_directory=(
                arguments.source_dir
                if arguments.source_dir.is_absolute()
                else project_root / arguments.source_dir
            ),
            output_directory=(
                arguments.output_dir
                if arguments.output_dir.is_absolute()
                else project_root / arguments.output_dir
            ),
            evaluation_directory=(
                arguments.evaluation_dir
                if arguments.evaluation_dir.is_absolute()
                else project_root / arguments.evaluation_dir
            ),
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        return 1

    split_counts = Counter(path_to_split.values())
    print("Final image counts:")
    for split_name in SPLIT_PROPORTIONS:
        print(f"  {split_name}: {split_counts[split_name]}")
    print(f"Prepared dataset: {arguments.output_dir.resolve()}")
    print(f"Split report: {split_report.resolve()}")
    print(f"Image manifest: {manifest_path.resolve()}")
    print("No model training was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
