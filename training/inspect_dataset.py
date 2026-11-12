"""Inspect a maize disease image dataset without changing its source files."""

import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


DATASET_PROFILES = {
    "disease": {
        "label": "maize disease",
        "expected_classes": {"healthy", "common rust", "gray leaf spot", "blight"},
        "dataset_directory": Path("datasets") / "maize_disease",
        "default_directory": Path("datasets") / "archive (3)" / "data",
        "output_directory": Path("outputs"),
        "report_path": Path("evaluation") / "dataset_inspection_report.txt",
        "report_title": "MAIZE DISEASE DATASET INSPECTION REPORT",
        "chart_title": "Maize disease dataset: images per class",
    },
    "nutrient": {
        "label": "maize nutrient deficiency",
        "expected_classes": {
            "helathy",
            "magnessium",
            "nitrogen",
            "phosphorous",
            "pottasium",
        },
        "dataset_directory": Path("datasets") / "Maize Nutrient Deficiency Dataset",
        "default_directory": Path("datasets") / "Maize Nutrient Deficiency Dataset",
        "output_directory": Path("outputs") / "nutrient_inspection",
        "report_path": Path("evaluation") / "nutrient" / "dataset_inspection_report.txt",
        "report_title": "MAIZE NUTRIENT DEFICIENCY DATASET INSPECTION REPORT",
        "chart_title": "Maize nutrient deficiency dataset: images per class",
    },
}
SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
SPLIT_DIRECTORY_NAMES = {"train", "training", "val", "valid", "validation", "test"}
SAMPLES_PER_CLASS = 4
SAMPLE_SIZE = (240, 180)


def normalize_class_name(name):
    """Normalize separators and capitalization for label comparisons."""
    return " ".join(name.replace("_", " ").replace("-", " ").split()).casefold()


def choose_dataset_directory(project_root, requested_directory, dataset_type="disease"):
    """Choose one dataset root without falling back across dataset types."""
    if requested_directory is not None:
        dataset_directory = requested_directory
        if not dataset_directory.is_absolute():
            dataset_directory = project_root / dataset_directory
        if not dataset_directory.is_dir():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {dataset_directory}"
            )
        return dataset_directory

    profile = DATASET_PROFILES[dataset_type]
    for relative_path in (
        profile["dataset_directory"],
        profile["default_directory"],
    ):
        discovered_directory = project_root / relative_path
        if discovered_directory.is_dir():
            return discovered_directory

    raise FileNotFoundError(
        f"Could not find the {profile['label']} dataset at "
        f"{project_root / profile['dataset_directory']} or "
        f"{project_root / profile['default_directory']}. "
        "Pass --dataset-dir to select its directory."
    )


def sha256_file(file_path):
    """Return a file's SHA-256 digest using bounded memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_digest(image):
    """Return a digest of decoded pixels, including image dimensions."""
    rgba_image = image.convert("RGBA")
    digest = hashlib.sha256()
    digest.update(f"{rgba_image.width}x{rgba_image.height}:".encode("ascii"))
    digest.update(rgba_image.tobytes())
    return digest.hexdigest()


def write_class_distribution(output_directory, class_counts):
    """Save class counts and percentages as a CSV file."""
    total = sum(class_counts.values())
    csv_path = output_directory / "class_distribution.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class_name", "image_count", "percentage"])
        for class_name, count in sorted(class_counts.items(), key=lambda item: item[0].casefold()):
            percentage = (count / total * 100) if total else 0
            writer.writerow([class_name, count, f"{percentage:.2f}"])
    return csv_path


def create_class_chart(output_directory, class_counts, chart_title):
    """Create a bar chart of image counts in each class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    class_names = list(class_counts)
    counts = [class_counts[name] for name in class_names]
    figure_width = max(7, len(class_names) * 2)
    figure, axis = plt.subplots(figsize=(figure_width, 4.5))
    bars = axis.bar(class_names, counts, color="#4c956c")
    axis.set_title(chart_title)
    axis.set_ylabel("Image count")
    axis.set_xlabel("Observed class folder")
    axis.tick_params(axis="x", labelrotation=20)
    axis.bar_label(bars, padding=3)
    figure.tight_layout()
    chart_path = output_directory / "class_distribution.png"
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)
    return chart_path


def create_sample_sheet(output_directory, class_samples):
    """Save a contact sheet of a few readable images from every class."""
    label_height = 42
    columns = SAMPLES_PER_CLASS
    rows = max(1, len(class_samples))
    cell_width, cell_height = SAMPLE_SIZE
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * (cell_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for row_index, (class_name, samples) in enumerate(class_samples.items()):
        for column_index, file_path in enumerate(samples[:SAMPLES_PER_CLASS]):
            try:
                with Image.open(file_path) as source_image:
                    image = ImageOps.exif_transpose(source_image).convert("RGB")
                    image.thumbnail(SAMPLE_SIZE)
                    x = column_index * cell_width
                    y = row_index * (cell_height + label_height)
                    image_x = x + (cell_width - image.width) // 2
                    image_y = y + (cell_height - image.height) // 2
                    sheet.paste(image, (image_x, image_y))
                    label = f"{class_name}: {file_path.name}"
                    draw.text((x + 4, y + cell_height + 3), label[:38], fill="black")
            except (OSError, ValueError) as error:
                print(
                    f"Could not add sample image {file_path}: {error}",
                    file=sys.stderr,
                )

    sample_path = output_directory / "image_samples.png"
    sheet.save(sample_path)
    return sample_path


def inspect_dataset(
    dataset_directory,
    output_directory,
    project_root,
    dataset_type="disease",
):
    """Inspect folders and images, then write a report and visual summaries."""
    profile = DATASET_PROFILES[dataset_type]
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = project_root / profile["report_path"]
    report_directory = report_path.parent
    report_directory.mkdir(parents=True, exist_ok=True)

    all_entries = sorted(dataset_directory.rglob("*"), key=lambda path: str(path).casefold())
    all_files = [path for path in all_entries if path.is_file()]
    all_directories = [path for path in all_entries if path.is_dir()]

    unsupported_files = [
        path for path in all_files if path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS
    ]
    image_files = [path for path in all_files if path not in unsupported_files]

    class_directories = sorted(
        (
            path
            for path in dataset_directory.iterdir()
            if path.is_dir() and normalize_class_name(path.name) not in SPLIT_DIRECTORY_NAMES
        ),
        key=lambda path: path.name.casefold(),
    )
    class_counts = {directory.name: 0 for directory in class_directories}
    class_image_files = defaultdict(list)

    for directory in class_directories:
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS:
                class_image_files[directory.name].append(file_path)
        class_image_files[directory.name].sort(key=lambda path: str(path).casefold())
        class_counts[directory.name] = len(class_image_files[directory.name])

    split_directories = [
        path
        for path in all_directories
        if normalize_class_name(path.name) in SPLIT_DIRECTORY_NAMES
    ]
    split_names = sorted({normalize_class_name(path.name) for path in split_directories})
    split_structure_found = bool(split_directories)

    image_format_counts = Counter()
    image_dimension_counts = Counter()
    per_class_formats = defaultdict(Counter)
    per_class_dimensions = defaultdict(Counter)
    corrupt_images = []
    byte_duplicate_groups = defaultdict(list)
    pixel_duplicate_groups = defaultdict(list)
    class_samples = {}

    for class_name, file_paths in class_image_files.items():
        class_samples[class_name] = []
        for file_path in file_paths:
            try:
                file_digest = sha256_file(file_path)
                byte_duplicate_groups[file_digest].append(file_path)

                with Image.open(file_path) as image:
                    image.verify()
                with Image.open(file_path) as image:
                    image.load()
                    image_format = image.format or "Unknown"
                    dimensions = f"{image.width}x{image.height}"
                    decoded_digest = pixel_digest(image)

                image_format_counts[image_format] += 1
                image_dimension_counts[dimensions] += 1
                per_class_formats[class_name][image_format] += 1
                per_class_dimensions[class_name][dimensions] += 1
                pixel_duplicate_groups[decoded_digest].append(file_path)

                if len(class_samples[class_name]) < SAMPLES_PER_CLASS:
                    class_samples[class_name].append(file_path)
            except (OSError, ValueError, SyntaxError) as error:
                corrupt_images.append((file_path, str(error)))

    byte_duplicates = [
        paths for paths in byte_duplicate_groups.values() if len(paths) > 1
    ]
    pixel_duplicates = [
        paths for paths in pixel_duplicate_groups.values() if len(paths) > 1
    ]
    cross_class_pixel_duplicates = [
        paths
        for paths in pixel_duplicates
        if len({path.relative_to(dataset_directory).parts[0] for path in paths}) > 1
    ]
    byte_duplicate_file_count = sum(len(paths) - 1 for paths in byte_duplicates)
    pixel_duplicate_file_count = sum(len(paths) - 1 for paths in pixel_duplicates)

    class_distribution_path = write_class_distribution(output_directory, class_counts)
    chart_path = create_class_chart(
        output_directory,
        class_counts,
        profile["chart_title"],
    )
    sample_path = create_sample_sheet(output_directory, class_samples)

    observed_normalized = {normalize_class_name(name) for name in class_counts}
    expected_classes = profile["expected_classes"]
    missing_expected = sorted(expected_classes - observed_normalized)
    unexpected_observed = sorted(observed_normalized - expected_classes)
    count_values = [count for count in class_counts.values() if count > 0]
    total_class_images = sum(class_counts.values())
    if count_values:
        smallest_count = min(count_values)
        largest_count = max(count_values)
        imbalance_ratio = largest_count / smallest_count if smallest_count else 0
    else:
        smallest_count = 0
        largest_count = 0
        imbalance_ratio = 0

    report_lines = [
        profile["report_title"],
        "=" * len(profile["report_title"]),
        "",
        f"Inspection date: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Project directory: {project_root}",
        f"Dataset directory inspected: {dataset_directory.resolve()}",
        f"Expected dataset path exists: "
        f"{(project_root / profile['dataset_directory']).is_dir()}",
        "",
        "Scope and safety",
        "----------------",
        f"Only the selected {profile['label']} dataset directory was inspected.",
        "No original dataset files were moved, modified, or deleted.",
        "No model training was performed.",
        (
            "The separate maize disease dataset was not inspected or mixed in."
            if dataset_type == "nutrient"
            else "The separate maize nutrient deficiency dataset was not inspected or mixed in."
        ),
        "",
        "Folder structure",
        "----------------",
        f"Total nested directories: {len(all_directories)}",
        f"Total files of all formats: {len(all_files)}",
        "Directories:",
    ]
    report_lines.extend(
        f"  - {path.relative_to(dataset_directory)}" for path in all_directories
    )
    if not all_directories:
        report_lines.append("  (none)")
    report_lines.extend(
        [
            f"Train/validation/test-style split directories found: {split_structure_found}",
            f"Split directory names found: {', '.join(split_names) if split_names else 'none'}",
            "Dataset layout: "
            + (
                "separate split directories exist; inspect their layout before training."
                if split_structure_found
                else "flat class-folder layout; no separate train, validation, or test folders."
            ),
            "",
            "Class folders and counts",
            "------------------------",
        ]
    )

    for class_name, count in class_counts.items():
        percentage = (count / total_class_images * 100) if total_class_images else 0
        report_lines.append(f"{class_name}: {count} images ({percentage:.2f}%)")
    report_lines.extend(
        [
            f"Total supported-extension images in class folders: {total_class_images}",
            f"Total supported-extension files found anywhere under dataset root: {len(image_files)}",
            "",
            "Expected class-name check",
            "-------------------------",
            "Expected labels (normalized for capitalization/separators): "
            + ", ".join(sorted(expected_classes)),
            "Observed class-folder names: "
            + (", ".join(class_counts) if class_counts else "none"),
            f"Expected labels missing: {', '.join(missing_expected) if missing_expected else 'none'}",
            f"Unexpected labels: {', '.join(unexpected_observed) if unexpected_observed else 'none'}",
            (
                "The folder named 'Blight' is reported exactly as supplied. "
                "Its specific disease identity (including whether it is Northern Leaf Blight) "
                "cannot be established from the folder name alone."
                if dataset_type == "disease"
                else "Nutrient class folder names are preserved exactly as supplied; "
                "verify their spellings and biological label definitions before training."
            ),
            "",
            "Class imbalance",
            "----------------",
            f"Smallest non-empty class: {smallest_count}",
            f"Largest class: {largest_count}",
            f"Largest-to-smallest ratio: {imbalance_ratio:.2f}:1",
            (
                "Observation: class counts are noticeably uneven; the largest class has "
                "at least twice the images of the smallest class."
                if imbalance_ratio >= 2
                else "Observation: no class is at least twice as large as the smallest non-empty class."
            ),
            "",
            "File formats, dimensions, and readability",
            "------------------------------------------",
            f"Supported image-extension files found: {len(image_files)}",
            f"Unsupported file count: {len(unsupported_files)}",
            "Supported image formats decoded:",
        ]
    )
    report_lines.extend(
        f"  - {image_format}: {count}" for image_format, count in sorted(image_format_counts.items())
    )
    if not image_format_counts:
        report_lines.append("  (none)")
    report_lines.append("Image dimensions (width x height):")
    report_lines.append(f"Distinct image dimensions: {len(image_dimension_counts)}")
    if image_dimension_counts:
        most_common_dimensions, most_common_dimension_count = image_dimension_counts.most_common(1)[0]
        report_lines.append(
            f"Most common size: {most_common_dimensions} ({most_common_dimension_count} images)"
        )
    report_lines.extend(
        f"  - {dimensions}: {count}" for dimensions, count in sorted(image_dimension_counts.items())
    )
    if not image_dimension_counts:
        report_lines.append("  (none)")
    report_lines.extend(
        [
            f"Corrupted or unreadable image count: {len(corrupt_images)}",
            "Corrupted/unreadable files:",
        ]
    )
    report_lines.extend(
        f"  - {path.relative_to(dataset_directory)}: {error}"
        for path, error in corrupt_images
    )
    if not corrupt_images:
        report_lines.append("  (none)")
    report_lines.extend(["Unsupported files:", *(
        f"  - {path.relative_to(dataset_directory)}" for path in unsupported_files
    )])
    if not unsupported_files:
        report_lines.append("  (none)")

    report_lines.extend(
        [
            "",
            "Duplicate images",
            "----------------",
            "Exact file duplicates are detected using SHA-256 over file bytes.",
            f"Duplicate file groups: {len(byte_duplicates)}",
            f"Duplicate file count beyond first copies: {byte_duplicate_file_count}",
            "Decoded-pixel duplicates compare pixel content after conversion to RGBA, "
            "so differently encoded copies can also be detected.",
            f"Decoded-pixel duplicate groups: {len(pixel_duplicates)}",
            f"Decoded-pixel duplicate count beyond first copies: {pixel_duplicate_file_count}",
            f"Duplicate groups appearing in more than one class: {len(cross_class_pixel_duplicates)}",
            "Duplicate groups (decoded-pixel matches):",
        ]
    )
    for duplicate_group in pixel_duplicates:
        report_lines.append(
            "  - " + " | ".join(
                str(path.relative_to(dataset_directory)) for path in duplicate_group
            )
        )
    if not pixel_duplicates:
        report_lines.append("  (none)")
    if any(image_format == "MPO" for image_format in image_format_counts):
        report_lines.append(
            "Format compatibility note: at least one .jpg/.jpeg file decoded as MPO; "
            "verify compatibility with the chosen training image loader."
        )

    report_lines.extend(["", "Per-class image properties", "--------------------------"])
    for class_name in class_counts:
        report_lines.append(f"{class_name}:")
        report_lines.append(f"  Formats: {dict(per_class_formats[class_name]) or 'none'}")
        report_lines.append(f"  Dimensions: {dict(per_class_dimensions[class_name]) or 'none'}")

    report_lines.extend(
        [
            "",
            "Generated outputs",
            "-----------------",
            f"Class distribution CSV: {class_distribution_path.resolve()}",
            f"Class distribution chart: {chart_path.resolve()}",
            f"Image sample contact sheet: {sample_path.resolve()}",
            "",
            "Issues to resolve before training",
            "----------------------------------",
        ]
    )
    issues = []
    expected_directory = project_root / profile["dataset_directory"]
    if not expected_directory.is_dir():
        issues.append(
            f"The expected {profile['label']} path {expected_directory} is absent; "
            "the explicitly selected dataset path was used instead."
        )
    if split_structure_found:
        issues.append(
            "The dataset includes split-like directories; confirm their intended roles and "
            "avoid data leakage when defining train/validation/test splits."
        )
    else:
        issues.append(
            "No train/validation/test splits are present; a leakage-aware split must be planned."
        )
    if missing_expected:
        issues.append("One or more expected class names are missing.")
    if unexpected_observed:
        issues.append("Unexpected class folder names are present and need label verification.")
    if unsupported_files:
        issues.append("Unsupported file formats are present and should be reviewed.")
    if corrupt_images:
        issues.append("Unreadable images should be reviewed before training.")
    if pixel_duplicate_file_count:
        if cross_class_pixel_duplicates:
            issues.append(
                f"{len(cross_class_pixel_duplicates)} duplicate image group(s) occur across "
                "different class labels; review these potentially conflicting labels and "
                "keep duplicates out of separate data splits."
            )
        else:
            issues.append(
                "Duplicate image content should be reviewed to avoid split leakage."
            )
    if dataset_type == "disease":
        issues.append(
            "The meaning of the 'Blight' label is not proven by this inspection; verify it "
            "against authoritative dataset metadata before making a disease-specific claim."
        )
    report_lines.extend(f"- {issue}" for issue in issues)

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report_path, class_counts, len(corrupt_images), pixel_duplicate_file_count


def main():
    """Parse command-line options and start the inspection."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Inspect one maize dataset without changing source files."
    )
    parser.add_argument(
        "--dataset-type",
        choices=sorted(DATASET_PROFILES),
        default="disease",
        help="Dataset profile to inspect (default: disease)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dataset root (default: discovered within the selected dataset profile)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV and image outputs (default depends on dataset type)",
    )
    arguments = parser.parse_args()

    try:
        dataset_paths = {
            "Prepared maize disease dataset": project_root / "datasets" / "maize_disease_processed",
            "Archived maize disease source": project_root / "datasets" / "archive (3)" / "data",
            "Maize nutrient deficiency dataset": project_root / "datasets" / "Maize Nutrient Deficiency Dataset",
        }
        print("Detected dataset paths:")
        for label, path in dataset_paths.items():
            print(f"  {label}: {path.resolve()} (exists: {path.is_dir()})")

        dataset_directory = choose_dataset_directory(
            project_root,
            arguments.dataset_dir,
            arguments.dataset_type,
        )
        output_directory = arguments.output_dir
        if output_directory is None:
            output_directory = project_root / DATASET_PROFILES[arguments.dataset_type]["output_directory"]
        if not output_directory.is_absolute():
            output_directory = project_root / output_directory

        report_path, counts, corrupt_count, duplicate_count = inspect_dataset(
            dataset_directory,
            output_directory,
            project_root,
            arguments.dataset_type,
        )
    except (FileNotFoundError, OSError, ImportError) as error:
        parser.error(str(error))

    print(f"Dataset inspected: {dataset_directory.resolve()}")
    print(f"Dataset type: {arguments.dataset_type}")
    if arguments.dataset_type == "nutrient":
        print(
            "Nutrient-only inspection confirmed: only images under the selected "
            "nutrient dataset root were inspected."
        )
    print("No training was started.")
    print("Image count by class:")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count}")
    print(f"Corrupted/unreadable image count: {corrupt_count}")
    print(f"Duplicate decoded-image count beyond first copies: {duplicate_count}")
    print(f"Report: {report_path.resolve()}")
    print(f"Outputs: {output_directory.resolve()}")


if __name__ == "__main__":
    main()
