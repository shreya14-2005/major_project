"""Read-only quality assessment for the maize nutrient image dataset.

Spelling-only folder corrections documented before defining the code mapping:
Helathy -> Healthy, Magnessium -> Magnesium, Phosphorous -> Phosphorus, and
Pottasium -> Potassium. Nitrogen is unchanged. These orthographic corrections
do not verify the biological labels or assert that an image depicts its folder
label.
"""

import csv
import hashlib
import itertools
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageOps


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "Maize Nutrient Deficiency Dataset"
OUTPUT_ROOT = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
EXPECTED_FOLDERS = (
    "Helathy",
    "Magnessium",
    "Nitrogen",
    "Phosphorous",
    "Pottasium",
)

# These are spelling corrections only; biological label validity needs
# confirmation from dataset provenance or a qualified reviewer.
CORRECTED_CLASS_MAPPING = {
    "Helathy": "Healthy",
    "Magnessium": "Magnesium",
    "Nitrogen": "Nitrogen",
    "Phosphorous": "Phosphorus",
    "Pottasium": "Potassium",
}

# Broad screening thresholds, not universal image-quality or exclusion rules.
SMALL_MIN_SIDE = 224
SMALL_MIN_PIXELS = 150_000
BLUR_LAPLACIAN_VARIANCE_MAX = 50.0
DARK_MEAN_LUMINANCE_MAX = 45.0
DARK_PIXEL_LEVEL = 32
DARK_PIXEL_FRACTION_MIN = 0.70
PHASH_MAX_HAMMING_DISTANCE = 6
SAMPLES_PER_CLASS = 20


def natural_key(path):
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


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
    pixels = np.asarray(grayscale, dtype=np.float32)
    low_frequency = cv2.dct(pixels)[:8, :8].flatten()
    threshold = float(np.median(low_frequency[1:]))
    value = 0
    for bit, coefficient in enumerate(low_frequency):
        if coefficient > threshold:
            value |= 1 << bit
    return value


def grouped_hashes(records, field_name):
    groups = defaultdict(list)
    for record in records:
        groups[record[field_name]].append(record)
    return [items for items in groups.values() if len(items) > 1]


def choose_representative_samples(paths, count):
    ordered = sorted(paths, key=natural_key)
    if len(ordered) <= count:
        return ordered
    indexes = np.linspace(0, len(ordered) - 1, num=count, dtype=int)
    return [ordered[index] for index in indexes]


def save_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_contact_sheet_cell(sheet, draw, path, x, y, cell_width, cell_height):
    image_area_height = cell_height - 34
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((cell_width - 8, image_area_height - 6))
            image_x = x + (cell_width - image.width) // 2
            image_y = y + (image_area_height - image.height) // 2
            sheet.paste(image, (image_x, image_y))
            label = f"{path.parent.name}/{path.name}"
            draw.text((x + 4, y + image_area_height), label[:38], fill="black")
    except (OSError, ValueError):
        draw.text((x + 4, y + 4), "Could not render", fill="red")


def create_contact_sheets(class_paths):
    contact_directory = OUTPUT_ROOT / "contact_sheets"
    contact_directory.mkdir(parents=True, exist_ok=True)
    cell_width, cell_height = 220, 174
    columns = 4
    overall_rows = []

    for folder in EXPECTED_FOLDERS:
        selected = choose_representative_samples(class_paths[folder], SAMPLES_PER_CLASS)
        rows = math.ceil(len(selected) / columns)
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(selected):
            x = (index % columns) * cell_width
            y = (index // columns) * cell_height
            add_contact_sheet_cell(sheet, draw, path, x, y, cell_width, cell_height)
        sheet.save(contact_directory / f"{folder.casefold()}_contact_sheet.jpg", quality=92)

        overall_rows.extend(
            (folder, path) for path in choose_representative_samples(class_paths[folder], 4)
        )

    all_sheet = Image.new(
        "RGB",
        (columns * cell_width, math.ceil(len(overall_rows) / columns) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(all_sheet)
    for index, (_, path) in enumerate(overall_rows):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        add_contact_sheet_cell(all_sheet, draw, path, x, y, cell_width, cell_height)
    all_sheet.save(contact_directory / "all_classes_representative_contact_sheet.jpg", quality=92)
    return contact_directory


def create_near_duplicate_review_sheet(near_rows):
    review_path = OUTPUT_ROOT / "perceptual_near_duplicate_review.jpg"
    if not near_rows:
        return None

    cell_width, cell_height = 220, 174
    header_height = 28
    sheet = Image.new(
        "RGB",
        (cell_width * 2, header_height + len(near_rows) * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(near_rows):
        y = header_height + row_index * cell_height
        draw.text(
            (4, y - header_height),
            f"pHash distance {row['phash_hamming_distance']}/64",
            fill="black",
        )
        for column, filename_key in enumerate(("file_a", "file_b")):
            path = DATASET_ROOT / row[f"folder_{'a' if column == 0 else 'b'}"] / row[filename_key]
            add_contact_sheet_cell(
                sheet,
                draw,
                path,
                column * cell_width,
                y,
                cell_width,
                cell_height,
            )
    sheet.save(review_path, quality=94)
    return review_path


def write_exact_duplicate_rows(records):
    rows = []
    for kind, field_name in (
        ("file_sha256", "file_sha256"),
        ("decoded_pixel_sha256", "pixel_sha256"),
    ):
        for group_number, group in enumerate(grouped_hashes(records, field_name), 1):
            for left, right in itertools.combinations(group, 2):
                rows.append(
                    {
                        "duplicate_type": kind,
                        "group_id": f"{kind}_{group_number}",
                        "folder_a": left["folder"],
                        "file_a": left["file_name"],
                        "folder_b": right["folder"],
                        "file_b": right["file_name"],
                    }
                )
    return rows


def write_near_duplicate_rows(records):
    rows = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            distance = (left["phash"] ^ right["phash"]).bit_count()
            if distance <= PHASH_MAX_HAMMING_DISTANCE:
                rows.append(
                    {
                        "folder_a": left["folder"],
                        "file_a": left["file_name"],
                        "folder_b": right["folder"],
                        "file_b": right["file_name"],
                        "phash_hamming_distance": distance,
                        "phash_similarity_percent": f"{100 * (64 - distance) / 64:.2f}",
                        "cross_class": left["folder"] != right["folder"],
                    }
                )
    return sorted(
        rows,
        key=lambda row: (
            int(row["phash_hamming_distance"]),
            row["folder_a"],
            row["file_a"],
            row["folder_b"],
            row["file_b"],
        ),
    )


def save_distribution_plots(records, class_counts):
    dimension_counts = defaultdict(int)
    for record in records:
        dimension_counts[(record["width"], record["height"])] += 1
    dimension_rows = [
        {"width": width, "height": height, "count": count}
        for (width, height), count in sorted(dimension_counts.items())
    ]
    save_csv(
        OUTPUT_ROOT / "dimension_distribution.csv",
        ["width", "height", "count"],
        dimension_rows,
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(
        [record["width"] for record in records],
        [record["height"] for record in records],
        alpha=0.65,
        s=22,
        color="#2a9d8f",
    )
    axes[0].set_title("Image dimensions")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Height (pixels)")
    axes[0].grid(alpha=0.2)
    aspect_ratios = [record["aspect_ratio"] for record in records]
    axes[1].hist(aspect_ratios, bins=24, color="#457b9d", edgecolor="white")
    axes[1].set_title("Aspect-ratio distribution")
    axes[1].set_xlabel("Width / height")
    axes[1].set_ylabel("Image count")
    axes[1].grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "dimension_and_aspect_ratio_distributions.png", dpi=160)
    plt.close(figure)

    class_names = list(EXPECTED_FOLDERS)
    fields = (
        ("width", "Width (px)"),
        ("height", "Height (px)"),
        ("aspect_ratio", "Aspect ratio"),
        ("mean_luminance", "Mean luminance"),
        ("laplacian_variance", "Laplacian variance"),
        ("pixel_count", "Pixel count"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(14, 14))
    for axis, (field, title) in zip(axes.flat, fields):
        data = [
            [record[field] for record in records if record["folder"] == name]
            for name in class_names
        ]
        axis.boxplot(data, tick_labels=class_names, showfliers=True)
        axis.set_title(f"{title} by source folder")
        axis.tick_params(axis="x", labelrotation=25)
        if field in {"laplacian_variance", "pixel_count"}:
            axis.set_yscale("log")
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "class_wise_visual_summary.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(class_names, [class_counts[name] for name in class_names], color="#457b9d")
    axis.set_title("Readable images by original folder")
    axis.set_ylabel("Image count")
    axis.tick_params(axis="x", labelrotation=20)
    axis.bar_label(bars)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "class_counts.png", dpi=160)
    plt.close(figure)


def image_statistics(path, image):
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    width, height = oriented.size
    rgb = np.asarray(oriented)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_luminance = float(np.mean(gray))
    dark_fraction = float(np.mean(gray <= DARK_PIXEL_LEVEL))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "path": str(path),
        "folder": path.parent.name,
        "file_name": path.name,
        "width": width,
        "height": height,
        "pixel_count": width * height,
        "aspect_ratio": width / height,
        "mean_luminance": mean_luminance,
        "dark_pixel_fraction": dark_fraction,
        "laplacian_variance": laplacian_variance,
        "file_sha256": sha256_file(path),
        "pixel_sha256": decoded_pixel_hash(image),
        "phash": perceptual_hash(image),
    }


def main():
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_ROOT}")
    found_folders = {path.name for path in DATASET_ROOT.iterdir() if path.is_dir()}
    if found_folders != set(EXPECTED_FOLDERS):
        raise RuntimeError(
            f"Expected exactly {EXPECTED_FOLDERS}, found {sorted(found_folders)}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    class_paths = {
        folder: sorted(
            (
                path
                for path in (DATASET_ROOT / folder).rglob("*")
                if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
            ),
            key=natural_key,
        )
        for folder in EXPECTED_FOLDERS
    }
    all_image_paths = [path for folder in EXPECTED_FOLDERS for path in class_paths[folder]]

    records = []
    corrupt_rows = []
    invalid_dimension_rows = []
    for path in all_image_paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                width, height = ImageOps.exif_transpose(image).size
                if width <= 0 or height <= 0:
                    invalid_dimension_rows.append(
                        {"folder": path.parent.name, "file_name": path.name, "width": width, "height": height}
                    )
                    continue
                records.append(image_statistics(path, image))
        except (OSError, ValueError, SyntaxError, cv2.error) as error:
            corrupt_rows.append(
                {
                    "folder": path.parent.name,
                    "file_name": path.name,
                    "path": str(path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    exact_rows = write_exact_duplicate_rows(records)
    near_rows = write_near_duplicate_rows(records)
    flags_by_image = defaultdict(list)
    for record in records:
        if (
            min(record["width"], record["height"]) < SMALL_MIN_SIDE
            or record["pixel_count"] < SMALL_MIN_PIXELS
        ):
            flags_by_image[record["path"]].append("small_resolution_screen")
        if record["laplacian_variance"] <= BLUR_LAPLACIAN_VARIANCE_MAX:
            flags_by_image[record["path"]].append("possible_blur_screen")
        if (
            record["mean_luminance"] < DARK_MEAN_LUMINANCE_MAX
            or record["dark_pixel_fraction"] >= DARK_PIXEL_FRACTION_MIN
        ):
            flags_by_image[record["path"]].append("dark_exposure_screen")

    for row in exact_rows:
        flags_by_image[str(DATASET_ROOT / row["folder_a"] / row["file_a"])].append(
            "exact_duplicate_candidate"
        )
        flags_by_image[str(DATASET_ROOT / row["folder_b"] / row["file_b"])].append(
            "exact_duplicate_candidate"
        )
    for row in near_rows:
        flags_by_image[str(DATASET_ROOT / row["folder_a"] / row["file_a"])].append(
            "perceptual_near_duplicate_candidate"
        )
        flags_by_image[str(DATASET_ROOT / row["folder_b"] / row["file_b"])].append(
            "perceptual_near_duplicate_candidate"
        )

    recommendation_rows = []
    for record in records:
        reasons = sorted(set(flags_by_image.get(record["path"], [])))
        if reasons:
            if "exact_duplicate_candidate" in reasons:
                recommendation = (
                    "Review the exact-match group; retain one verified copy or keep "
                    "duplicates in the same data split. Do not delete automatically."
                )
            elif "perceptual_near_duplicate_candidate" in reasons:
                recommendation = (
                    "Visually review the candidate pair and keep related images in the "
                    "same data split; perceptual similarity alone is not grounds for deletion."
                )
            elif any(
                flag in reasons
                for flag in (
                    "small_resolution_screen",
                    "possible_blur_screen",
                    "dark_exposure_screen",
                )
            ):
                recommendation = (
                    "Inspect the full-resolution image; exclude only if the relevant "
                    "leaf detail is unusable under documented criteria."
                )
            else:
                recommendation = "Manually review before deciding whether to exclude."
            recommendation_rows.append(
                {
                    "folder": record["folder"],
                    "file_name": record["file_name"],
                    "path": record["path"],
                    "screen_flags": ";".join(reasons),
                    "recommendation": recommendation,
                    "deleted": False,
                }
            )
    for row in corrupt_rows:
        recommendation_rows.append(
            {
                "folder": row["folder"],
                "file_name": row["file_name"],
                "path": row["path"],
                "screen_flags": "corrupt_or_unreadable",
                "recommendation": "Exclude from training if corruption is confirmed.",
                "deleted": False,
            }
        )

    save_csv(
        OUTPUT_ROOT / "corrupted_images.csv",
        ["folder", "file_name", "path", "error_type", "error"],
        corrupt_rows,
    )
    save_csv(
        OUTPUT_ROOT / "invalid_dimensions.csv",
        ["folder", "file_name", "width", "height"],
        invalid_dimension_rows,
    )
    save_csv(
        OUTPUT_ROOT / "duplicate_images.csv",
        ["duplicate_type", "group_id", "folder_a", "file_a", "folder_b", "file_b"],
        exact_rows,
    )
    save_csv(
        OUTPUT_ROOT / "perceptual_near_duplicate_candidates.csv",
        [
            "folder_a",
            "file_a",
            "folder_b",
            "file_b",
            "phash_hamming_distance",
            "phash_similarity_percent",
            "cross_class",
        ],
        near_rows,
    )
    metric_fields = [
        "folder",
        "file_name",
        "width",
        "height",
        "pixel_count",
        "aspect_ratio",
        "mean_luminance",
        "dark_pixel_fraction",
        "laplacian_variance",
        "file_sha256",
        "pixel_sha256",
        "phash",
    ]
    save_csv(
        OUTPUT_ROOT / "image_metrics.csv",
        metric_fields,
        [
            {key: record[key] for key in metric_fields}
            for record in records
        ],
    )
    class_counts = {folder: len(class_paths[folder]) for folder in EXPECTED_FOLDERS}
    class_rows = []
    for folder in EXPECTED_FOLDERS:
        items = [record for record in records if record["folder"] == folder]
        class_rows.append(
            {
                "original_folder": folder,
                "corrected_spelling_only_label": CORRECTED_CLASS_MAPPING[folder],
                "source_images": len(class_paths[folder]),
                "readable_images": len(items),
                "corrupt_or_unreadable": sum(row["folder"] == folder for row in corrupt_rows),
                "dimension_outliers": sum(
                    bool(
                        min(record["width"], record["height"]) < SMALL_MIN_SIDE
                        or record["pixel_count"] < SMALL_MIN_PIXELS
                    )
                    for record in items
                ),
                "possible_blur_screens": sum(
                    record["laplacian_variance"] <= BLUR_LAPLACIAN_VARIANCE_MAX
                    for record in items
                ),
                "dark_exposure_screens": sum(
                    record["mean_luminance"] < DARK_MEAN_LUMINANCE_MAX
                    or record["dark_pixel_fraction"] >= DARK_PIXEL_FRACTION_MIN
                    for record in items
                ),
            }
        )
    save_csv(
        OUTPUT_ROOT / "class_wise_summary.csv",
        list(class_rows[0].keys()),
        class_rows,
    )
    save_csv(
        OUTPUT_ROOT / "recommended_exclusions.csv",
        ["folder", "file_name", "path", "screen_flags", "recommendation", "deleted"],
        recommendation_rows,
    )
    save_distribution_plots(records, class_counts)
    contact_directory = create_contact_sheets(class_paths)
    near_duplicate_review_path = create_near_duplicate_review_sheet(near_rows)

    class_stats = {}
    for folder in EXPECTED_FOLDERS:
        items = [record for record in records if record["folder"] == folder]
        class_stats[folder] = {
            "count": len(items),
            "median_width": float(np.median([item["width"] for item in items])) if items else 0,
            "median_height": float(np.median([item["height"] for item in items])) if items else 0,
            "median_aspect": float(np.median([item["aspect_ratio"] for item in items])) if items else 0,
            "median_luminance": float(np.median([item["mean_luminance"] for item in items])) if items else 0,
            "median_blur_score": float(np.median([item["laplacian_variance"] for item in items])) if items else 0,
        }
    all_widths = [record["width"] for record in records]
    all_heights = [record["height"] for record in records]
    all_aspect_ratios = [record["aspect_ratio"] for record in records]
    small_screen_count = sum(
        min(record["width"], record["height"]) < SMALL_MIN_SIDE
        or record["pixel_count"] < SMALL_MIN_PIXELS
        for record in records
    )
    blur_screen_count = sum(
        record["laplacian_variance"] <= BLUR_LAPLACIAN_VARIANCE_MAX
        for record in records
    )
    dark_screen_count = sum(
        record["mean_luminance"] < DARK_MEAN_LUMINANCE_MAX
        or record["dark_pixel_fraction"] >= DARK_PIXEL_FRACTION_MIN
        for record in records
    )

    byte_duplicates = grouped_hashes(records, "file_sha256")
    pixel_duplicates = grouped_hashes(records, "pixel_sha256")
    exact_cross_class = sum(
        len({record["folder"] for record in group}) > 1
        for group in byte_duplicates + pixel_duplicates
    )
    near_cross_class = sum(row["cross_class"] for row in near_rows)
    report_lines = [
        "# Nutrient Dataset Quality Assessment",
        "",
        f"- Assessment time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Dataset: `{DATASET_ROOT}`",
        f"- Quality outputs: `{OUTPUT_ROOT}`",
        "- Scope: read-only image-quality assessment; no training was performed.",
        "- Dataset files were not moved, renamed, overwritten, or deleted.",
        "- The disease dataset, trained models, and disease evaluation reports were not used.",
        "",
        "## Dataset inventory",
        "",
        f"- Original image files found: {len(all_image_paths)}",
        f"- Successfully decoded and measured: {len(records)}",
        f"- Corrupt or unreadable: {len(corrupt_rows)}",
        f"- Invalid dimensions: {len(invalid_dimension_rows)}",
        f"- Original extension types: {', '.join(sorted({p.suffix.lower() for p in all_image_paths})) or '(none)'}",
        "",
        "| Original folder | Corrected spelling only | Files | Readable |",
        "|---|---|---:|---:|",
    ]
    for folder in EXPECTED_FOLDERS:
        report_lines.append(
            f"| {folder} | {CORRECTED_CLASS_MAPPING[folder]} | "
            f"{class_counts[folder]} | {class_stats[folder]['count']} |"
        )
    report_lines.extend(
        [
            "",
            "## Folder-name spelling notes and mapping",
            "",
            "The following orthographic corrections are documented before the code mapping: "
            "`Helathy` -> `Healthy`, `Magnessium` -> `Magnesium`, "
            "`Phosphorous` -> `Phosphorus`, and `Pottasium` -> `Potassium`. "
            "`Nitrogen` is unchanged. The mapping is present only as a Python constant in "
            "`assess_dataset_quality.py`; no mapping JSON was created. This does not verify "
            "that images truly show the nutrient condition named by their source folder.",
            "",
            "## Exact duplicate results",
            "",
            f"- Duplicate byte-SHA256 groups: {len(byte_duplicates)}",
            f"- Duplicate decoded-pixel SHA256 groups: {len(pixel_duplicates)}",
            f"- Exact duplicate groups crossing source folders: {exact_cross_class}",
            f"- Exact duplicate pair rows: {len(exact_rows)}",
            "File hashes detect byte-identical copies. Decoded-pixel hashes additionally "
            "detect identical oriented RGB pixels despite different file encoding/metadata.",
            "",
            "## Perceptual near-duplicate screening",
            "",
            f"- Algorithm: 64-bit DCT perceptual hash (8x8 low-frequency coefficients).",
            f"- Candidate threshold: Hamming distance <= {PHASH_MAX_HAMMING_DISTANCE}/64.",
            f"- Candidate pairs: {len(near_rows)}; cross-folder candidate pairs: {near_cross_class}.",
            "Near-duplicate candidates require visual review; pHash similarity is not proof "
            "that two files are the same source image.",
            (
                f"Pair review sheet: `{near_duplicate_review_path}`."
                if near_duplicate_review_path
                else "No near-duplicate review sheet was needed."
            ),
            "",
            "## Image quality screens",
            "",
            f"- Small-resolution screen: shortest side < {SMALL_MIN_SIDE}px OR total pixels < {SMALL_MIN_PIXELS:,}.",
            f"- Possible blur screen: variance of Laplacian <= {BLUR_LAPLACIAN_VARIANCE_MAX}.",
            f"- Dark screen: mean grayscale luminance < {DARK_MEAN_LUMINANCE_MAX}, or at least "
            f"{DARK_PIXEL_FRACTION_MIN:.0%} of pixels <= {DARK_PIXEL_LEVEL}.",
            "These broad thresholds flag candidates for review only. Blur scores depend on "
            "resolution/content; brightness depends on capture conditions. They are not "
            "automatic exclusion criteria.",
            f"- Images flagged by small-resolution screen: {small_screen_count}",
            f"- Images flagged by possible-blur screen: {blur_screen_count}",
            f"- Images flagged by dark-exposure screen: {dark_screen_count}",
            f"- Width range: {min(all_widths)}-{max(all_widths)} px; "
            f"height range: {min(all_heights)}-{max(all_heights)} px.",
            f"- Aspect-ratio range: {min(all_aspect_ratios):.3f}-"
            f"{max(all_aspect_ratios):.3f}; median: {np.median(all_aspect_ratios):.3f}.",
            "",
            "| Original folder | Count | Median W x H | Median W/H | Median luminance | Median Laplacian variance |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for folder in EXPECTED_FOLDERS:
        stats = class_stats[folder]
        report_lines.append(
            f"| {folder} | {stats['count']} | {stats['median_width']:.0f} x "
            f"{stats['median_height']:.0f} | {stats['median_aspect']:.3f} | "
            f"{stats['median_luminance']:.1f} | {stats['median_blur_score']:.2f} |"
        )
    report_lines.extend(
        [
            "",
            "Distinct width/height pairs: "
            f"{len({(r['width'], r['height']) for r in records})}. "
            "Per-image dimensions and aspect ratios are in `image_metrics.csv`; "
            "the aggregated dimension counts are in `dimension_distribution.csv`.",
            "",
            "## Visual representativeness and label validity",
            "",
            f"Class-wise sheets show up to {SAMPLES_PER_CLASS} deterministically sampled "
            "images per folder, plus an all-class representative sheet. The reviewed samples "
            "are close-up leaf photographs; they vary in framing, orientation, shadows, and "
            "background, and visible green/yellow/brown/purple color patterns occur across "
            "the collection. These observations are not a diagnosis. The five within-folder "
            "perceptual-near-duplicate pairs were visually reviewed and appear similar; "
            "they are not exact pixel duplicates.",
            "The images alone do not establish that each photo genuinely belongs to its "
            "assigned nutrient class. No biological class was inferred or assigned from "
            "appearance. Confirm dataset provenance and labels with the source documentation "
            "and a qualified maize/agronomy reviewer before treating these folders as "
            "verified classes.",
            "",
            "## Training suitability",
            "",
            "Current status: **not ready for unsupervised/automatic training approval**. "
            "The number of images is modest and class counts are uneven; the dataset is a "
            "single flat folder set without train/validation/test splits. Most importantly, "
            "visual inspection alone cannot verify the nutrient labels. Before training, "
            "manually confirm labels and image provenance, review every listed quality flag "
            "and near-duplicate candidate, decide documented exclusion criteria, and create "
            "a leakage-aware split (keeping near-duplicate/source-related images together).",
            "",
            "Recommended exclusions are suggestions only; **no files were deleted**. "
            "The CSV marks every candidate and records `deleted=False`.",
            "",
            "## Output files",
            "",
            "- `corrupted_images.csv`",
            "- `invalid_dimensions.csv`",
            "- `duplicate_images.csv`",
            "- `perceptual_near_duplicate_candidates.csv`",
            "- `image_metrics.csv`",
            "- `dimension_distribution.csv`",
            "- `class_wise_summary.csv`",
            "- `recommended_exclusions.csv`",
            "- `dimension_and_aspect_ratio_distributions.png`",
            "- `class_wise_visual_summary.png`",
            "- `class_counts.png`",
            "- `perceptual_near_duplicate_review.jpg` (when candidate pairs exist)",
            f"- Contact sheets: `{contact_directory}`",
            "",
        ]
    )
    (OUTPUT_ROOT / "dataset_quality_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Images found: {len(all_image_paths)}; readable: {len(records)}")
    print(
        f"Corrupt/unreadable: {len(corrupt_rows)}; invalid dimensions: "
        f"{len(invalid_dimension_rows)}"
    )
    print(
        f"Exact duplicate rows: {len(exact_rows)}; perceptual near-duplicate "
        f"candidate pairs: {len(near_rows)}"
    )
    print(f"Quality report: {OUTPUT_ROOT / 'dataset_quality_report.md'}")


if __name__ == "__main__":
    main()
