# Nutrient Dataset Quality Assessment

- Assessment time: 2026-11-11T23:43:27+05:30
- Dataset: `C:\Users\ADMIN\Desktop\major_project\datasets\Maize Nutrient Deficiency Dataset`
- Quality outputs: `C:\Users\ADMIN\Desktop\major_project\evaluation\nutrient_dataset_quality`
- Scope: read-only image-quality assessment; no training was performed.
- Dataset files were not moved, renamed, overwritten, or deleted.
- The disease dataset, trained models, and disease evaluation reports were not used.

## Dataset inventory

- Original image files found: 463
- Successfully decoded and measured: 463
- Corrupt or unreadable: 0
- Invalid dimensions: 0
- Original extension types: .jpeg

| Original folder | Corrected spelling only | Files | Readable |
|---|---|---:|---:|
| Helathy | Healthy | 72 | 72 |
| Magnessium | Magnesium | 123 | 123 |
| Nitrogen | Nitrogen | 99 | 99 |
| Phosphorous | Phosphorus | 113 | 113 |
| Pottasium | Potassium | 56 | 56 |

## Folder-name spelling notes and mapping

The following orthographic corrections are documented before the code mapping: `Helathy` -> `Healthy`, `Magnessium` -> `Magnesium`, `Phosphorous` -> `Phosphorus`, and `Pottasium` -> `Potassium`. `Nitrogen` is unchanged. The mapping is present only as a Python constant in `assess_dataset_quality.py`; no mapping JSON was created. This does not verify that images truly show the nutrient condition named by their source folder.

## Exact duplicate results

- Duplicate byte-SHA256 groups: 0
- Duplicate decoded-pixel SHA256 groups: 0
- Exact duplicate groups crossing source folders: 0
- Exact duplicate pair rows: 0
File hashes detect byte-identical copies. Decoded-pixel hashes additionally detect identical oriented RGB pixels despite different file encoding/metadata.

## Perceptual near-duplicate screening

- Algorithm: 64-bit DCT perceptual hash (8x8 low-frequency coefficients).
- Candidate threshold: Hamming distance <= 6/64.
- Candidate pairs: 5; cross-folder candidate pairs: 0.
Near-duplicate candidates require visual review; pHash similarity is not proof that two files are the same source image.
Pair review sheet: `C:\Users\ADMIN\Desktop\major_project\evaluation\nutrient_dataset_quality\perceptual_near_duplicate_review.jpg`.

## Image quality screens

- Small-resolution screen: shortest side < 224px OR total pixels < 150,000.
- Possible blur screen: variance of Laplacian <= 50.0.
- Dark screen: mean grayscale luminance < 45.0, or at least 70% of pixels <= 32.
These broad thresholds flag candidates for review only. Blur scores depend on resolution/content; brightness depends on capture conditions. They are not automatic exclusion criteria.
- Images flagged by small-resolution screen: 49
- Images flagged by possible-blur screen: 85
- Images flagged by dark-exposure screen: 0
- Width range: 164-1121 px; height range: 179-958 px.
- Aspect-ratio range: 0.325-5.106; median: 1.121.

| Original folder | Count | Median W x H | Median W/H | Median luminance | Median Laplacian variance |
|---|---:|---:|---:|---:|---:|
| Helathy | 72 | 525 x 601 | 0.926 | 147.9 | 197.31 |
| Magnessium | 123 | 579 x 430 | 1.570 | 155.0 | 76.82 |
| Nitrogen | 99 | 555 x 467 | 0.949 | 161.9 | 111.88 |
| Phosphorous | 113 | 711 x 552 | 1.294 | 147.9 | 282.89 |
| Pottasium | 56 | 664 x 550 | 1.181 | 149.2 | 140.17 |

Distinct width/height pairs: 448. Per-image dimensions and aspect ratios are in `image_metrics.csv`; the aggregated dimension counts are in `dimension_distribution.csv`.

## Visual representativeness and label validity

Class-wise sheets show up to 20 deterministically sampled images per folder, plus an all-class representative sheet. The reviewed samples are close-up leaf photographs; they vary in framing, orientation, shadows, and background, and visible green/yellow/brown/purple color patterns occur across the collection. These observations are not a diagnosis. The five within-folder perceptual-near-duplicate pairs were visually reviewed and appear similar; they are not exact pixel duplicates.
The images alone do not establish that each photo genuinely belongs to its assigned nutrient class. No biological class was inferred or assigned from appearance. Confirm dataset provenance and labels with the source documentation and a qualified maize/agronomy reviewer before treating these folders as verified classes.

## Training suitability

Current status: **not ready for unsupervised/automatic training approval**. The number of images is modest and class counts are uneven; the dataset is a single flat folder set without train/validation/test splits. Most importantly, visual inspection alone cannot verify the nutrient labels. Before training, manually confirm labels and image provenance, review every listed quality flag and near-duplicate candidate, decide documented exclusion criteria, and create a leakage-aware split (keeping near-duplicate/source-related images together).

Recommended exclusions are suggestions only; **no files were deleted**. The CSV marks every candidate and records `deleted=False`.

## Output files

- `corrupted_images.csv`
- `invalid_dimensions.csv`
- `duplicate_images.csv`
- `perceptual_near_duplicate_candidates.csv`
- `image_metrics.csv`
- `dimension_distribution.csv`
- `class_wise_summary.csv`
- `recommended_exclusions.csv`
- `dimension_and_aspect_ratio_distributions.png`
- `class_wise_visual_summary.png`
- `class_counts.png`
- `perceptual_near_duplicate_review.jpg` (when candidate pairs exist)
- Contact sheets: `C:\Users\ADMIN\Desktop\major_project\evaluation\nutrient_dataset_quality\contact_sheets`
