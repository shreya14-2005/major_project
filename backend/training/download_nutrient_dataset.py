"""Download and verify the Maize Nutrient Deficiency Dataset from Mendeley Data (DOI: 10.17632/34gb2gr7p2.1)."""

import hashlib
import json
import time
from pathlib import Path
import requests

FOLDER_MAPPING = {
    "fcd22498-9c9e-4335-9597-68283dbba340": "Helathy",
    "9ce3cb5c-c839-4d1e-a248-8bfde9e59d92": "Magnessium",
    "2c9206ca-9b88-4b05-b5d4-7318ce2878b3": "Nitrogen",
    "557d8713-b74e-4830-966a-f04850b75c4b": "Phosphorous",
    "c51c0e2d-35ba-4dd7-92c6-2a282b7ee95c": "Pottasium",
}

EXPECTED_COUNTS = {
    "Helathy": 72,
    "Magnessium": 123,
    "Nitrogen": 99,
    "Phosphorous": 113,
    "Pottasium": 56,
}


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "datasets" / "Maize Nutrient Deficiency Dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(r"C:\Users\chait\.gemini\antigravity\brain\4b41527f-3a87-45c8-be4c-b3873bc26068\.system_generated\steps\128\content.md")
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cached Mendeley API response not found: {cache_path}")

    data = None
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.startswith('{"id":"34gb2gr7p2"'):
            data = json.loads(line)
            break
    if not data:
        raise ValueError("Could not find Mendeley JSON metadata in cache")

    files = data.get("files", [])
    print(f"Total files in dataset metadata: {len(files)}")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    session = requests.Session()
    session.headers.update(headers)

    downloaded = 0
    verified = 0

    for idx, file_info in enumerate(files, 1):
        folder_id = file_info["folder_id"]
        class_folder = FOLDER_MAPPING[folder_id]
        target_dir = dataset_dir / class_folder
        target_dir.mkdir(parents=True, exist_ok=True)

        filename = file_info["filename"]
        target_file = target_dir / filename
        expected_hash = file_info["content_details"]["sha256_hash"]
        expected_size = file_info["size"]
        download_url = file_info["content_details"]["download_url"]

        if target_file.is_file() and target_file.stat().st_size == expected_size:
            if sha256_file(target_file) == expected_hash:
                verified += 1
                if idx % 50 == 0 or idx == len(files):
                    print(f"[{idx}/{len(files)}] Existing verified: {class_folder}/{filename}")
                continue

        retries = 3
        while retries > 0:
            try:
                resp = session.get(download_url, timeout=30)
                if resp.status_code == 200:
                    target_file.write_bytes(resp.content)
                    actual_hash = sha256_file(target_file)
                    if actual_hash != expected_hash:
                        raise ValueError(f"Hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}")
                    downloaded += 1
                    verified += 1
                    break
                else:
                    time.sleep(1)
                    retries -= 1
            except Exception as e:
                time.sleep(2)
                retries -= 1
                if retries == 0:
                    raise RuntimeError(f"Failed to download {filename}: {e}")

        if idx % 50 == 0 or idx == len(files):
            print(f"[{idx}/{len(files)}] Processed: {class_folder}/{filename} ({verified} verified)")

    print("\nDownload & Verification Summary:")
    print(f"Total files verified: {verified}/{len(files)}")
    for class_folder, expected_count in EXPECTED_COUNTS.items():
        actual_count = len(list((dataset_dir / class_folder).glob("*")))
        print(f"  {class_folder}: {actual_count} images (expected: {expected_count})")
        assert actual_count == expected_count, f"Count mismatch for {class_folder}"

    print(f"\nDataset successfully placed at: {dataset_dir.resolve()}")


if __name__ == "__main__":
    main()
