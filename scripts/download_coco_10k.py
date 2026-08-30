"""Download a subset of MS COCO 2017 train images with their captions.

Source: https://cocodataset.org/

Output layout (consumed by app/train.py):
    <out_dir>/
        images/
        captions.csv        image_id, file_name, caption
        metadata.json

Usage:
    python scripts/download_coco_10k.py --out_dir data/coco_10k --num_images 10000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = "http://images.cocodataset.org/train2017"
CAPTIONS_MEMBER = "annotations/captions_train2017.json"
USER_AGENT = "Mozilla/5.0"


def http_get(url: str, retries: int, retry_delay: float, timeout: int = 60) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[DOWNLOAD]: {url}")
            request = Request(url, headers= {"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def download_file(url: str, destination: Path, retries: int, retry_delay: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[SKIP] {destination}")
        return
    print(f"[DOWNLOAD] {url}")
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.write_bytes(http_get(url, retries, retry_delay))
    tmp.replace(destination)  # atomic: a partial file is never mistaken for a finished one
    print(f"[OK] {destination}")


def prepare_annotations(out_dir: Path, retries: int, retry_delay: float) -> dict:
    zip_path = out_dir / "annotations_trainval2017.zip"
    json_path = out_dir / CAPTIONS_MEMBER

    download_file(ANNOTATIONS_URL, zip_path, retries, retry_delay)
    if not json_path.exists():
        print("Extracting captions_train2017.json ...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            if CAPTIONS_MEMBER not in zip_ref.namelist():
                raise FileNotFoundError(f"{CAPTIONS_MEMBER} missing from {zip_path}")
            zip_ref.extract(CAPTIONS_MEMBER, out_dir)

    print("Loading captions ...")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_subset(coco: dict, num_images: int, seed: int) -> Tuple[List[dict], Dict[int, List[str]]]:
    images = coco["images"]
    print(f"Total COCO train images: {len(images):,}")
    if num_images > len(images):
        raise ValueError(f"Requested {num_images:,} images but COCO has {len(images):,}.")

    random.seed(seed)
    selected = random.sample(images, num_images)
    selected_ids: Set[int] = {image["id"] for image in selected}

    captions_by_image: Dict[int, List[str]] = {}
    for ann in coco["annotations"]:
        if ann["image_id"] in selected_ids:
            captions_by_image.setdefault(ann["image_id"], []).append(ann["caption"])

    valid = [image for image in selected if image["id"] in captions_by_image]
    print(f"Selected images with captions: {len(valid):,}")
    return valid, captions_by_image


def download_images(
    selected: List[dict], image_dir: Path, workers: int, retries: int, retry_delay: float
) -> Tuple[List[dict], List[dict]]:
    image_dir.mkdir(parents=True, exist_ok=True)

    def fetch(image: dict) -> Tuple[dict, bool, str]:
        destination = image_dir / image["file_name"]
        if destination.exists() and destination.stat().st_size > 0:
            return image, True, "already exists"
        try:
            url = image.get("coco_url") or f"{IMAGE_BASE_URL}/{image['file_name']}"
            destination.write_bytes(http_get(url, retries, retry_delay))
            return image, True, "downloaded"
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            return image, False, str(exc)

    print(f"Downloading {len(selected):,} images with {workers} workers ...")
    ok: List[dict] = []
    failed: List[dict] = []
    total = len(selected)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, image) for image in selected]
        for index, future in enumerate(as_completed(futures), start=1):
            image, success, message = future.result()
            (ok if success else failed).append(
                image if success else {"file_name": image["file_name"], "error": message}
            )
            if index % 100 == 0 or index == total:
                print(f"Progress: {index:,}/{total:,} | ok {len(ok):,} | failed {len(failed):,}")
    return ok, failed


def save_caption_csv(images: List[dict], captions_by_image: Dict[int, List[str]], out_dir: Path) -> int:
    csv_path = out_dir / "captions.csv"
    rows = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "file_name", "caption"])
        for image in images:
            for caption in captions_by_image[image["id"]]:
                writer.writerow([image["id"], image["file_name"], " ".join(caption.split())])
                rows += 1
    print(f"[OK] {csv_path} ({rows:,} caption rows)")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a COCO 2017 train subset.")
    parser.add_argument("--out_dir", type=str, default="data/coco_10k")
    parser.add_argument("--num_images", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry_delay", type=float, default=2.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = prepare_annotations(out_dir, args.retries, args.retry_delay)
    selected, captions_by_image = create_subset(coco, args.num_images, args.seed)
    downloaded, failed = download_images(
        selected, out_dir / "images", args.workers, args.retries, args.retry_delay
    )

    # The CSV only lists images that are actually on disk, so training never
    # points at a missing file.
    caption_rows = save_caption_csv(downloaded, captions_by_image, out_dir)

    if failed:
        failed_path = out_dir / "failed_downloads.json"
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        print(f"WARNING: {len(failed):,} images failed; see {failed_path}")

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "MS COCO 2017 Train",
                "num_images_requested": args.num_images,
                "num_images_downloaded": len(downloaded),
                "num_caption_rows": caption_rows,
                "seed": args.seed,
                "image_resolution_note": "Original COCO resolutions are preserved.",
            },
            f,
            indent=4,
        )

    print(f"\nDONE — dataset at {out_dir.resolve()}")


if __name__ == "__main__":
    main()
