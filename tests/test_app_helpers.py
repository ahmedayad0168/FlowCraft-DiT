"""Tests for the training script's data pipeline and schedule helpers."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image  # noqa: E402

from app.train import CocoCaptionDataset, drop_captions, lr_lambda, parse_args  # noqa: E402


def make_dataset_dir(tmp_path: Path, num_images: int = 3, extra_missing_rows: int = 0) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for i in range(num_images):
        name = f"img_{i}.jpg"
        Image.new("RGB", (64, 48), color=(i * 10, 20, 30)).save(image_dir / name)
        rows.append((i, name, f"caption number {i}"))
    for j in range(extra_missing_rows):
        rows.append((100 + j, f"ghost_{j}.jpg", "caption for a file that never downloaded"))

    with open(tmp_path / "captions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "file_name", "caption"])
        writer.writerows(rows)
    return tmp_path


def test_dataset_returns_normalized_square_tensors(tmp_path):
    dataset = CocoCaptionDataset(make_dataset_dir(tmp_path), resolution=32)
    image, caption = dataset[0]
    assert image.shape == (3, 32, 32)
    assert image.min() >= -1.0 and image.max() <= 1.0
    assert caption == "caption number 0"


def test_dataset_skips_rows_whose_image_is_missing(tmp_path):
    dataset = CocoCaptionDataset(make_dataset_dir(tmp_path, 2, extra_missing_rows=3), resolution=32)
    assert len(dataset) == 2


def test_dataset_can_limit_rows_for_overfit_debugging(tmp_path):
    data_dir = make_dataset_dir(tmp_path, 3)
    dataset = CocoCaptionDataset(data_dir, resolution=32, max_samples=2)
    assert len(dataset) == 2
    with pytest.raises(ValueError):
        CocoCaptionDataset(data_dir, max_samples=0)


def test_dataset_errors_without_captions_csv(tmp_path):
    with pytest.raises(FileNotFoundError):
        CocoCaptionDataset(tmp_path)


def test_dataset_errors_when_no_row_matches_an_image(tmp_path):
    (tmp_path / "images").mkdir()
    with open(tmp_path / "captions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "file_name", "caption"])
        writer.writerow([1, "nope.jpg", "a caption"])
    with pytest.raises(RuntimeError):
        CocoCaptionDataset(tmp_path)


def test_lr_schedule_warms_up_then_decays_to_the_floor():
    warmup, total, floor = 10, 100, 0.05
    assert lr_lambda(0, warmup, total, floor) == pytest.approx(0.1)
    assert lr_lambda(warmup - 1, warmup, total, floor) == pytest.approx(1.0)
    assert lr_lambda(total, warmup, total, floor) == pytest.approx(floor)
    mid = lr_lambda(total // 2, warmup, total, floor)
    assert floor < mid < 1.0


def test_caption_dropout_bounds():
    captions = [f"caption {i}" for i in range(200)]
    assert drop_captions(captions, 0.0) == captions
    assert all(c == "" for c in drop_captions(captions, 1.0))
    dropped = sum(1 for c in drop_captions(captions, 0.5) if c == "")
    assert 0 < dropped < len(captions)


def test_parse_args_defaults_enable_cfg_and_ema():
    args = parse_args([])
    assert args.cond_dropout > 0.0  # without caption dropout, CFG at inference is a no-op
    assert args.ema_decay > 0.0
    assert args.precision in ("fp32", "bf16", "fp16")
    assert args.preview_seed == 1234
