"""Kvasir-SEG download, extraction, and integrity verification.

Tries the Kaggle API first (`debeshjha1/kvasirseg`), falling back to the official
Simula zip if Kaggle credentials are missing or the API call fails. The Simula
download link is resolved by fetching the index page and parsing out the current
`.zip` href — never hardcoded, since Simula has changed the asset name before.
Idempotent: a verified existing copy at `paths.data_root` skips the download.
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import cv2
import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)

_KAGGLE_DATASET = "debeshjha1/kvasirseg"
_SIMULA_INDEX_URL = "https://datasets.simula.no/kvasir-seg/"
_EXPECTED_COUNT = 1000


class DatasetVerificationError(RuntimeError):
    """Raised when the dataset at a given path fails integrity verification."""


def ensure_kvasir_seg(cfg: dict[str, Any]) -> tuple[Path, tuple[int, int, int]]:
    """Ensure a verified copy of Kvasir-SEG exists at `cfg['paths']['data_root']`.

    Returns (data_root, (n_images, n_masks, n_unreadable)). Skips the download
    entirely if a verified copy is already present.
    """
    data_root = Path(cfg["paths"]["data_root"])

    try:
        counts = verify_integrity(data_root)
        logger.info("Kvasir-SEG already verified at %s, skipping download.", data_root)
        return data_root, counts
    except (DatasetVerificationError, FileNotFoundError) as exc:
        logger.info("No verified copy at %s (%s); downloading.", data_root, exc)

    data_root.mkdir(parents=True, exist_ok=True)
    extracted_root = _download_via_kaggle(data_root)
    if extracted_root is None:
        extracted_root = _download_via_simula(data_root)
    _flatten_into(extracted_root, data_root)

    counts = verify_integrity(data_root)
    return data_root, counts


def verify_integrity(data_root: Path) -> tuple[int, int, int]:
    """Assert exactly 1000 images and 1000 masks with matching filenames and no
    corrupt reads. Returns (n_images, n_masks, n_unreadable) on success.
    """
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(f"{images_dir} and/or {masks_dir} do not exist.")

    image_files = {p.name: p for p in images_dir.iterdir() if p.is_file()}
    mask_files = {p.name: p for p in masks_dir.iterdir() if p.is_file()}

    if len(image_files) != _EXPECTED_COUNT:
        raise DatasetVerificationError(
            f"Expected {_EXPECTED_COUNT} images, found {len(image_files)}."
        )
    if len(mask_files) != _EXPECTED_COUNT:
        raise DatasetVerificationError(
            f"Expected {_EXPECTED_COUNT} masks, found {len(mask_files)}."
        )

    if image_files.keys() != mask_files.keys():
        missing_masks = sorted(image_files.keys() - mask_files.keys())[:5]
        missing_images = sorted(mask_files.keys() - image_files.keys())[:5]
        raise DatasetVerificationError(
            f"images/ and masks/ filenames don't match "
            f"(e.g. missing masks: {missing_masks}, missing images: {missing_images})."
        )

    n_unreadable = sum(
        1
        for path in (*image_files.values(), *mask_files.values())
        if cv2.imread(str(path)) is None
    )
    if n_unreadable:
        raise DatasetVerificationError(f"{n_unreadable} file(s) failed to read.")

    return len(image_files), len(mask_files), n_unreadable


def _download_via_kaggle(data_root: Path) -> Path | None:
    """Download+extract via the Kaggle API. Returns the extraction root, or None to
    signal the caller should fall back to the Simula zip.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.info("kaggle package not installed; falling back to Simula.")
        return None

    tmp_dir = data_root / "_kaggle_download"
    try:
        api = KaggleApi()
        api.authenticate()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(_KAGGLE_DATASET, path=str(tmp_dir), unzip=True, quiet=False)
        return tmp_dir
    except Exception as exc:  # kaggle raises ad hoc exceptions for missing/invalid creds
        logger.info("Kaggle API unavailable (%s); falling back to Simula.", exc)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None


def _download_via_simula(data_root: Path) -> Path:
    """Fetch the Simula index page, resolve the current `.zip` download link, then
    download and extract it. The link is parsed at runtime, never hardcoded.
    """
    resp = requests.get(_SIMULA_INDEX_URL, timeout=30)
    resp.raise_for_status()
    zip_hrefs = re.findall(r'href="([^"]+\.zip)"', resp.text)
    if not zip_hrefs:
        raise RuntimeError(f"Could not find a .zip download link on {_SIMULA_INDEX_URL}")
    zip_href = next((h for h in zip_hrefs if "kvasir-seg" in h.lower()), zip_hrefs[0])
    zip_url = urljoin(_SIMULA_INDEX_URL, zip_href)

    tmp_dir = data_root / "_simula_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_dir / "download.zip"

    logger.info("Downloading %s ...", zip_url)
    with requests.get(zip_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)
    zip_path.unlink()
    return tmp_dir


def _flatten_into(extracted_root: Path, data_root: Path) -> None:
    """Locate `images/` and `masks/` anywhere under `extracted_root`, move them
    directly under `data_root`, and clean up the extraction scratch directory.
    """
    images_src = _find_subdir(extracted_root, "images")
    masks_src = _find_subdir(extracted_root, "masks")
    if images_src is None or masks_src is None:
        raise DatasetVerificationError(
            f"Could not locate images/ and masks/ under {extracted_root}"
        )

    for name, src in (("images", images_src), ("masks", masks_src)):
        dst = data_root / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))

    shutil.rmtree(extracted_root, ignore_errors=True)


def _find_subdir(root: Path, name: str) -> Path | None:
    """Return the first directory named `name` under `root` (including `root` itself)."""
    if (root / name).is_dir():
        return root / name
    for candidate in root.rglob(name):
        if candidate.is_dir():
            return candidate
    return None
