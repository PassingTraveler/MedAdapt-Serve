from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import requests

from config import DEFAULT_CMB_ZIP_URL, PATHS


def download_resumable(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    append = existing > 0 and response.status_code == 206
    if not append:
        existing = 0
    mode = "ab" if append else "wb"
    total = response.headers.get("Content-Length")
    total_bytes = existing + int(total) if total and total.isdigit() else None
    downloaded = existing
    with partial.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                handle.write(chunk)
                downloaded += len(chunk)
                if total_bytes and downloaded % (64 * chunk_size) < len(chunk):
                    print(f"downloaded {downloaded / 1e6:.1f}/{total_bytes / 1e6:.1f} MB")
    partial.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return digest


def extract_safely(archive: zipfile.ZipFile, destination: Path) -> None:
    """拒绝绝对路径与 ../ 穿越成员后解压，防恶意压缩包路径穿越（zip-slip）。"""
    dest = destination.resolve()
    for member in archive.infolist():
        target = (dest / member.filename).resolve()
        if not target.is_relative_to(dest):
            raise ValueError(f"refusing unsafe zip member: {member.filename!r}")
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the official CMB archive")
    parser.add_argument("--url", default=DEFAULT_CMB_ZIP_URL)
    parser.add_argument("--output-dir", type=Path, default=PATHS.raw / "cmb")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    archive = args.output_dir.parent / "CMB-datasets.zip"
    if archive.exists() and not args.force:
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        print(f"using existing archive: {archive} sha256={digest}")
    else:
        print(f"downloading CMB archive to {archive}")
        digest = download_resumable(args.url, archive)
        print(f"download complete sha256={digest}")

    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as archive_file:
        extract_safely(archive_file, args.output_dir)
    manifest = {
        "source_url": args.url,
        "archive": str(archive),
        "archive_sha256": digest,
        "extracted_to": str(args.output_dir),
    }
    manifest_path = PATHS.manifests / "cmb_download.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"extracted to {args.output_dir}")


if __name__ == "__main__":
    main()

