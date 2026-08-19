from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import requests
from tqdm import tqdm

from config import DEFAULT_MODEL_ID, PATHS


def api_model_files(model_id: str, revision: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{model_id}"
    response = requests.get(url, params={"revision": revision}, timeout=60)
    response.raise_for_status()
    siblings = response.json().get("siblings", [])
    return [item for item in siblings if item.get("rfilename")]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(model_id: str, revision: str, filename: str, destination: Path, expected_size: int | None = None) -> str:
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}?download=true"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    response = requests.get(url, headers=headers, stream=True, timeout=(30, 30))
    if existing and response.status_code == 200:
        # The endpoint ignored Range; restart safely instead of appending a full file.
        response.close()
        existing = 0
        headers = {}
        response = requests.get(url, headers=headers, stream=True, timeout=(30, 30))
    response.raise_for_status()
    total = int(response.headers.get("content-length", "0")) + existing
    digest = hashlib.sha256()
    if existing:
        with partial.open("rb") as previous:
            for chunk in iter(lambda: previous.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    mode = "ab" if existing else "wb"
    with partial.open(mode) as handle, tqdm(total=total or expected_size or None, initial=existing, unit="B", unit_scale=True, desc=filename) as bar:
        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
            if chunk:
                handle.write(chunk)
                digest.update(chunk)
                bar.update(len(chunk))
    if expected_size and partial.stat().st_size != expected_size:
        raise IOError(f"incomplete download for {filename}: {partial.stat().st_size} != {expected_size}")
    os.replace(partial, destination)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model without git-lfs")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--include", nargs="*", default=None, help="Optional filename fragments to download")
    args = parser.parse_args()

    output_dir = args.output_dir or PATHS.models / args.model_id.replace("/", "--")
    files = api_model_files(args.model_id, args.revision)
    selected = []
    for item in files:
        name = item["rfilename"]
        if args.include and not any(fragment in name for fragment in args.include):
            continue
        selected.append(item)
    metadata = {"model_id": args.model_id, "revision": args.revision, "files": selected}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "download_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.metadata_only:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    digests = {}
    for item in selected:
        filename = item["rfilename"]
        destination = output_dir / filename
        oid = (item.get("lfs") or {}).get("oid")
        if destination.exists() and item.get("size") and destination.stat().st_size == item["size"]:
            if oid:
                # 大小相同还不够：HF API 提供 LFS sha256，本地校验后再复用。
                local_hash = sha256_file(destination)
                if local_hash == oid:
                    digests[filename] = local_hash
                    print(f"skip verified {filename}")
                    continue
                print(f"existing {filename} fails sha256; re-downloading", flush=True)
                destination.unlink()
            else:
                print(f"skip existing {filename} (size match, no hash available)")
                continue
        for attempt in range(1, 8):
            try:
                digests[filename] = download_file(args.model_id, args.revision, filename, destination, item.get("size"))
                break
            except (requests.RequestException, TimeoutError, IOError):
                # IOError 覆盖“流提前结束导致 .part 不完整”这一常见失败模式：
                # .part 保留，下一轮从断点续传。
                if attempt == 7:
                    raise
                print(f"download interrupted for {filename}; retry {attempt}/6 from saved partial", flush=True)
    metadata["downloaded_sha256"] = digests
    (output_dir / "download_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"model files ready at {output_dir}")


if __name__ == "__main__":
    main()
