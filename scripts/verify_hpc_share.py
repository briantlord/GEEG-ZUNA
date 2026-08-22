"""Verify an HPC share exactly matches its release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest_path = root / "RELEASE_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing release manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "geeg-zuna-release-v1":
        raise ValueError("Unsupported release-manifest schema")
    declared = {row["path"]: row for row in manifest.get("files", [])}
    if len(declared) != len(manifest.get("files", [])):
        raise ValueError("Release manifest contains duplicate paths")
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    missing = sorted(set(declared) - set(actual))
    extra = sorted(set(actual) - set(declared))
    mismatched = []
    for relative in sorted(set(declared) & set(actual)):
        row, path = declared[relative], actual[relative]
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            mismatched.append(relative)
    if missing or extra or mismatched:
        raise ValueError(
            "Release verification failed: "
            f"missing={missing[:3]}, extra={extra[:3]}, mismatched={mismatched[:3]}"
        )
    content_hash = hashlib.sha256(
        "\n".join(f"{declared[path]['sha256']}  {path}" for path in sorted(declared)).encode()
    ).hexdigest()
    if content_hash != manifest.get("content_hash"):
        raise ValueError("Release content hash does not match the manifest")
    return {
        "status": "pass", "files": len(actual), "content_hash": content_hash,
        "git_commit": manifest.get("git_commit"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("share", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.share), sort_keys=True))


if __name__ == "__main__":
    main()
