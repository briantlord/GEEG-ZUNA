"""Verify Stage-0 entries and inspect (never auto-delete) cache locks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import stage0_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("entries", nargs="*", type=Path)
    parser.add_argument("--cache-root", type=Path, default=stage0_cache.DEFAULT_CACHE_ROOT)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--inspect-locks", action="store_true")
    args = parser.parse_args()

    if args.inspect_locks:
        for lock in sorted(args.cache_root.resolve().glob(".*.lock")):
            owner = lock / "owner.json"
            detail = json.loads(owner.read_text(encoding="utf-8")) if owner.is_file() else {}
            print(json.dumps({"lock": str(lock), "owner": detail}, sort_keys=True))

    for entry in args.entries:
        _data, _names, _positions, manifest = stage0_cache.verify_entry(
            entry, verify_raw=not args.skip_raw
        )
        print(json.dumps({
            "entry": str(entry.resolve()),
            "status": "pass",
            "cache_key_sha256": manifest["identity"]["cache_key_sha256"],
        }, sort_keys=True))


if __name__ == "__main__":
    main()
