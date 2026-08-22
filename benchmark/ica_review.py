"""Create an inspectable component-level ICA review table from Stage-0 v3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


COLUMNS = (
    "component", "excluded", "ocular", "muscle", "muscle_score",
    "max_abs_ocular_score", "strongest_topography_channels",
    "pca_explained_variance", "review_decision", "review_notes",
)


def component_rows(manifest: dict) -> list[dict]:
    meta = manifest["preprocessing_meta"]["artifact_components"]
    n_components = int(meta["n_components"])
    channels = list(meta["ica_channel_names"])
    topographies = np.asarray(meta["component_topographies"], dtype=float)
    muscle_scores = np.asarray(meta["muscle_scores"], dtype=float)
    variance = np.asarray(meta["pca_explained_variance"], dtype=float)
    if topographies.shape != (n_components, len(channels)):
        raise ValueError("ICA topography dimensions do not match component/channel counts")
    if muscle_scores.shape != (n_components,) or variance.size < n_components:
        raise ValueError("ICA score/variance dimensions do not match component count")
    ocular_scores = {
        name: np.asarray(values, dtype=float)
        for name, values in meta["ocular_scores"].items()
    }
    if any(values.shape != (n_components,) for values in ocular_scores.values()):
        raise ValueError("Ocular score dimensions do not match component count")
    excluded = set(int(value) for value in meta["excluded_components"])
    ocular = set(int(value) for value in meta["ocular_components"])
    muscle = set(int(value) for value in meta["muscle_components"])
    rows = []
    for component in range(n_components):
        strongest = np.argsort(np.abs(topographies[component]))[::-1][:5]
        rows.append({
            "component": component,
            "excluded": component in excluded,
            "ocular": component in ocular,
            "muscle": component in muscle,
            "muscle_score": float(muscle_scores[component]),
            "max_abs_ocular_score": max(
                (abs(float(values[component])) for values in ocular_scores.values()),
                default=0.0,
            ),
            "strongest_topography_channels": "+".join(channels[index] for index in strongest),
            "pca_explained_variance": float(variance[component]),
            "review_decision": "",
            "review_notes": "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage0_manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.stage0_manifest.read_text(encoding="utf-8"))
    rows = component_rows(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.out)
    print(json.dumps({
        "components": len(rows),
        "excluded": sum(row["excluded"] for row in rows),
        "review_csv": str(args.out.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
