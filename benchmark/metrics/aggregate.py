"""Validate and aggregate Phase 3 metric rows against the test-retest floor."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
for path in (HERE, BENCH):
    if path not in sys.path:
        sys.path.insert(0, path)

from contract import CONTRACT_SHA256, EXPERIMENT_ID
from protocol_v2 import PREPROCESSING_SHA256, PROTOCOL_ID
from schema_v3 import (
    COLUMNS, KEY_COLUMNS, RESULT_SCHEMA, classify_method, make_truth_unit_id,
    reference_frame_id,
)


RECORDING = re.compile(r"G(\d+)Day(\d+)Rest(\d+)", re.I)
SHA256 = re.compile(r"[0-9a-f]{64}")


def parse_recording(recording):
    match = RECORDING.search(recording)
    return None if not match else ("G" + match.group(1), int(match.group(2)), int(match.group(3)))


def finite_number(value, field, row_number):
    try:
        number = float(value)
    except Exception as error:
        raise ValueError(f"row {row_number}: {field} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"row {row_number}: {field} is non-finite: {value!r}")
    return number


def load_validated_rows(path):
    """Load only complete corrected-v2 / result-v3 rows; legacy mixing is impossible."""
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError(
                f"incompatible result schema in {path}; expected exact Phase 3 columns"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"no result rows in {path}")

    seen = set()
    run_ids = set()
    truth_units = set()
    truth_by_id = {}
    method_units = defaultdict(set)
    metric_provenance = {}
    for row_number, row in enumerate(rows, start=2):
        if row["result_schema"] != RESULT_SCHEMA:
            raise ValueError(f"row {row_number}: legacy/mixed result_schema {row['result_schema']!r}")
        if not SHA256.fullmatch(row["run_id"]):
            raise ValueError(f"row {row_number}: invalid run_id")
        run_ids.add(row["run_id"])
        if row["unit_status"] != "success":
            raise ValueError(f"row {row_number}: non-success result belongs in the unit ledger")
        if row["protocol_id"] != PROTOCOL_ID:
            raise ValueError(f"row {row_number}: legacy/mixed protocol_id {row['protocol_id']!r}")
        if row["experiment_id"] != EXPERIMENT_ID:
            raise ValueError(f"row {row_number}: mixed experiment_id")
        if row["scientific_contract_sha256"] != CONTRACT_SHA256:
            raise ValueError(f"row {row_number}: mixed scientific contract")
        if row["preprocessing_sha256"] != PREPROCESSING_SHA256:
            raise ValueError(f"row {row_number}: preprocessing digest does not match corrected-v2")
        if not SHA256.fullmatch(row["stage0_cache_key"]):
            raise ValueError(f"row {row_number}: invalid Stage-0 cache key")
        if not row["metric"] or not row["submetric"] or not row["metric_implementation"]:
            raise ValueError(f"row {row_number}: incomplete metric provenance")
        try:
            metric_diagnostics = json.loads(row["metric_diagnostics_json"])
        except Exception as error:
            raise ValueError(f"row {row_number}: invalid metric diagnostics JSON") from error
        if not isinstance(metric_diagnostics, dict):
            raise ValueError(f"row {row_number}: metric diagnostics must be an object")
        if row["metric"] == "specparam_peaks":
            required_diagnostics = {
                "fit_status", "posterior_channel_count", "r_squared",
                "mean_absolute_error", "detected_peak_count", "alpha_peak_count",
            }
            if not required_diagnostics.issubset(metric_diagnostics):
                raise ValueError(f"row {row_number}: incomplete specparam fit diagnostics")
            if metric_diagnostics["fit_status"] != "success":
                raise ValueError(f"row {row_number}: unsuccessful specparam fit in success row")
            for field in ("r_squared", "mean_absolute_error"):
                if not math.isfinite(float(metric_diagnostics[field])):
                    raise ValueError(f"row {row_number}: non-finite specparam {field}")
        provenance_fields = (
            "metric_source_sha256", "metric_config_sha256", "metrics_common_sha256",
            "runner_sha256", "qc_sha256", "schema_sha256", "aggregator_sha256",
        )
        if any(not SHA256.fullmatch(row[field]) for field in provenance_fields):
            raise ValueError(f"row {row_number}: invalid source/config hash provenance")
        provenance = tuple(row[field] for field in provenance_fields)
        prior = metric_provenance.setdefault(row["metric"], provenance)
        if prior != provenance:
            raise ValueError(f"row {row_number}: mixed provenance for metric {row['metric']}")

        key = tuple(row[column] for column in KEY_COLUMNS)
        if key in seen:
            raise ValueError(f"row {row_number}: duplicate result key {key}")
        seen.add(key)
        identity = parse_recording(row["recording"])
        if identity is None or identity[0].upper() != row["subject"].upper():
            raise ValueError(f"row {row_number}: recording/subject identity mismatch")

        truth = finite_number(row["truth"], "truth", row_number)
        value = finite_number(row["value"], "value", row_number)
        absolute_error = finite_number(row["abs_err"], "abs_err", row_number)
        unit = (row["recording"], row["drop_set"], row["metric"], row["submetric"])
        expected_frame = reference_frame_id(row["drop_set"])
        if row["reference_frame_id"] != expected_frame:
            raise ValueError(f"row {row_number}: wrong drop-set reference frame")
        expected_truth_id = make_truth_unit_id(
            row["recording"], row["stage0_cache_key"], row["drop_set"],
            row["metric"], row["submetric"],
        )
        if row["truth_unit_id"] != expected_truth_id:
            raise ValueError(f"row {row_number}: invalid truth_unit_id")

        if row["kind"] == "truth":
            expected = {
                "method": "-", "comparator_class": "truth",
                "reconstruction_key": "-", "gate_status": "not_applicable",
                "power_ratio_1_45": "-", "rms_ratio": "-", "max_abs_uv": "-",
            }
            for field, expected_value in expected.items():
                if row[field] != expected_value:
                    raise ValueError(f"row {row_number}: truth {field} must be {expected_value!r}")
            if abs(truth - value) > 1e-9 or absolute_error != 0:
                raise ValueError(f"row {row_number}: malformed truth identity row")
            truth_units.add(unit)
            truth_by_id[row["truth_unit_id"]] = truth
        elif row["kind"] == "recon":
            expected_class = classify_method(row["method"])
            if row["comparator_class"] != expected_class:
                raise ValueError(
                    f"row {row_number}: {row['method']} must be classed as {expected_class}"
                )
            if row["drop_set"] == "-" or row["gate_status"] != "pass":
                raise ValueError(f"row {row_number}: reconstructed row was not gated successfully")
            if not SHA256.fullmatch(row["reconstruction_key"]):
                raise ValueError(f"row {row_number}: invalid reconstruction key")
            finite_number(row["power_ratio_1_45"], "power_ratio_1_45", row_number)
            finite_number(row["rms_ratio"], "rms_ratio", row_number)
            finite_number(row["max_abs_uv"], "max_abs_uv", row_number)
            if abs(absolute_error - abs(value - truth)) > 2.1e-6:
                raise ValueError(f"row {row_number}: abs_err is inconsistent with truth/value")
            method_units[row["method"]].add(unit)
        else:
            raise ValueError(f"row {row_number}: invalid kind {row['kind']!r}")

    if len(run_ids) != 1:
        raise ValueError("result CSV mixes multiple run IDs")
    for row_number, row in enumerate(rows, start=2):
        if row["kind"] == "recon":
            expected_truth = truth_by_id.get(row["truth_unit_id"])
            if expected_truth is None:
                raise ValueError(f"row {row_number}: reconstruction has no matching truth row")
            if abs(float(row["truth"]) - expected_truth) > 1e-9:
                raise ValueError(f"row {row_number}: reconstruction truth differs from truth unit")

    if not method_units:
        raise ValueError("no reconstructed method rows are present")
    for method, units in method_units.items():
        missing = truth_units - units
        extra = units - truth_units
        if missing or extra:
            raise ValueError(
                f"method {method!r} is incomplete relative to truth: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
    return rows


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/metric_eval_v3.csv")
    parser.add_argument(
        "--include-oracles", action="store_true",
        help="show explicitly labeled oracle comparators (excluded by default)",
    )
    args = parser.parse_args()
    rows = load_validated_rows(args.csv)

    truth = defaultdict(dict)
    for row in rows:
        if row["kind"] != "truth":
            continue
        identity = parse_recording(row["recording"])
        truth[(row["metric"], row["submetric"], row["drop_set"], identity[0], identity[1])][identity[2]] = float(row["value"])
    floors = defaultdict(list)
    for (metric, submetric, drop_set, _subject, _day), values in truth.items():
        if 1 in values and 2 in values:
            floors[(metric, submetric, drop_set)].append(abs(values[1] - values[2]))

    errors = defaultdict(list)
    for row in rows:
        if row["kind"] == "recon":
            errors[(row["metric"], row["submetric"], row["drop_set"], row["method"])].append(float(row["abs_err"]))

    all_methods = sorted({row["method"] for row in rows if row["kind"] == "recon"})
    excluded_oracles = [method for method in all_methods if classify_method(method) == "oracle"]
    methods = [
        method for method in all_methods
        if args.include_oracles or classify_method(method) != "oracle"
    ]
    order = []
    seen = set()
    for row in rows:
        key = (row["metric"], row["submetric"], row["drop_set"])
        if key not in seen:
            seen.add(key)
            order.append(key)

    width = 15
    print(f"\nValidated Phase 3 preservation vs same-day test-retest floor  ({args.csv})\n")
    if excluded_oracles and not args.include_oracles:
        print(
            "Excluded oracle comparator(s) from the main table: "
            + ", ".join(excluded_oracles)
            + "  (use --include-oracles to display)\n"
        )
    heading = f"{'metric / submetric':<34}{'floor':>9}  " + "".join(
        f"{method:>{width}}" for method in methods
    )
    print(heading)
    print("-" * len(heading))
    for metric, submetric, drop_set in order:
        paired = floors[(metric, submetric, drop_set)]
        if not paired:
            raise ValueError(
                f"no same-day Rest1/Rest2 floor pairs for {metric}/{submetric}/{drop_set}"
            )
        floor = mean(paired)
        line = f"{metric + ' / ' + submetric:<34}{floor:>9.3f}  "
        for method in methods:
            error = mean(errors[(metric, submetric, drop_set, method)])
            marker = "" if math.isnan(error) or math.isnan(floor) else ("  ok" if error < floor else " OVER")
            line += f"{error:>{width - 5}.3f}{marker}"
        print(line + f"   (n={len(paired)} paired days; {drop_set})")
    print("-" * len(heading))
    print("ok = error below the same-day Rest1/Rest2 floor; OVER = error at or above that floor\n")


if __name__ == "__main__":
    main()
