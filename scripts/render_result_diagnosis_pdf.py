#!/usr/bin/env python3
"""Render a colleague-ready PDF from the prediction-result gap diagnosis CSV."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def add_header(axis, title: str, subtitle: str = "") -> None:
    axis.set_axis_off()
    axis.text(0.02, 0.96, title, fontsize=21, fontweight="bold", color="#123b55", va="top")
    if subtitle:
        axis.text(0.02, 0.915, subtitle, fontsize=10.5, color="#536574", va="top")


def add_table(axis, columns: list[str], rows: list[list[str]], bbox: tuple[float, float, float, float]) -> None:
    table = axis.table(cellText=rows, colLabels=columns, cellLoc="left", colLoc="left", bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c8d4db")
        if row == 0:
            cell.set_facecolor("#e8f0f4")
            cell.set_text_props(weight="bold", color="#123b55")
        else:
            cell.set_facecolor("#ffffff")


def write_wrapped(axis, text: str, x: float, y: float, width: int = 108, fontsize: float = 10.5) -> None:
    import textwrap
    axis.text(x, y, "\n".join(textwrap.wrap(text, width=width)), fontsize=fontsize, va="top", linespacing=1.42)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--diagnosis-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.summary_csv.open(newline="", encoding="utf-8-sig") as handle:
        summary = list(csv.DictReader(handle))
    with args.diagnosis_csv.open(newline="", encoding="utf-8") as handle:
        diagnosis = list(csv.DictReader(handle))
    status = Counter(row["status"] for row in summary)
    by_diagnosis = Counter(row["diagnosis"] for row in diagnosis)
    total = len(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        figure, axis = plt.subplots(figsize=(8.5, 11))
        add_header(axis, "ECH2O 30 m prediction-result diagnosis", "Prepared 2026-09-01  |  Scope: 4,472 MTBS records")
        axis.add_patch(plt.Rectangle((0.02, 0.72), 0.96, 0.13, facecolor="#eef6f8", edgecolor="#287c91", linewidth=1.2))
        write_wrapped(axis, "Bottom line: 82.4% of records have complete results. Most remaining gaps are due to unavailable input folders or explicit fixed-window eligibility rules—not widespread model failure. Of 55 all-NA records, five have valid current support and can be rerun.", 0.045, 0.825, 115, 11)
        rows = [[name.title(), f"{count:,}", f"{count / total:.1%}", explanation] for name, count, explanation in [
            ("good", status["good"], "All five prediction targets contain valid results."),
            ("missing", status["missing"], "No usable prediction product was available."),
            ("all NA", status["all_na"], "No supported pixels remained after masking."),
        ]]
        add_table(axis, ["Summary status", "Records", "Share", "Meaning"], rows, (0.02, 0.47, 0.96, 0.19))
        axis.text(0.02, 0.42, "Interpretation", fontsize=14, fontweight="bold", color="#123b55")
        write_wrapped(axis, "The production model uses a causal 90-day forcing sequence. A pixel is exported only where every selected static channel and every forcing channel are finite throughout that history. This strict mask prevents undocumented imputation or bridging missing daily inputs.", 0.02, 0.38)
        axis.text(0.02, 0.24, "Audit inputs", fontsize=14, fontweight="bold", color="#123b55")
        write_wrapped(axis, "The diagnosis reconciled the supplied status summary with the corrected worker-input folders, original 240 m products, direct 30 m products, VIIRS date windows, static support, and causal 90-day forcing support. No prediction values or model weights were changed.", 0.02, 0.20)
        pdf.savefig(figure); plt.close(figure)

        figure, axis = plt.subplots(figsize=(8.5, 11))
        add_header(axis, "Missing results: 733 records", "Three mutually exclusive operational/data-availability causes")
        rows = [
            ["Missing input folder", f"{by_diagnosis['missing_input_folder']:,}", "Absent from both corrected worker inputs and the original 240 m input collection. There is no state-prefix gate; prediction can begin once a site grid and valid raster coverage exist."],
            ["Insufficient Seq90 forcing history", f"{by_diagnosis['no_240m_prediction_output']:,}", "Input exists, but the final 240 m run explicitly skipped it. Calendar-year forcing starts January 1, so early-year dates lack the required 90 prior forcing days."],
        ]
        add_table(axis, ["Diagnosis", "Count", "Interpretation"], rows, (0.02, 0.51, 0.96, 0.34))
        axis.text(0.02, 0.45, "Important distinction", fontsize=14, fontweight="bold", color="#123b55")
        write_wrapped(axis, "The 94 Seq90 cases are expected ineligibilities under the present Jan-1 forcing contract. They are not file-transfer failures. Supporting them requires prior-year forcing or a separately validated hidden-state/initialization policy.", 0.02, 0.41)
        axis.text(0.02, 0.27, "Recommended handling", fontsize=14, fontweight="bold", color="#123b55")
        write_wrapped(axis, "1. Produce source input folders for the 639 absent-input events before attempting inference.  2. Assess eligibility using the actual site grid and raster support rather than its state prefix.  3. Keep the 94 early-year events explicitly ineligible unless the temporal input contract changes.", 0.02, 0.23)
        pdf.savefig(figure); plt.close(figure)

        figure, axis = plt.subplots(figsize=(8.5, 11))
        add_header(axis, "All-NA results: 55 records", "Support-mask diagnosis and immediate remediation")
        rows = [
            ["No complete 90-day dynamic support", f"{by_diagnosis['no_complete_90_day_dynamic_support']:,}", "No pixel remains finite across all forcing channels and all 90 days. Inspected examples show sparse tmin nodata on one or more dates; the full-window intersection then becomes empty."],
            ["No complete static support", f"{by_diagnosis['no_complete_static_support']:,}", "CA3335911836420070510 has zero valid TWI pixels in the selected static stack."],
            ["Current support exists; stale all-NA product", f"{by_diagnosis['support_exists_unexpected_all_na']:,}", "Current native 30 m statics and causal forcing support overlap; the stored products should be rerun with the current direct-30 m workflow."],
        ]
        add_table(axis, ["Diagnosis", "Count", "Evidence and action"], rows, (0.02, 0.49, 0.96, 0.36))
        axis.text(0.02, 0.43, "Five rerun candidates", fontsize=14, fontweight="bold", color="#123b55")
        axis.text(0.04, 0.385, "CA3260811624320120517\nAZ3134111110120110429\nAZ3135011111020160516\nNM3134910867820180516\nAZ3135511114320210620", family="monospace", fontsize=10, va="top")
        axis.text(0.02, 0.19, "Recommended sequence", fontsize=14, fontweight="bold", color="#123b55")
        write_wrapped(axis, "Rerun the five listed sites first. Repair source tmin nodata for the 49 dynamic-support cases before rerunning; do not silently impute in the model pipeline without a documented QA decision. Repair/replace TWI coverage for the one static-support case.", 0.02, 0.15)
        pdf.savefig(figure); plt.close(figure)


if __name__ == "__main__":
    main()
