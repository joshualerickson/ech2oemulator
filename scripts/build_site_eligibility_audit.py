#!/usr/bin/env python3
"""Build a shareable PDF and CSV explaining ECH2O site eligibility."""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fixed_window_reason(screen: Path) -> str:
    """Match the 30-day June--September sequence eligibility rule."""
    rows = read_csv(screen)
    by_date = {date.fromisoformat(row["date"]): row for row in rows}
    target_days = [day for day, row in by_date.items()
                   if day.month in {6, 7, 8, 9} and row["target_qa_valid"] == "True"]
    if not target_days:
        return "no_target_QA-valid_Jun-Sep_day"
    for target in target_days:
        history = [by_date.get(target - timedelta(days=offset)) for offset in range(30)]
        if all(day is not None and day["forcing_valid"] == "True" for day in history):
            return "eligible"
    return "no_contiguous_30-day_forcing_history_for_QA-valid_summer_target"


def draw_page(pdf: PdfPages, title: str, sections: list[tuple[str, str]]) -> None:
    figure = Figure(figsize=(8.5, 11))
    axis = figure.add_axes((0.07, 0.06, 0.86, 0.88))
    axis.axis("off")
    axis.text(0, 1, title, fontsize=18, fontweight="bold", va="top", color="#17212b")
    y = 0.94
    for heading, body in sections:
        axis.text(0, y, heading, fontsize=12, fontweight="bold", va="top", color="#1769aa")
        y -= 0.028
        wrapped = "\n".join(textwrap.wrap(body, width=103))
        axis.text(0, y, wrapped, fontsize=9.5, va="top", linespacing=1.45, color="#24313d")
        y -= 0.035 * (wrapped.count("\n") + 1) + 0.026
    axis.text(0, 0, "ECH2O emulator eligibility audit • generated from persisted Phase 1 and manifest artifacts",
              fontsize=8, color="#52616d", va="bottom")
    pdf.savefig(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=Path("artifacts/schema_reports/phase1_schema.json"))
    parser.add_argument("--daily-dir", type=Path, default=Path("artifacts/manifests/phase2_daily"))
    parser.add_argument("--fixed-manifest", type=Path, default=Path("artifacts/manifests/full75_val25_jun_sep_v1/manifest.csv"))
    parser.add_argument("--external-manifest", type=Path, default=Path("artifacts/manifests/external_panel_v1/manifest.csv"))
    parser.add_argument("--water-manifest", type=Path, default=Path("artifacts/manifests/full75_val25_water_year_v1/manifest.csv"))
    parser.add_argument("--water-summary", type=Path, default=Path("artifacts/manifests/full75_val25_water_year_v1/summary.json"))
    parser.add_argument("--external-water-summary", type=Path, default=Path("artifacts/manifests/external_panel_water_year_v1/summary.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("artifacts/reports/site_eligibility_audit.csv"))
    parser.add_argument("--pdf-output", type=Path, default=Path("artifacts/reports/site_eligibility_summary.pdf"))
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text())
    raw_sites = {row["site_id"]: row for row in schema["sites"]}
    phase_issues = {site: "; ".join(row["issues"]) for site, row in raw_sites.items() if row["issues"]}
    fixed_sites = {row["site_id"] for row in read_csv(args.fixed_manifest)}
    external_sites = {row["site_id"] for row in read_csv(args.external_manifest)}
    water_sites = {row["site_id"] for row in read_csv(args.water_manifest)}
    water_exclusions = {row["site_id"]: row["reason"] for row in json.loads(args.water_summary.read_text())["excluded_sites"]}
    external_water_exclusions = {row["site_id"]: row["reason"] for row in json.loads(args.external_water_summary.read_text())["excluded_sites"]}
    screen_by_site = {path.name.split("_")[0]: path for path in args.daily_dir.glob("*_daily_screen.csv")}

    rows: list[dict[str, str]] = []
    for site in sorted(raw_sites):
        phase_status = "excluded" if site in phase_issues else "eligible_bundle"
        phase_reason = phase_issues.get(site, "")
        if site in phase_issues:
            fixed_status, fixed_reason = "excluded", f"phase1_schema: {phase_reason}"
        elif site in external_sites:
            fixed_status, fixed_reason = "held_out", "reserved external site-disjoint test panel"
        elif site in fixed_sites:
            fixed_status, fixed_reason = "used", "eligible 30-day Jun-Sep sequence; assigned train or spatial validation"
        else:
            fixed_status, fixed_reason = "excluded", fixed_window_reason(screen_by_site[site])

        if site in phase_issues:
            water_status, water_reason = "excluded", f"phase1_schema: {phase_reason}"
        elif site in external_sites:
            water_status = "held_out"
            water_reason = "reserved external site-disjoint test panel"
            if site in external_water_exclusions:
                water_reason += f"; full water-year replay unavailable: {external_water_exclusions[site]}"
        elif site not in fixed_sites:
            water_status, water_reason = "excluded", f"not in fixed-window cohort: {fixed_reason}"
        elif site in water_sites:
            water_status, water_reason = "used", "continuous Oct-01 through Sep-30 forcing path"
        else:
            water_status, water_reason = "excluded", water_exclusions[site]

        if fixed_status != "used" or water_status != "used":
            rows.append({
                "site_id": site, "state": site[:2], "phase1_bundle_status": phase_status,
                "phase1_reason": phase_reason, "jun_sep_30day_status": fixed_status,
                "jun_sep_30day_reason": fixed_reason, "full_water_year_bptt_status": water_status,
                "full_water_year_bptt_reason": water_reason,
            })

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fixed_excluded = Counter(row["jun_sep_30day_reason"] for row in rows if row["jun_sep_30day_status"] == "excluded")
    water_added = Counter(row["state"] for row in rows if row["jun_sep_30day_status"] == "used" and row["full_water_year_bptt_status"] == "excluded")
    raw_count, valid_bundle_count = len(raw_sites), len(raw_sites) - len(phase_issues)
    fixed_count, water_count = len(fixed_sites), len(water_sites)
    held_out_count = len(external_sites)
    with PdfPages(args.pdf_output) as pdf:
        draw_page(pdf, "ECH2O site eligibility: June–September and full BPTT", [
            ("Purpose", "This audit explains why sites are included or excluded from the reproducible recurrent-model cohorts. It distinguishes data-contract failures, intentionally held-out external test sites, June–September fixed-window eligibility, and the stricter full-water-year BPTT eligibility rule."),
            ("Cohort flow", f"{raw_count} raw folders → {valid_bundle_count} valid Phase 1 bundles → {fixed_count} fixed-window train/validation sites → {water_count} full-water-year BPTT train/validation sites. The fixed-window cohort is 323 training plus 108 site-disjoint validation sites. The BPTT cohort is 290 training plus 97 validation sites."),
            ("Not a random removal", "The split itself is site-level and geographically/covariate balanced. A site is never dropped merely because of state, terrain, climate, bbox size, or clipping. The removal rules protect temporal alignment and prevent the recurrent state from silently crossing missing or invalid forcing days."),
        ])
        draw_page(pdf, "Why sites are not in a cohort", [
            ("Phase 1: 7 raw folders", "Two sites fail the expected forcing-to-target one-cell-inset relationship, and five have forcing/target band-count disagreement. These are excluded before daily QA or modeling."),
            (f"External panel: {held_out_count} sites", "These sites are intentionally excluded from both training and validation to retain a site-disjoint external test panel. They are not data-quality failures. One external CO site also has a full-water-year forcing gap, so it cannot be used for stateful Oct–Sep replay."),
            ("Fixed June–September: 73 additional sites", f"{fixed_excluded['no_contiguous_30-day_forcing_history_for_QA-valid_summer_target']} sites have at least one QA-valid summer target but no continuous 30-day forcing history ending on one; {fixed_excluded['no_target_QA-valid_Jun-Sep_day']} have no QA-valid June–September target day."),
            ("Full water-year BPTT: 44 additional sites", f"These were eligible for fixed windows but have a forcing gap or invalid forcing day somewhere from Oct. 1 through Sep. 30. By state: " + ", ".join(f"{state} {count}" for state, count in sorted(water_added.items())) + ". BPTT must reject them because the hidden state cannot be carried honestly across a gap."),
        ])
        draw_page(pdf, "How to use the accompanying CSV", [
            ("One row per excluded-or-held-out raw site", "The CSV contains every raw site not used by at least one of the two modeling protocols. It has separate status and reason columns for the June–September 30-day cohort and the full-water-year BPTT cohort."),
            ("Status meanings", "used = included in that protocol's train/validation cohort; held_out = reserved for the external site-disjoint test panel; excluded = failed a documented data/sequence eligibility rule. A site excluded from fixed windows is also unavailable for BPTT because BPTT begins from the fixed eligible cohort."),
            ("Source artifacts", "Phase 1 schema report; persisted full75_val25 June–September split; persisted full75_val25 water-year manifest and exclusion summary; and phase2 daily QA screens. The report is a cohort audit, not a claim that excluded sites are scientifically unusable after data repair."),
        ])
    print(json.dumps({"csv": str(args.csv_output), "pdf": str(args.pdf_output), "raw_sites": raw_count,
                      "fixed_window_sites": fixed_count, "full_bptt_sites": water_count,
                      "audit_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
