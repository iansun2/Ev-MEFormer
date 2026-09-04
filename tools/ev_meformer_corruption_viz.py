#!/usr/bin/env python3
"""Build thesis-oriented corruption visualization reports for Ev-MEFormer.

The script intentionally depends only on the Python standard library.  Existing
probe runs may already contain PNG plots from ``tools/visualize_edl_logs.py``;
this wrapper turns their metrics and summaries into one compact HTML/Markdown
dashboard with inline SVG charts.
"""

import argparse
import csv
import html
import json
import math
import os
import re
from pathlib import Path


BASELINE = {
    "label": "MEFormer R50",
    "clean_nds": 0.7096,
    "clean_map": 0.6768,
    "mra_nds": 0.8422,
    "mra_map": 0.7470,
    "mrra_nds": 1.251,
}

V10_REV20A_PRESET = {
    "label": "Ev-MEFormer-UMR",
    "clean_nds": 0.6883,
    "clean_map": 0.6425,
    "mra_nds": 0.8521,
    "mra_map": 0.7458,
    "mrra_nds": -1.886,
    "status": "final_thesis_stage2",
    "source": "ai_memory/thesis_experiment_archive_20260616_0944/70_THESIS_USABLE_RESULTS.md",
    "notes": [
        "Final Stage 2 model under the no-validation-policy-learning thesis protocol.",
        "UMR = Uncertainty-aware Modality Routing.",
        "Implementation alias: Rev20-A / V10 starting point.",
        "Clean-balanced result; not a claim of across-the-board corruption superiority.",
        "Known weak areas: beamsreducing, pointsreducing, spatialmisalignment, temporalmisalignment.",
    ],
}

SAFE_NOTES = [
    "Stage 2 final model is Ev-MEFormer-UMR.",
    "Rev20-A / V10 is an implementation alias, not the formal thesis name.",
    "V10+EDL-cue calibration is an ablation/negative result only, not the formal Stage 2 route.",
    "V10.2, V11, Router-B, and GCR policy-learning variants are diagnostic/supporting unless explicitly superseded.",
]


def fmt(value, digits=4):
    if value is None:
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_metrics_summary(path):
    if not path or not Path(path).is_file():
        return {}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    data = {
        "metrics_summary": str(path),
        "severity_rows": [],
        "ra_nds_per_c": {},
        "ra_map_per_c": {},
        "rra_nds_per_c": {},
    }

    clean = re.search(
        r"Clean baseline:\s*NDS\s*=\s*([0-9.]+),\s*mAP\s*=\s*([0-9.]+)",
        text,
    )
    if not clean:
        clean = re.search(
            r"Your model:\s*Clean NDS\s*=\s*([0-9.]+),\s*Clean mAP\s*=\s*([0-9.]+)",
            text,
        )
    if clean:
        data["clean_nds"] = float(clean.group(1))
        data["clean_map"] = float(clean.group(2))

    for line in text.splitlines():
        row = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)\s+([123])\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
            line,
        )
        if row:
            data["severity_rows"].append({
                "corruption": row.group(1),
                "severity": int(row.group(2)),
                "nds": float(row.group(3)),
                "map": float(row.group(4)),
                "ra_nds": float(row.group(5)),
                "ra_map": float(row.group(6)),
            })
            continue

        ra = re.match(
            r"^\s*>> RA_([A-Za-z_][A-Za-z0-9_]*)\s+([0-9.]+)\s+([0-9.]+)\s*$",
            line,
        )
        if ra:
            data["ra_nds_per_c"][ra.group(1)] = float(ra.group(2))
            data["ra_map_per_c"][ra.group(1)] = float(ra.group(3))
            continue

        mra = re.match(r"^\s*mRA_targeted\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
        if mra:
            data["mra_nds"] = float(mra.group(1))
            data["mra_map"] = float(mra.group(2))
            continue

        rra = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)\s+([0-9.]+)\s+([0-9.]+)\s+([+-][0-9.]+)\s*$",
            line,
        )
        if rra:
            data["rra_nds_per_c"][rra.group(1)] = float(rra.group(4))
            continue

        mrra = re.match(r"^\s*mRRA_targeted\s+([+-][0-9.]+)\s*$", line)
        if mrra:
            data["mrra_nds"] = float(mrra.group(1))

    if not data["ra_nds_per_c"] and data["severity_rows"]:
        by_corr = {}
        for row in data["severity_rows"]:
            by_corr.setdefault(row["corruption"], []).append(row)
        for corr, rows in by_corr.items():
            data["ra_nds_per_c"][corr] = sum(r["ra_nds"] for r in rows) / len(rows)
            data["ra_map_per_c"][corr] = sum(r["ra_map"] for r in rows) / len(rows)
        if "mra_nds" not in data:
            data["mra_nds"] = sum(data["ra_nds_per_c"].values()) / len(data["ra_nds_per_c"])
        if "mra_map" not in data:
            data["mra_map"] = sum(data["ra_map_per_c"].values()) / len(data["ra_map_per_c"])

    return data


def parse_run_dir(run_dir, label=None):
    root = Path(run_dir)
    record = {
        "label": label or root.name,
        "run_dir": str(root),
        "status": "probe_run",
        "notes": [],
    }

    metrics_path = root / "metrics_summary.txt"
    record.update(parse_metrics_summary(metrics_path))

    robustness_path = root / "visualizations" / "summary" / "robustness_metrics.json"
    if robustness_path.is_file():
        robust = read_json(robustness_path)
        record["robustness_metrics"] = str(robustness_path)
        for key in ("mra_nds", "mra_map", "mrra_nds"):
            if key in robust:
                record[key] = robust[key]
        for key in ("ra_nds_per_c", "ra_map_per_c", "rra_nds_per_c"):
            if key in robust:
                record[key] = robust[key]
        record["baseline_name"] = robust.get("baseline_name")
        record["baseline_clean_nds"] = robust.get("baseline_clean_nds")

    summary_path = root / "visualizations" / "summary" / "visualization_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        record["visualization_summary"] = str(summary_path)
        record["conditions"] = list(summary.get("conditions", {}).keys())
        for key in ("v8_summary", "v9_summary"):
            if key in summary:
                record[key] = summary[key]

    summary_png_dir = root / "visualizations" / "summary"
    qual_png_dir = root / "visualizations" / "qualitative"
    if summary_png_dir.is_dir():
        record["summary_pngs"] = sorted(str(p) for p in summary_png_dir.glob("*.png"))
    if qual_png_dir.is_dir():
        record["qualitative_pngs"] = sorted(str(p) for p in qual_png_dir.glob("*.png"))
    return record


def load_manual_metrics(path):
    obj = read_json(path)
    records = obj if isinstance(obj, list) else [obj]
    out = []
    for idx, rec in enumerate(records):
        item = dict(rec)
        item.setdefault("label", f"manual_{idx}")
        item.setdefault("status", "manual_metrics")
        item.setdefault("manual_metrics", str(path))
        out.append(item)
    return out


def color_for_ratio(value, low=0.4, high=1.0):
    if value is None or math.isnan(float(value)):
        return "#e5e7eb"
    value = max(low, min(high, float(value)))
    t = (value - low) / (high - low)
    if t < 0.5:
        u = t / 0.5
        r, g, b = 239, int(101 + 80 * u), 72
    else:
        u = (t - 0.5) / 0.5
        r, g, b = int(251 - 217 * u), int(191 - 94 * u), int(36 + 58 * u)
    return f"#{r:02x}{g:02x}{b:02x}"


def bar_svg(title, items, width=900, height=340, zero_line=None, suffix=""):
    items = [(str(k), safe_float(v)) for k, v in items if safe_float(v) is not None]
    if not items:
        return "<p class=\"muted\">No data.</p>"
    pad_l, pad_r, pad_t, pad_b = 72, 26, 48, 84
    chart_w = width - pad_l - pad_r
    chart_h = height - pad_t - pad_b
    values = [v for _, v in items]
    min_v = min(values + ([zero_line] if zero_line is not None else []))
    max_v = max(values + ([zero_line] if zero_line is not None else []))
    if min_v == max_v:
        min_v -= 0.1
        max_v += 0.1
    if zero_line is None and min_v > 0:
        min_v = 0.0
    span = max_v - min_v
    step = chart_w / len(items)
    bar_w = max(18, min(54, step * 0.58))

    def y_of(v):
        return pad_t + chart_h - ((v - min_v) / span) * chart_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{width/2:.1f}" y="24" text-anchor="middle" class="svg-title">{html.escape(title)}</text>',
        f'<line x1="{pad_l}" y1="{pad_t + chart_h}" x2="{width-pad_r}" y2="{pad_t + chart_h}" class="axis"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+chart_h}" class="axis"/>',
    ]
    if zero_line is not None and min_v < zero_line < max_v:
        y0 = y_of(zero_line)
        parts.append(f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{width-pad_r}" y2="{y0:.1f}" class="zero"/>')

    palette = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626", "#0891b2", "#4b5563"]
    for idx, (name, value) in enumerate(items):
        x = pad_l + idx * step + (step - bar_w) / 2
        y = y_of(max(value, zero_line if zero_line is not None else min_v))
        y_base = y_of(zero_line if zero_line is not None else min_v)
        h = abs(y_base - y_of(value))
        if zero_line is not None and value < zero_line:
            y = y_base
        color = palette[idx % len(palette)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="value">{fmt(value, 3)}{suffix}</text>')
        label = html.escape(name)
        parts.append(
            f'<text x="{x + bar_w/2:.1f}" y="{height-20}" text-anchor="end" '
            f'transform="rotate(-35 {x + bar_w/2:.1f} {height-20})" class="tick">{label}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def heatmap_svg(record, title, metric="ra_nds", width=900):
    rows = record.get("severity_rows") or []
    if not rows:
        return "<p class=\"muted\">No severity-level rows available.</p>"
    corruptions = sorted({r["corruption"] for r in rows})
    cell_w, cell_h = 96, 42
    left, top = 180, 54
    height = top + len(corruptions) * cell_h + 42
    width = max(width, left + 3 * cell_w + 32)
    by_key = {(r["corruption"], int(r["severity"])): r for r in rows}
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{width/2:.1f}" y="24" text-anchor="middle" class="svg-title">{html.escape(title)}</text>',
    ]
    for s in (1, 2, 3):
        x = left + (s - 1) * cell_w
        parts.append(f'<text x="{x + cell_w/2}" y="{top - 14}" text-anchor="middle" class="tick">S{s}</text>')
    for i, corr in enumerate(corruptions):
        y = top + i * cell_h
        parts.append(f'<text x="{left-12}" y="{y + 27}" text-anchor="end" class="tick">{html.escape(corr)}</text>')
        for s in (1, 2, 3):
            x = left + (s - 1) * cell_w
            row = by_key.get((corr, s), {})
            val = safe_float(row.get(metric))
            color = color_for_ratio(val)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-4}" height="{cell_h-4}" rx="4" fill="{color}"/>')
            parts.append(f'<text x="{x + cell_w/2 - 2}" y="{y + 25}" text-anchor="middle" class="heat">{fmt(val, 3)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def html_table(headers, rows):
    out = ["<table>", "<thead><tr>"]
    for h in headers:
        out.append(f"<th>{html.escape(h)}</th>")
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>")
        for cell in row:
            out.append(f"<td>{html.escape(str(cell))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def rel_path(path, out_dir):
    try:
        return os.path.relpath(path, out_dir)
    except ValueError:
        return path


def collect_key_summary(record):
    rows = []
    v8 = record.get("v8_summary") or {}
    v9 = record.get("v9_summary") or {}
    interesting = [
        ("v8", "mean_v8_fused_keep"),
        ("v8", "mean_v8_center_fused_keep"),
        ("v8", "mean_v8_bev_reliability"),
        ("v8", "mean_v8_img_reliability"),
        ("v8", "mean_v8_bev_damage"),
        ("v8", "mean_v8_img_damage"),
        ("v8", "mean_v8_soft_weight_ratio"),
        ("v9", "mean_v9_a_bev"),
        ("v9", "mean_v9_a_img"),
        ("v9", "mean_v9_clean_like_mask"),
        ("v9", "mean_v9_velocity_l1"),
    ]
    for group, key in interesting:
        source = v8 if group == "v8" else v9
        if key in source:
            rows.append((key, source[key]))
    return rows


def write_outputs(records, out_dir, title):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"title": title, "records": records, "notes": SAFE_NOTES}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "status", "clean_nds", "clean_map", "mra_nds", "mra_map", "mrra_nds", "source"])
        for rec in records:
            writer.writerow([
                rec.get("label"),
                rec.get("status"),
                rec.get("clean_nds"),
                rec.get("clean_map"),
                rec.get("mra_nds"),
                rec.get("mra_map"),
                rec.get("mrra_nds"),
                rec.get("source") or rec.get("run_dir") or rec.get("manual_metrics") or "",
            ])

    md = []
    md.append(f"# {title}\n")
    md.append("## Thesis Guardrails\n")
    for note in SAFE_NOTES:
        md.append(f"- {note}\n")
    md.append("\n## Model Summary\n")
    md.append("| model | status | clean NDS | clean mAP | mRA NDS | mRA mAP | mRRA NDS (%) |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for rec in records:
        md.append(
            f"| {rec.get('label')} | {rec.get('status','')} | {fmt(rec.get('clean_nds'))} | "
            f"{fmt(rec.get('clean_map'))} | {fmt(rec.get('mra_nds'))} | {fmt(rec.get('mra_map'))} | "
            f"{fmt(rec.get('mrra_nds'), 3)} |\n"
        )
    md.append("\n## Generated Files\n")
    md.append("- `index.html`\n- `summary.json`\n- `summary.csv`\n")
    (out_dir / "report.md").write_text("".join(md), encoding="utf-8")

    rows = []
    for rec in records:
        rows.append([
            rec.get("label", ""),
            rec.get("status", ""),
            fmt(rec.get("clean_nds")),
            fmt(rec.get("clean_map")),
            fmt(rec.get("mra_nds")),
            fmt(rec.get("mra_map")),
            fmt(rec.get("mrra_nds"), 3),
        ])

    clean_items = [(rec.get("label", ""), rec.get("clean_nds")) for rec in records if rec.get("clean_nds") is not None]
    mra_items = [(rec.get("label", ""), rec.get("mra_nds")) for rec in records if rec.get("mra_nds") is not None]
    mrra_items = [(rec.get("label", ""), rec.get("mrra_nds")) for rec in records if rec.get("mrra_nds") is not None]

    sections = [
        "<!doctype html><html><head><meta charset=\"utf-8\">",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Inter,Arial,sans-serif;margin:0;background:#f7f7f4;color:#1f2937;}",
        "main{max-width:1180px;margin:0 auto;padding:32px 24px 64px;}",
        "h1{font-size:30px;margin:0 0 8px;} h2{margin-top:34px;font-size:20px;}",
        ".lead{color:#4b5563;margin-bottom:24px;}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px;}",
        ".card{background:white;border:1px solid #deded8;border-radius:8px;padding:18px;box-shadow:0 1px 2px rgba(0,0,0,.04);}",
        "table{border-collapse:collapse;width:100%;background:white;}th,td{border-bottom:1px solid #e5e7eb;padding:9px 10px;text-align:left;font-size:14px;}th{background:#efefea;}",
        ".note{border-left:4px solid #2563eb;background:#eef4ff;padding:12px 14px;margin:14px 0;border-radius:4px;}",
        ".muted{color:#6b7280}.svg-title{font-size:18px;font-weight:700;fill:#111827}.axis{stroke:#6b7280;stroke-width:1}.zero{stroke:#111827;stroke-width:1;stroke-dasharray:4 4}.tick{font-size:12px;fill:#374151}.value{font-size:12px;fill:#111827}.heat{font-size:13px;fill:#111827;font-weight:700}",
        ".gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}.gallery img{width:100%;background:white;border:1px solid #ddd;border-radius:6px}",
        "</style></head><body><main>",
        f"<h1>{html.escape(title)}</h1>",
        "<p class=\"lead\">Thesis-oriented corruption visualization summary for Ev-MEFormer.</p>",
        "<div class=\"note\"><strong>Guardrails.</strong><ul>",
    ]
    for note in SAFE_NOTES:
        sections.append(f"<li>{html.escape(note)}</li>")
    sections.append("</ul></div>")
    sections.append("<h2>Model Summary</h2>")
    sections.append(html_table(["model", "status", "clean NDS", "clean mAP", "mRA NDS", "mRA mAP", "mRRA NDS (%)"], rows))
    sections.append("<div class=\"grid\">")
    sections.append(f"<div class=\"card\">{bar_svg('Clean NDS', clean_items)}</div>")
    sections.append(f"<div class=\"card\">{bar_svg('mRA NDS', mra_items)}</div>")
    sections.append(f"<div class=\"card\">{bar_svg('mRRA NDS (%)', mrra_items, zero_line=0.0, suffix='%')}</div>")
    sections.append("</div>")

    for rec in records:
        label = rec.get("label", "model")
        sections.append(f"<h2>{html.escape(label)}</h2>")
        if rec.get("notes"):
            sections.append("<div class=\"note\"><ul>")
            for note in rec["notes"]:
                sections.append(f"<li>{html.escape(str(note))}</li>")
            sections.append("</ul></div>")
        ra = rec.get("ra_nds_per_c") or {}
        rra = rec.get("rra_nds_per_c") or {}
        if ra:
            sections.append("<div class=\"grid\">")
            sections.append(f"<div class=\"card\">{bar_svg('RA NDS by corruption', sorted(ra.items()))}</div>")
            if rra:
                sections.append(f"<div class=\"card\">{bar_svg('RRA NDS by corruption (%)', sorted(rra.items()), zero_line=0.0, suffix='%')}</div>")
            sections.append("</div>")
        if rec.get("severity_rows"):
            sections.append(f"<div class=\"card\">{heatmap_svg(rec, 'Severity RA NDS heatmap')}</div>")
        key_rows = collect_key_summary(rec)
        if key_rows:
            sections.append("<h3>Routing / Reliability Diagnostics</h3>")
            sections.append(html_table(["metric", "value"], [(k, json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else fmt(v)) for k, v in key_rows]))
        pngs = (rec.get("summary_pngs") or [])[:12]
        qpngs = (rec.get("qualitative_pngs") or [])[:12]
        if pngs or qpngs:
            sections.append("<h3>Existing Probe Images</h3><div class=\"gallery\">")
            for p in pngs + qpngs:
                rp = rel_path(p, out_dir)
                sections.append(f'<a href="{html.escape(rp)}"><img src="{html.escape(rp)}" alt="{html.escape(Path(p).name)}"></a>')
            sections.append("</div>")
    sections.append("</main></body></html>")
    (out_dir / "index.html").write_text("\n".join(sections), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build Ev-MEFormer corruption visualization dashboard.")
    parser.add_argument("--run-dir", action="append", default=[], help="Probe output directory. Can be repeated.")
    parser.add_argument("--run-label", action="append", default=[], help="Label for each --run-dir, in order.")
    parser.add_argument("--manual-metrics", action="append", default=[], help="Manual JSON metrics file. Can be repeated.")
    parser.add_argument("--preset-umr", action="store_true", help="Include final thesis Ev-MEFormer-UMR aggregate metrics.")
    parser.add_argument("--preset-v10-rev20a", action="store_true", help="Backward-compatible alias for --preset-umr.")
    parser.add_argument("--include-meformer-r50", action="store_true", help="Include original MEFormer R50 reference aggregate metrics.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Ev-MEFormer Corruption Dashboard")
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    if args.include_meformer_r50:
        records.append(dict(BASELINE, status="reference_baseline", notes=["Original MEFormer R50 MultiCorrupt baseline reference."]))
    if args.preset_umr or args.preset_v10_rev20a:
        records.append(dict(V10_REV20A_PRESET))
    for i, run_dir in enumerate(args.run_dir):
        label = args.run_label[i] if i < len(args.run_label) else None
        records.append(parse_run_dir(run_dir, label))
    for path in args.manual_metrics:
        records.extend(load_manual_metrics(path))
    if not records:
        raise SystemExit("No records supplied. Use --run-dir, --manual-metrics, or --preset-umr.")
    write_outputs(records, args.output_dir, args.title)
    print(f"Wrote dashboard: {Path(args.output_dir) / 'index.html'}")
    print(f"Wrote report:    {Path(args.output_dir) / 'report.md'}")
    print(f"Wrote summary:   {Path(args.output_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
