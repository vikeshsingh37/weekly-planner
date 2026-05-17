#!/usr/bin/env python3
"""
Visualize eval results from a saved JSON report.

Usage:
    uv run python eval_data/visualize.py                        # latest in eval_results/
    uv run python eval_data/visualize.py eval_results/20260516_143022.json
    uv run python eval_data/visualize.py --html                 # also save HTML report
    uv run python eval_data/visualize.py --compare a.json b.json

Output:
    Terminal: summary table + per-case grid
    HTML:     eval_results/<stem>_report.html  (with --html flag)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


# ── ANSI helpers ──────────────────────────────────────────────────────────────

_GREEN = "\033[92m"
_RED   = "\033[91m"
_YELLOW = "\033[93m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_RESET = "\033[0m"

def _c(text: str, *codes: str) -> str:
    return "".join(codes) + str(text) + _RESET


def _pass_fail(ok: bool) -> str:
    return _c("PASS", _GREEN, _BOLD) if ok else _c("FAIL", _RED, _BOLD)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def latest_report() -> str:
    results_dir = Path("eval_results")
    if not results_dir.exists():
        sys.exit("No eval_results/ directory found. Run python run_evals.py first.")
    files = sorted(results_dir.glob("*.json"))
    if not files:
        sys.exit("No result files found in eval_results/.")
    return str(files[-1])


# ── Terminal rendering ────────────────────────────────────────────────────────

def _bar(fraction: float, width: int = 20) -> str:
    filled = round(fraction * width)
    bar = "█" * filled + "░" * (width - filled)
    color = _GREEN if fraction >= 0.9 else (_YELLOW if fraction >= 0.7 else _RED)
    return _c(bar, color)


def _col_widths(*cols: list[str]) -> list[int]:
    return [max(len(str(v)) for v in col) for col in cols]


def print_summary(report: dict, label: str = "") -> None:
    s = report["summary"]
    total, passed = s["total_cases"], s["passed_cases"]
    pct = s["overall_pct"]
    chk_total, chk_pass = s["total_checks"], s["passed_checks"]
    chk_pct = s["check_pct"]

    title = f"Eval Results{f' — {label}' if label else ''}"
    print(f"\n{_c(title, _BOLD)}")
    print(f"  Cases  : {_bar(passed/total if total else 0)}  {passed}/{total}  ({pct:.1f}%)")
    print(f"  Checks : {_bar(chk_pass/chk_total if chk_total else 0)}  {chk_pass}/{chk_total}  ({chk_pct:.1f}%)")


def print_by_category(report: dict) -> None:
    by_cat = report.get("by_category", {})
    if not by_cat:
        return

    print(f"\n{_c('By Category', _BOLD)}")
    header = ("Category", "Pass", "Total", "Score", "Bar")
    rows = []
    for cat, data in sorted(by_cat.items()):
        p, t = data["passed"], data["total"]
        pct = data["pct"]
        rows.append((cat, str(p), str(t), f"{pct:.0f}%", _bar(p / t if t else 0, 15)))

    w = [max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(4)]
    sep = "  ".join("─" * wi for wi in w)
    fmt = "  ".join(f"{{:<{wi}}}" for wi in w)

    print("  " + fmt.format(*header[:4]) + "  " + header[4])
    print("  " + sep + "  " + "─" * 15)
    for row in rows:
        cat_colored = _c(row[0], _BOLD)
        print("  " + fmt.format(cat_colored, row[1], row[2], row[3]) + "  " + row[4])


def print_case_grid(report: dict) -> None:
    cases = report.get("cases", [])
    if not cases:
        return

    print(f"\n{_c('Per-Case Detail', _BOLD)}")

    # Determine column widths
    ids   = [c["name"] for c in cases]
    cats  = [c["category"] for c in cases]
    durs  = [f"{c['duration_seconds']:.1f}s" for c in cases]
    stats = ["PASS" if c["passed"] else "FAIL" for c in cases]

    w_id  = max(len("Case"), max(len(i) for i in ids))
    w_cat = max(len("Category"), max(len(c) for c in cats))
    w_dur = max(len("Time"), max(len(d) for d in durs))

    header = f"  {'Case':<{w_id}}  {'Category':<{w_cat}}  {'Time':>{w_dur}}  Status  Checks"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for case in cases:
        chks = case["checks"]
        n_pass = sum(1 for c in chks if c["passed"])
        n_total = len(chks)
        check_str = f"{n_pass}/{n_total}"
        status = _pass_fail(case["passed"])
        dur = f"{case['duration_seconds']:.1f}s"
        print(
            f"  {case['name']:<{w_id}}  {case['category']:<{w_cat}}  {dur:>{w_dur}}  {status}    {check_str}"
        )

        # Show failing check descriptions indented
        if not case["passed"]:
            for chk in chks:
                if not chk["passed"]:
                    msg = chk["description"][:80]
                    err = f" [{chk['error']}]" if chk.get("error") else ""
                    print(f"  {_c(f'    ✗ {msg}{err}', _DIM)}")


def print_comparison(reports: list[tuple[str, dict]]) -> None:
    """Side-by-side category scores for two reports."""
    if len(reports) != 2:
        return
    (la, ra), (lb, rb) = reports
    cats = sorted(set(ra.get("by_category", {})) | set(rb.get("by_category", {})))

    print(f"\n{_c('Comparison', _BOLD)}")
    w = max(len(c) for c in cats)
    print(f"  {'Category':<{w}}  {la:<12}  {lb:<12}  Delta")
    print("  " + "─" * (w + 36))
    for cat in cats:
        a = ra.get("by_category", {}).get(cat, {})
        b = rb.get("by_category", {}).get(cat, {})
        pa = a.get("pct", 0)
        pb = b.get("pct", 0)
        delta = pb - pa
        sym = _c(f"+{delta:.0f}%", _GREEN) if delta > 0 else (_c(f"{delta:.0f}%", _RED) if delta < 0 else _c("=", _DIM))
        print(f"  {cat:<{w}}  {pa:.0f}%{' ':9}  {pb:.0f}%{' ':9}  {sym}")


# ── HTML rendering ────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Weekly Planner Eval — {title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; background: #f8f8f8; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  .meta {{ color: #666; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; background: #fff;
           box-shadow: 0 1px 3px rgba(0,0,0,.1); border-radius: 8px; overflow: hidden; }}
  th {{ background: #1a1a1a; color: #fff; padding: 10px 14px; text-align: left; font-size: 0.85rem; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #eee; font-size: 0.87rem; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f0f4ff; }}
  .pass {{ color: #15803d; font-weight: 600; }}
  .fail {{ color: #b91c1c; font-weight: 600; }}
  .bar-wrap {{ background: #e5e7eb; border-radius: 4px; height: 10px; width: 120px; display:inline-block; vertical-align:middle; }}
  .bar-fill {{ height: 10px; border-radius: 4px; }}
  .bar-green {{ background: #16a34a; }}
  .bar-yellow {{ background: #ca8a04; }}
  .bar-red {{ background: #dc2626; }}
  .checks {{ font-size: 0.78rem; color: #555; margin-top: 4px; }}
  .check-fail {{ color: #b91c1c; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px; }}
  .summary-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1.5rem; }}
  .card {{ background:#fff; border-radius:8px; padding:1rem 1.25rem; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
  .card-label {{ font-size:0.78rem; color:#666; text-transform:uppercase; letter-spacing:.05em; }}
  .card-value {{ font-size:1.8rem; font-weight:700; margin:.2rem 0; }}
  .card-sub {{ font-size:0.82rem; color:#555; }}
</style>
</head>
<body>
<h1>Weekly Planner — Eval Report</h1>
<p class="meta">Generated {ts} · Source: {path}</p>

<div class="summary-grid">
  <div class="card">
    <div class="card-label">Cases</div>
    <div class="card-value">{passed_cases}/{total_cases}</div>
    <div class="card-sub">{overall_pct:.1f}% pass rate</div>
  </div>
  <div class="card">
    <div class="card-label">Checks</div>
    <div class="card-value">{passed_checks}/{total_checks}</div>
    <div class="card-sub">{check_pct:.1f}% pass rate</div>
  </div>
</div>

<h2>By Category</h2>
<table>
  <thead><tr><th>Category</th><th>Pass</th><th>Total</th><th>Score</th><th>Progress</th></tr></thead>
  <tbody>{cat_rows}</tbody>
</table>

<h2>Per-Case Detail</h2>
<table>
  <thead><tr><th>Case</th><th>Category</th><th>Status</th><th>Checks</th><th>Time</th></tr></thead>
  <tbody>{case_rows}</tbody>
</table>
</body>
</html>
"""


def _bar_html(fraction: float, width: int = 120) -> str:
    filled = round(fraction * width)
    cls = "bar-green" if fraction >= 0.9 else ("bar-yellow" if fraction >= 0.7 else "bar-red")
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar-fill {cls}" style="width:{filled}px"></div>'
        f'</div>'
    )


def render_html(report: dict, source_path: str) -> str:
    s = report["summary"]
    by_cat = report.get("by_category", {})
    cases = report.get("cases", [])

    cat_rows = ""
    for cat, data in sorted(by_cat.items()):
        p, t = data["passed"], data["total"]
        pct = data["pct"]
        cat_rows += (
            f"<tr><td>{cat}</td><td>{p}</td><td>{t}</td>"
            f"<td><b>{pct:.0f}%</b></td>"
            f"<td>{_bar_html(p / t if t else 0)}</td></tr>\n"
        )

    case_rows = ""
    for c in cases:
        chks = c["checks"]
        n_pass = sum(1 for ch in chks if ch["passed"])
        n_total = len(chks)
        status_cls = "pass" if c["passed"] else "fail"
        status_txt = "PASS" if c["passed"] else "FAIL"

        failing = "".join(
            f'<div class="check-fail">✗ {ch["description"][:90]}</div>'
            for ch in chks if not ch["passed"]
        )
        checks_cell = f"{n_pass}/{n_total}{f'<div class=\"checks\">{failing}</div>' if failing else ''}"

        case_rows += (
            f"<tr><td><code>{c['name']}</code></td>"
            f"<td>{c['category']}</td>"
            f'<td><span class="{status_cls}">{status_txt}</span></td>'
            f"<td>{checks_cell}</td>"
            f"<td>{c['duration_seconds']:.1f}s</td></tr>\n"
        )

    return _HTML_TEMPLATE.format(
        title=Path(source_path).stem,
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        path=source_path,
        passed_cases=s["passed_cases"],
        total_cases=s["total_cases"],
        overall_pct=s["overall_pct"],
        passed_checks=s["passed_checks"],
        total_checks=s["total_checks"],
        check_pct=s["check_pct"],
        cat_rows=cat_rows,
        case_rows=case_rows,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize eval results")
    parser.add_argument(
        "files",
        nargs="*",
        help="JSON result file(s). Omit to use the latest in eval_results/",
    )
    parser.add_argument("--html", action="store_true", help="Save an HTML report alongside the JSON")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        help="Side-by-side category comparison of two result files",
    )
    args = parser.parse_args()

    if args.compare:
        reports = [(p, load_report(p)) for p in args.compare]
        for path, report in reports:
            print_summary(report, label=Path(path).stem)
        print_comparison(reports)
        return

    paths = args.files if args.files else [latest_report()]

    for path in paths:
        report = load_report(path)
        print_summary(report, label=Path(path).stem)
        print_by_category(report)
        print_case_grid(report)

        if args.html:
            html_path = str(Path(path).with_suffix("")) + "_report.html"
            Path(html_path).write_text(render_html(report, path))
            print(f"\n  HTML report → {html_path}")


if __name__ == "__main__":
    main()