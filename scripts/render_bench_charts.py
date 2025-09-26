"""
Render benchmark charts from existing experiment JSON files without external libs.

Generates simple SVGs:
- equity_curve_quarters.svg: Equity curves per quarter (sampled)
- total_return_quarters.svg: Bar chart of total return per quarter
- winrate_quarters.svg: Bar chart of win rate per quarter

Inputs: domains/predict/experiments/backtest_AttentionTCN_*.json
Outputs: domains/predict/docs/assets/*.svg
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
EXP_DIR = BASE / "experiments"
ASSETS_DIR = BASE / "docs" / "assets"


@dataclass
class QuarterResult:
    name: str
    total_return: float
    win_rate: float
    equity_samples: list[float]


def load_quarter_results() -> list[QuarterResult]:
    mapping = {
        "Q1 2023": "backtest_AttentionTCN_20250822_153701.json",
        "Q2 2023": "backtest_AttentionTCN_20250822_153724.json",
        "Q3 2023": "backtest_AttentionTCN_20250822_153747.json",
        "Q4 2023": "backtest_AttentionTCN_20250822_153810.json",
    }
    results: list[QuarterResult] = []
    for qname, fname in mapping.items():
        fpath = EXP_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing experiment file: {fpath}")
        with open(fpath) as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        results.append(
            QuarterResult(
                name=qname,
                total_return=float(metrics.get("total_return", 0.0)),
                win_rate=float(metrics.get("win_rate", 0.0)),
                equity_samples=[float(x) for x in data.get("equity_curve_samples", [])],
            )
        )
    return results


def _scale_points(values: list[float], width: int, height: int, pad: int) -> list[tuple[float, float]]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    # Avoid div by zero
    rng = (vmax - vmin) or 1e-9
    n = len(values)
    pts: list[tuple[float, float]] = []
    for i, v in enumerate(values):
        x = pad + (i / max(1, n - 1)) * (width - 2 * pad)
        # invert y for SVG (0 at top)
        y = pad + (1 - (v - vmin) / rng) * (height - 2 * pad)
        pts.append((x, y))
    return pts


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'


def _svg_footer() -> str:
    return "</svg>\n"


def _axis_and_title(title: str, width: int, height: int, pad: int) -> str:
    # simple border + title
    s = []
    s.append(f'<rect x="{pad}" y="{pad}" width="{width-2*pad}" height="{height-2*pad}" fill="white" stroke="#ddd"/>')
    s.append(f'<text x="{width/2}" y="{pad-8}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#333">{title}</text>')
    return "\n".join(s) + "\n"


def render_equity_curves(quarters: list[QuarterResult], outpath: Path):
    width, height, pad = 800, 400, 40
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    svg = [_svg_header(width, height)]
    svg.append(_axis_and_title("Equity Curves by Quarter (sampled)", width, height, pad))
    # Legend
    legend_y = pad + 10
    for i, q in enumerate(quarters):
        svg.append(f'<rect x="{pad+10+i*140}" y="{legend_y}" width="12" height="12" fill="{colors[i%len(colors)]}"/>')
        svg.append(f'<text x="{pad+28+i*140}" y="{legend_y+11}" font-family="sans-serif" font-size="12" fill="#333">{q.name}</text>')
    # Plot lines (normalize within each quarter for relative shape)
    for i, q in enumerate(quarters):
        pts = _scale_points(q.equity_samples, width, height, pad)
        if not pts:
            continue
        d = "M " + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        svg.append(f'<path d="{d}" fill="none" stroke="{colors[i%len(colors)]}" stroke-width="2" />')
    svg.append(_svg_footer())
    outpath.write_text("".join(svg), encoding="utf-8")


def render_bar_chart(
    quarters: list[QuarterResult],
    get_value,
    title: str,
    outpath: Path,
    value_fmt: str = "{v:.2f}%",
):
    width, height, pad = 700, 420, 50
    bar_w = 90
    spacing = 40
    colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2"]
    svg = [_svg_header(width, height)]
    svg.append(_axis_and_title(title, width, height, pad))

    # Compute scaling
    values = [float(get_value(q)) for q in quarters]
    vmin, vmax = min(0.0, min(values)), max(values)
    rng = (vmax - vmin) or 1.0
    zero_y = pad + (1 - (0 - vmin) / rng) * (height - 2 * pad)

    # Draw zero line
    svg.append(f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{width-pad}" y2="{zero_y:.1f}" stroke="#ccc" stroke-dasharray="4 3" />')

    # Bars
    start_x = pad + 20
    for i, q in enumerate(quarters):
        v = float(get_value(q))
        x = start_x + i * (bar_w + spacing)
        top_y = pad + (1 - (max(v, 0) - vmin) / rng) * (height - 2 * pad)
        bottom_y = pad + (1 - (min(v, 0) - vmin) / rng) * (height - 2 * pad)
        bar_h = abs(bottom_y - top_y)
        y = min(top_y, bottom_y)
        svg.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{colors[i%len(colors)]}" rx="4" />')
        # Labels
        svg.append(
            f'<text x="{x + bar_w/2}" y="{y - 6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#333">{value_fmt.format(v=v)}</text>'
        )
        svg.append(f'<text x="{x + bar_w/2}" y="{height - pad + 18}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#555">{q.name}</text>')

    svg.append(_svg_footer())
    outpath.write_text("".join(svg), encoding="utf-8")


def main() -> None:
    quarters = load_quarter_results()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    render_equity_curves(quarters, ASSETS_DIR / "equity_curve_quarters.svg")
    render_bar_chart(
        quarters,
        get_value=lambda q: q.total_return,
        title="Total Return by Quarter (2023)",
        outpath=ASSETS_DIR / "total_return_quarters.svg",
    )
    render_bar_chart(
        quarters,
        get_value=lambda q: q.win_rate,
        title="Win Rate by Quarter (2023)",
        outpath=ASSETS_DIR / "winrate_quarters.svg",
        value_fmt="{v:.1f}%",
    )
    print(f"Charts written to: {ASSETS_DIR}")


if __name__ == "__main__":
    main()
