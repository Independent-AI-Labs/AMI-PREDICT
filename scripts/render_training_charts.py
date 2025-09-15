"""
Render training and verification charts from experiments without external libs.

Reads:
- experiments/all_results.json (list of experiments with summary + training_history)
- experiments/*/experiment.json (if needed)

Outputs SVG to docs/assets/:
- learning_curves_best.svg        (train/val accuracy and loss over epochs for best model)
- val_accuracy_ranking.svg        (bar chart of best val acc across experiments)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any


BASE = Path(__file__).resolve().parents[1]
EXP_FILE = BASE / "experiments" / "all_results.json"
ASSETS_DIR = BASE / "docs" / "assets"


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'


def _svg_footer() -> str:
    return "</svg>\n"


def _axis_and_title(title: str, width: int, height: int, pad: int) -> str:
    s = []
    s.append(f'<rect x="{pad}" y="{pad}" width="{width-2*pad}" height="{height-2*pad}" fill="white" stroke="#ddd"/>')
    s.append(f'<text x="{width/2}" y="{pad-8}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#333">{title}</text>')
    return "\n".join(s) + "\n"


def _scale_series(xs: List[float], ys: List[float], width: int, height: int, pad: int, yrange: Tuple[float, float] | None = None) -> List[Tuple[float, float]]:
    if not xs or not ys:
        return []
    xmin, xmax = 0, max(xs) if xs else 1
    ymin, ymax = (min(ys), max(ys)) if yrange is None else yrange
    yrng = (ymax - ymin) or 1e-9
    xrng = (xmax - xmin) or 1e-9
    pts = []
    for x, y in zip(xs, ys):
        sx = pad + (x - xmin) / xrng * (width - 2 * pad)
        sy = pad + (1 - (y - ymin) / yrng) * (height - 2 * pad)
        pts.append((sx, sy))
    return pts


def _polyline(points: List[Tuple[float, float]], color: str, width: float = 2.0) -> str:
    if not points:
        return ""
    d = "M " + " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" />\n'


def _legend(items: List[Tuple[str, str]], x: int, y: int) -> str:
    s = []
    for i, (label, color) in enumerate(items):
        yy = y + i * 18
        s.append(f'<rect x="{x}" y="{yy}" width="12" height="12" fill="{color}"/>')
        s.append(f'<text x="{x+18}" y="{yy+11}" font-family="sans-serif" font-size="12" fill="#333">{label}</text>')
    return "\n".join(s) + "\n"


def _bar_chart(width: int, height: int, pad: int, labels: List[str], values: List[float], title: str, value_fmt: str = "{v:.2f}%") -> str:
    svg = [_svg_header(width, height)]
    svg.append(_axis_and_title(title, width, height, pad))
    vmin, vmax = min(0.0, min(values) if values else 0.0), max(values) if values else 1.0
    rng = (vmax - vmin) or 1.0
    zero_y = pad + (1 - (0 - vmin) / rng) * (height - 2 * pad)
    svg.append(f'<line x1="{pad}" y1="{zero_y:.1f}" x2="{width-pad}" y2="{zero_y:.1f}" stroke="#ccc" stroke-dasharray="4 3" />')
    # layout
    n = len(values)
    bar_w = max(30, int((width - 2*pad) / max(1, 2*n)))
    spacing = min(40, max(20, int((width - 2*pad - n*bar_w) / max(1, n-1))))
    x = pad + ( (width - 2*pad) - (n*bar_w + (n-1)*spacing) )/2
    colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f", "#edc948"]
    for i, (lab, v) in enumerate(zip(labels, values)):
        bx = x + i * (bar_w + spacing)
        top_y = pad + (1 - (max(v, 0) - vmin) / rng) * (height - 2 * pad)
        bottom_y = pad + (1 - (min(v, 0) - vmin) / rng) * (height - 2 * pad)
        bar_h = abs(bottom_y - top_y)
        y = min(top_y, bottom_y)
        color = colors[i % len(colors)]
        svg.append(f'<rect x="{bx:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{color}" rx="4" />')
        svg.append(f'<text x="{bx + bar_w/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#333">{value_fmt.format(v=v)}</text>')
        svg.append(f'<text x="{bx + bar_w/2:.1f}" y="{height - pad + 16}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">{lab}</text>')
    svg.append(_svg_footer())
    return "".join(svg)


def render_learning_curves_best(all_results: List[Dict[str, Any]], outpath: Path) -> None:
    successes = [e for e in all_results if e.get("status") == "success" and e.get("training_history")]
    if not successes:
        raise RuntimeError("No successful experiments with training_history found")
    # best by best_val_acc
    best = max(successes, key=lambda e: e.get("summary", {}).get("best_val_acc", 0))
    hist = best["training_history"]
    epochs = [h["epoch"] for h in hist]
    tr_acc = [float(h["train_acc"]) for h in hist]
    va_acc = [float(h["val_acc"]) for h in hist]
    tr_loss = [float(h["train_loss"]) for h in hist]
    va_loss = [float(h["val_loss"]) for h in hist]

    width, height, pad = 860, 420, 50
    svg = [_svg_header(width, height)]
    svg.append(_axis_and_title(f"Learning Curves — {best.get('name','')} ({best.get('model_type','')})", width, height, pad))

    # Two panels
    mid = width // 2
    # Accuracy panel
    acc_x0, acc_y0, acc_w, acc_h = pad, pad+10, mid - pad*1.2, height - pad*2 - 20
    # Draw panel border
    svg.append(f'<rect x="{acc_x0}" y="{acc_y0}" width="{acc_w}" height="{acc_h}" fill="white" stroke="#eee"/>')
    pts_tr = _scale_series(epochs, tr_acc, int(acc_w), int(acc_h), 10, yrange=(min(tr_acc+va_acc), max(tr_acc+va_acc)))
    pts_va = _scale_series(epochs, va_acc, int(acc_w), int(acc_h), 10, yrange=(min(tr_acc+va_acc), max(tr_acc+va_acc)))
    # offset into panel coords
    pts_tr = [(x+acc_x0, y+acc_y0) for x, y in pts_tr]
    pts_va = [(x+acc_x0, y+acc_y0) for x, y in pts_va]
    svg.append(_polyline(pts_tr, "#4e79a7"))
    svg.append(_polyline(pts_va, "#e15759"))
    svg.append(_legend([("Train Acc", "#4e79a7"), ("Val Acc", "#e15759")], int(acc_x0+10), int(acc_y0+10)))

    # Loss panel
    loss_x0, loss_y0, loss_w, loss_h = mid, pad+10, width - mid - pad, height - pad*2 - 20
    svg.append(f'<rect x="{loss_x0}" y="{loss_y0}" width="{loss_w}" height="{loss_h}" fill="white" stroke="#eee"/>')
    pts_tl = _scale_series(epochs, tr_loss, int(loss_w), int(loss_h), 10, yrange=(min(tr_loss+va_loss), max(tr_loss+va_loss)))
    pts_vl = _scale_series(epochs, va_loss, int(loss_w), int(loss_h), 10, yrange=(min(tr_loss+va_loss), max(tr_loss+va_loss)))
    pts_tl = [(x+loss_x0, y+loss_y0) for x, y in pts_tl]
    pts_vl = [(x+loss_x0, y+loss_y0) for x, y in pts_vl]
    svg.append(_polyline(pts_tl, "#59a14f"))
    svg.append(_polyline(pts_vl, "#f28e2c"))
    svg.append(_legend([("Train Loss", "#59a14f"), ("Val Loss", "#f28e2c")], int(loss_x0+10), int(loss_y0+10)))

    svg.append(_svg_footer())
    outpath.write_text("".join(svg), encoding="utf-8")


def render_val_accuracy_ranking(all_results: List[Dict[str, Any]], outpath: Path) -> None:
    successes = [e for e in all_results if e.get("status") == "success" and e.get("summary")]
    if not successes:
        raise RuntimeError("No successful experiments with summaries found")
    # sort by best val acc desc and take top 6
    successes.sort(key=lambda e: e.get("summary", {}).get("best_val_acc", 0), reverse=True)
    top = successes[:6]
    labels = [e.get("name", e.get("model_type","")) for e in top]
    values = [float(e["summary"].get("best_val_acc", 0.0)) for e in top]
    svg = _bar_chart(780, 420, 50, labels, values, "Best Validation Accuracy (Top Experiments)")
    outpath.write_text(svg, encoding="utf-8")


def main() -> None:
    if not EXP_FILE.exists():
        raise FileNotFoundError(f"Missing experiments file: {EXP_FILE}")
    with open(EXP_FILE, "r") as f:
        all_results = json.load(f)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    render_learning_curves_best(all_results, ASSETS_DIR / "learning_curves_best.svg")
    render_val_accuracy_ranking(all_results, ASSETS_DIR / "val_accuracy_ranking.svg")
    print(f"Wrote training charts to: {ASSETS_DIR}")


if __name__ == "__main__":
    main()

