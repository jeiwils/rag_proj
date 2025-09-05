import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple
import json
import csv
try:  # pragma: no cover - pillow may be absent
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # noqa: BLE001
    Image = ImageDraw = ImageFont = None


def ensure_output_path(path: Path) -> Path:
    """Create parent directories for ``path`` and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

logger = logging.getLogger(__name__)

# Manual mapping of teacher models to their distilled counterparts.
# Example pair provided in the task description included for completeness.
TEACHER_DISTILL_PAIRS: List[Tuple[str, str]] = [
    ("qwen2.5-2x7b-moe-power-coder-v4", "deepseek-r1-distill-qwen-14b"),
    ("qwen2.5-14b-instruct", "deepseek-r1-distill-qwen-14b"),
    ("qwen2.5-7b-instruct", "deepseek-r1-distill-qwen-7b"),
]

SPLIT = "dev"


def _available_datasets(model: str) -> List[str]:
    base = Path(f"data/results/{model}")
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir())


def _summary_path(model: str, dataset: str, seed: int) -> Path:
    return Path(
        f"data/results/{model}/{dataset}/{SPLIT}/dense_seed{seed}/"
        f"summary_metrics_dense_seed{seed}_{SPLIT}.json"
    )


def _mean_f1(model: str, dataset: str) -> float | None:
    f1s: List[float] = []
    base = Path(f"data/results/{model}/{dataset}/{SPLIT}")
    if not base.exists():
        return None
    for seed_dir in base.glob("dense_seed*"):
        m = re.search(r"dense_seed(\d+)", seed_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        sp = _summary_path(model, dataset, seed)
        if not sp.exists():
            logger.warning("Missing summary metrics: %s", sp)
            continue
        with sp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        dense_eval = data.get("dense_eval", data)
        f1 = dense_eval.get("F1") or dense_eval.get("f1")
        if f1 is not None:
            try:
                f1s.append(float(f1))
            except (TypeError, ValueError):
                logger.debug("Non-numeric F1 in %s", sp)
    if not f1s:
        return None
    return float(sum(f1s) / len(f1s))


def collect_parity() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for teacher, distilled in TEACHER_DISTILL_PAIRS:
        teacher_datasets = _available_datasets(teacher)
        distill_datasets = _available_datasets(distilled)
        datasets = sorted(set(teacher_datasets) & set(distill_datasets))
        for dataset in datasets:
            t_f1 = _mean_f1(teacher, dataset)
            d_f1 = _mean_f1(distilled, dataset)
            if t_f1 is None or d_f1 is None:
                continue
            rows.append(
                {
                    "pair": f"{teacher} ↔ {distilled}",
                    "dataset": dataset,
                    "teacher_f1": t_f1,
                    "distilled_f1": d_f1,
                    "delta": d_f1 - t_f1,
                }
            )
    return rows


def plot_parity(rows: List[Dict[str, object]], out_dir: Path) -> None:
    if not rows:
        logger.warning("No data available for plotting")
        return
    if Image is None:
        logger.error("Pillow is required for plotting")
        return
    width, height = 600, 600
    margin = 50
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    all_teacher = [r["teacher_f1"] for r in rows]
    all_distill = [r["distilled_f1"] for r in rows]
    min_f = min(min(all_teacher), min(all_distill))
    max_f = max(max(all_teacher), max(all_distill))
    span = max_f - min_f if max_f > min_f else 1.0

    # Axes
    draw.line((margin, height - margin, width - margin, height - margin), fill="black")
    draw.line((margin, height - margin, margin, margin), fill="black")

    # Diagonal reference y=x
    draw.line(
        (
            margin,
            height - margin,
            width - margin,
            margin,
        ),
        fill="grey",
        width=1,
    )

    colors = ["red", "blue", "green", "purple", "orange"]
    by_pair: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r)

    for idx, (pair, vals) in enumerate(by_pair.items()):
        color = colors[idx % len(colors)]
        for v in vals:
            x = margin + (v["teacher_f1"] - min_f) / span * (width - 2 * margin)
            y = height - margin - (v["distilled_f1"] - min_f) / span * (height - 2 * margin)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)

    ensure_output_path(out_dir / "distillation_parity_scatter.png")
    img.save(out_dir / "distillation_parity_scatter.png")


def save_summary(rows: List[Dict[str, object]], out_dir: Path) -> None:
    if not rows:
        return
    by_pair: Dict[str, List[float]] = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r["delta"])
    summary_rows: List[List[object]] = []
    for pair, deltas in by_pair.items():
        mean_delta = sum(deltas) / len(deltas)
        parity_pct = (
            sum(1 for d in deltas if abs(d) <= 1.0) / len(deltas) * 100.0
        )
        summary_rows.append([pair, mean_delta, parity_pct])

    # Save CSV
    csv_path = ensure_output_path(out_dir / "delta_summary.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair", "mean_delta", "parity_pct"])
        for row in summary_rows:
            writer.writerow(row)

    # Render table as simple image
    header = ["Pair", "Mean ΔF1", "% ≤1.0 F1"]
    rows_to_draw = [header] + [
        [pair, f"{mean:.3f}", f"{pct:.1f}"] for pair, mean, pct in summary_rows
    ]
    width, row_h = 800, 40
    if Image is None:
        logger.error("Pillow is required for plotting")
        return
    img = Image.new("RGB", (width, row_h * len(rows_to_draw)), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for idx, row in enumerate(rows_to_draw):
        y = idx * row_h + 10
        draw.text((10, y), row[0], fill="black", font=font)
        draw.text((300, y), row[1], fill="black", font=font)
        draw.text((500, y), row[2], fill="black", font=font)
    ensure_output_path(out_dir / "delta_summary.png")
    img.save(out_dir / "delta_summary.png")


def main() -> None:
    out_dir = Path("graphs/distillation_parity")
    rows = collect_parity()
    if not rows:
        logger.warning("No overlapping datasets for teacher/distilled pairs")
        return
    csv_path = ensure_output_path(out_dir / "distillation_parity.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pair", "dataset", "teacher_f1", "distilled_f1", "delta"]
        )
        writer.writeheader()
        writer.writerows(rows)
    plot_parity(rows, out_dir)
    save_summary(rows, out_dir)


if __name__ == "__main__":  # pragma: no cover
    main()