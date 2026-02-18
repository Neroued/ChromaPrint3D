#!/usr/bin/env python3
"""
Plot per-channel response across layers for each color in 1_single_stage.json.

Example:
  python -m modeling.eval.plot_stage_channels \
    --input modeling/output/params/1_single_stage.json \
    --value-key measured_srgb \
    --x-key step_layers \
    --output-dir modeling/output/reports/plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "modeling" / "1_single_stage.json"
DEFAULT_OUTPUT = REPO_ROOT / "modeling" / "plots" / "1_single_stage_channels"

VALUE_LABELS = {
    "measured_srgb": "sRGB (0-255)",
    "measured_linear_rgb": "Linear RGB (0-1)",
    "measured_lab_d65": "Lab D65",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot channel response vs layers for each color."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to 1_single_stage.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--value-key",
        choices=("measured_srgb", "measured_linear_rgb", "measured_lab_d65"),
        default="measured_srgb",
        help="Vector field to plot.",
    )
    parser.add_argument(
        "--x-key",
        choices=("step_layers", "thickness_mm"),
        default="step_layers",
        help="X axis key to use.",
    )
    parser.add_argument(
        "--substrate",
        type=str,
        default=None,
        help="Only include a specific substrate_id.",
    )
    parser.add_argument(
        "--colors",
        type=str,
        default=None,
        help="Comma-separated color_name list to include.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=8.0,
        help="Figure width (inches).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=5.0,
        help="Figure height (inches).",
    )
    parser.add_argument(
        "--legend-cols",
        type=int,
        default=2,
        help="Legend column count.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Disable legend.",
    )
    parser.add_argument(
        "--sort-colors",
        action="store_true",
        help="Sort series by color_name.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively.",
    )
    return parser.parse_args()


def normalize_key(value: str) -> str:
    return value.strip().lower()


def sanitize_filename(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch in (" ", "."):
            safe.append("_")
    return "".join(safe).strip("_") or "unknown"


def find_vector_length(datasets: Iterable[Dict[str, object]], value_key: str) -> int:
    for dataset in datasets:
        aggregated = dataset.get("aggregated") or []
        for entry in aggregated:
            vec = entry.get(value_key)
            if isinstance(vec, (list, tuple)) and len(vec) > 0:
                return len(vec)
    return 0


def infer_channel_labels(value_key: str, length: int) -> Tuple[str, ...]:
    if value_key in ("measured_srgb", "measured_linear_rgb") and length == 3:
        return ("R", "G", "B")
    if value_key == "measured_lab_d65" and length == 3:
        return ("L*", "a*", "b*")
    return tuple(f"C{i}" for i in range(length))


def collect_series(
    datasets: Iterable[Dict[str, object]],
    x_key: str,
    value_key: str,
    vector_len: int,
    substrate_filter: Optional[str],
    color_filter: Optional[Sequence[str]],
) -> Tuple[Dict[str, List[Tuple[str, List[float], List[List[float]]]]], int]:
    grouped: Dict[str, List[Tuple[str, List[float], List[List[float]]]]] = {}
    skipped = 0
    normalized_colors = (
        {normalize_key(c) for c in color_filter if c.strip()} if color_filter else None
    )
    normalized_substrate = normalize_key(substrate_filter) if substrate_filter else None

    for dataset in datasets:
        substrate_id = str(dataset.get("substrate_id", "unknown"))
        if normalized_substrate and normalize_key(substrate_id) != normalized_substrate:
            continue
        color_name = str(dataset.get("color_name", "unknown"))
        if normalized_colors and normalize_key(color_name) not in normalized_colors:
            continue

        aggregated = dataset.get("aggregated") or []
        points: List[Tuple[float, Sequence[float]]] = []
        for entry in aggregated:
            if x_key not in entry or value_key not in entry:
                continue
            vec = entry.get(value_key)
            if not isinstance(vec, (list, tuple)) or len(vec) != vector_len:
                continue
            points.append((float(entry[x_key]), vec))

        if not points:
            skipped += 1
            continue

        points.sort(key=lambda item: item[0])
        x_vals = [pt[0] for pt in points]
        y_by_channel: List[List[float]] = [[] for _ in range(vector_len)]
        for _, vec in points:
            for idx in range(vector_len):
                y_by_channel[idx].append(float(vec[idx]))

        grouped.setdefault(substrate_id, []).append((color_name, x_vals, y_by_channel))

    return grouped, skipped


def color_cycle(count: int) -> List[Tuple[float, float, float, float]]:
    if count <= 10:
        palette = plt.cm.tab10(np.linspace(0, 1, max(count, 1)))
    elif count <= 20:
        palette = plt.cm.tab20(np.linspace(0, 1, max(count, 1)))
    else:
        palette = plt.cm.nipy_spectral(np.linspace(0, 1, max(count, 1)))
    return [tuple(color) for color in palette]


class SimpleProgress:
    def __init__(self, total: int, desc: str = "Progress") -> None:
        self.total = max(int(total), 1)
        self.desc = desc
        self.count = 0
        self._last_len = 0
        self._render()

    def update(self, step: int = 1) -> None:
        self.count = min(self.total, self.count + step)
        self._render()

    def write(self, message: str) -> None:
        print(message)
        self._render()

    def close(self) -> None:
        self._render(final=True)
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self, final: bool = False) -> None:
        ratio = min(1.0, self.count / self.total)
        bar_len = 30
        filled = int(bar_len * ratio)
        bar = "=" * filled + "-" * (bar_len - filled)
        text = f"{self.desc}: [{bar}] {self.count}/{self.total}"
        if final:
            text += " done"
        pad = max(0, self._last_len - len(text))
        sys.stderr.write("\r" + text + (" " * pad))
        sys.stderr.flush()
        self._last_len = len(text)


def create_progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit="plot")
    return SimpleProgress(total=total, desc=desc)


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"Input JSON not found: {args.input}", file=sys.stderr)
        return 1

    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    datasets = payload.get("datasets") or []
    if not datasets:
        print("No datasets found in input JSON.", file=sys.stderr)
        return 1

    vector_len = find_vector_length(datasets, args.value_key)
    if vector_len == 0:
        print(f"No entries with {args.value_key} found in datasets.", file=sys.stderr)
        return 1

    channel_labels = infer_channel_labels(args.value_key, vector_len)
    color_filter = (
        [c.strip() for c in args.colors.split(",") if c.strip()]
        if args.colors
        else None
    )
    grouped, skipped = collect_series(
        datasets,
        args.x_key,
        args.value_key,
        vector_len,
        args.substrate,
        color_filter,
    )

    if not grouped:
        print("No matching series found after filtering.", file=sys.stderr)
        return 1

    if skipped:
        print(
            f"Skipped {skipped} dataset(s) without usable {args.value_key} entries.",
            file=sys.stderr,
        )

    x_label = (
        "Layer count (step_layers)" if args.x_key == "step_layers" else "Thickness (mm)"
    )
    value_label = VALUE_LABELS.get(args.value_key, args.value_key)
    y_limits = None
    if args.value_key == "measured_srgb":
        y_limits = (0, 255)
    elif args.value_key == "measured_linear_rgb":
        y_limits = (0, 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_plots = len(channel_labels) * len(grouped)
    progress = create_progress(total_plots, desc="Plotting")

    for substrate_id, series_list in grouped.items():
        if args.sort_colors:
            series_list = sorted(series_list, key=lambda item: item[0])

        substrate_dir = args.output_dir / sanitize_filename(substrate_id)
        substrate_dir.mkdir(parents=True, exist_ok=True)
        palette = color_cycle(len(series_list))

        for channel_idx, channel_label in enumerate(channel_labels):
            fig, ax = plt.subplots(figsize=(args.width, args.height))
            for idx, (color_name, x_vals, y_by_channel) in enumerate(series_list):
                y_vals = y_by_channel[channel_idx]
                ax.plot(
                    x_vals,
                    y_vals,
                    marker="o",
                    linewidth=1.6,
                    markersize=3,
                    label=color_name,
                    color=palette[idx % len(palette)],
                )

            ax.set_xlabel(x_label)
            ax.set_ylabel(f"{channel_label} ({value_label})")
            ax.set_title(f"{substrate_id} - {channel_label}")
            if y_limits:
                ax.set_ylim(*y_limits)
            ax.grid(True, linewidth=0.4, alpha=0.6)

            if not args.no_legend:
                ax.legend(
                    ncol=max(1, args.legend_cols),
                    fontsize=8,
                    frameon=False,
                    loc="best",
                )

            fig.tight_layout()
            file_name = (
                f"{sanitize_filename(args.value_key)}__"
                f"{sanitize_filename(args.x_key)}__"
                f"{sanitize_filename(channel_label)}.png"
            )
            output_path = substrate_dir / file_name
            fig.savefig(output_path, dpi=args.dpi)
            if args.show:
                plt.show()
            plt.close(fig)

            progress.write(f"Saved: {output_path}")
            progress.update(1)

    progress.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
