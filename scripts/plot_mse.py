"""
compare_registration_mse.py

Scans pipeline_output date folders for matching ORB / cross-correlation MSE CSVs,
compares their per-frame MSE values, and saves summary plots + a combined CSV.

Usage:
    python compare_registration_mse.py --root "C:/Users/user/Desktop/Release-Point/data/pipeline_output"

Optional:
    --out_dir   where to save plots/summary  (default: alongside this script)
    --date      restrict to a single date folder, e.g. 2025-08-15
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── helpers ──────────────────────────────────────────────────────────────────

def find_csv_pairs(root: Path, date_filter: str | None = None):
    """
    Walk root/YYYY-MM-DD/ and find matching orb + cross CSVs.
    Returns a list of (game_id, orb_csv_path, cross_csv_path).
    """
    pairs = []
    date_dirs = sorted(root.iterdir()) if root.is_dir() else []

    for date_dir in date_dirs:
        if not date_dir.is_dir():
            continue
        if date_filter and date_dir.name != date_filter:
            continue
        # skip folders that don't look like dates
        if not re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", date_dir.name):
            continue

        csvs = list(date_dir.glob("*.csv"))
        # group by game_id (strip _orb_overlay_mse / _cross_overlay_mse suffix)
        orb_map   = {}
        cross_map = {}
        for csv in csvs:
            m = re.match(r"^(.+?)_(orb|cross)_overlay_mse\.csv$", csv.name)
            if not m:
                continue
            game_id, reg_type = m.group(1), m.group(2)
            if reg_type == "orb":
                orb_map[game_id] = csv
            else:
                cross_map[game_id] = csv

        for game_id in orb_map.keys() & cross_map.keys():
            pairs.append((date_dir.name, game_id, orb_map[game_id], cross_map[game_id]))

    return pairs


def load_and_merge(orb_path: Path, cross_path: Path) -> pd.DataFrame:
    orb   = pd.read_csv(orb_path)
    cross = pd.read_csv(cross_path)

    orb["method"]   = "orb"
    cross["method"] = "cross"

    merged = pd.merge(
        orb[["frame_idx", "video_pair", "mse"]].rename(columns={"mse": "orb_mse"}),
        cross[["frame_idx", "video_pair", "mse"]].rename(columns={"mse": "cross_mse"}),
        on=["frame_idx", "video_pair"],
        how="inner",
    )
    merged["delta"] = merged["orb_mse"] - merged["cross_mse"]  # negative = ORB better
    return merged


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = {"orb": "#E05C2A", "cross": "#2A7BE0"}

def plot_mse_over_frames(df: pd.DataFrame, game_id: str, date: str, out_dir: Path):
    pairs = sorted(df["video_pair"].unique())
    n = len(pairs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(f"{date}  ·  {game_id}\nMSE per frame: ORB vs Cross-Correlation",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, pair in zip(axes, pairs):
        sub = df[df["video_pair"] == pair].sort_values("frame_idx")
        ax.plot(sub["frame_idx"], sub["orb_mse"],   label="ORB",   color=COLORS["orb"],   linewidth=1.4)
        ax.plot(sub["frame_idx"], sub["cross_mse"], label="Cross", color=COLORS["cross"], linewidth=1.4)
        ax.set_title(f"Video pair {pair}", fontsize=10)
        ax.set_ylabel("MSE")
        ax.set_xlabel("Frame")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"{date}_{game_id}_mse_per_frame.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_summary_bar(summary_rows: list[dict], out_dir: Path):
    """Bar chart of mean MSE per game across all dates."""
    df = pd.DataFrame(summary_rows)
    df["label"] = df["date"] + "\n" + df["game_id"]

    x = range(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.4), 5))
    ax.bar([i - width / 2 for i in x], df["orb_mean_mse"],   width, label="ORB",   color=COLORS["orb"],   alpha=0.85)
    ax.bar([i + width / 2 for i in x], df["cross_mean_mse"], width, label="Cross", color=COLORS["cross"], alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["label"], fontsize=8)
    ax.set_ylabel("Mean MSE")
    ax.set_title("Mean MSE — ORB vs Cross-Correlation (all games)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = out_dir / "summary_mean_mse.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",    required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--date",    default=None,  help="Restrict to one date folder e.g. 2025-08-15")
    args = parser.parse_args()

    root    = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "mse_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_csv_pairs(root, args.date)
    if not pairs:
        print("No matching ORB + cross CSV pairs found.")
        return

    print(f"Found {len(pairs)} matched game(s).\n")

    all_merged   = []
    summary_rows = []

    for date, game_id, orb_path, cross_path in pairs:
        print(f"[{date}]  {game_id}")
        df = load_and_merge(orb_path, cross_path)
        df["date"]    = date
        df["game_id"] = game_id
        all_merged.append(df)

        plot_mse_over_frames(df, game_id, date, out_dir)

        summary_rows.append({
            "date":            date,
            "game_id":         game_id,
            "orb_mean_mse":    df["orb_mse"].mean(),
            "cross_mean_mse":  df["cross_mse"].mean(),
            "orb_better_pct":  (df["delta"] < 0).mean() * 100,
            "n_frames":        len(df),
        })

    # Combined CSV
    combined = pd.concat(all_merged, ignore_index=True)
    combined_path = out_dir / "all_games_mse_comparison.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined CSV saved to {combined_path}")

    # Summary bar chart
    if len(summary_rows) > 1:
        plot_summary_bar(summary_rows, out_dir)

    # Print summary table
    summary_df = pd.DataFrame(summary_rows)
    print("\n── Summary ──────────────────────────────────────")
    print(summary_df.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()