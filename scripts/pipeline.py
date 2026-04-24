#!/usr/bin/env python3
"""
End-to-end pipeline:
  1. Download Statcast data for a date
  2. Compute TDR on all consecutive pitch pairs to find the best tunneling pairs
  3. Download only the two videos per top pair
  4. Generate and save the overlay for each pair
"""

import sys
import time
import argparse
import warnings
import logging
from datetime import date
from pathlib import Path
from itertools import pairwise

import numpy as np
import pandas as pd
import requests
import pybaseball as pb
from bs4 import BeautifulSoup
from scipy import integrate

# Make ML overlay package importable
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML-auto-baseball-pitching-overlay"))

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from src.get_pitch_frames import get_pitch_frames
from src.generate_overlay import generate_overlay

warnings.simplefilter("ignore")
tf.get_logger().setLevel(logging.ERROR)

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_TYPE      = "HOME"
TUNNEL_POINT    = 0.125   # seconds — decision point from TDR paper
PLATE_TIME      = 0.330   # seconds — release to plate
MIN_TDR         = 0.3     # pairs below this are skipped
TOP_N_PAIRS     = 5       # max pairs to process per run
MIN_PITCHES     = 3
MAX_PITCHES     = 8
TERMINAL_EVENTS = ["strikeout", "home_run", "walk", "single", "double", "triple"]
PHYSICS_COLS    = ["vx0", "vy0", "vz0", "ax", "ay", "az",
                   "release_pos_x", "release_pos_y", "release_pos_z"]
MODEL_PATH      = REPO_ROOT / "ML-auto-baseball-pitching-overlay" / "model" / "yolov4-tiny-baseball-416"
OUT_BASE        = REPO_ROOT / "data" / "pipeline_output"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36"
})

# ── Physics (from compute_tunneling.py) ───────────────────────────────────────
def pos(t, p0, v0, a):
    return a * t**2 + v0 * t + p0

def ball_distance(t, p1, p2):
    dx = pos(t, p2["px0"], p2["vx0"], p2["ax"]) - pos(t, p1["px0"], p1["vx0"], p1["ax"])
    dy = pos(t, p2["py0"], p2["vy0"], p2["ay"]) - pos(t, p1["py0"], p1["vy0"], p1["ay"])
    dz = pos(t, p2["pz0"], p2["vz0"], p2["az"]) - pos(t, p1["pz0"], p1["vz0"], p1["az"])
    return np.sqrt(dx**2 + dy**2 + dz**2)

def row_to_physics(row):
    return {
        "px0": row["release_pos_x"], "py0": row["release_pos_y"], "pz0": row["release_pos_z"],
        "vx0": row["vx0"],           "vy0": row["vy0"],           "vz0": row["vz0"],
        "ax":  row["ax"],            "ay":  row["ay"],            "az":  row["az"],
    }

def compute_tdr(p1, p2):
    early, _ = integrate.quad(ball_distance, 0,            TUNNEL_POINT, args=(p1, p2))
    late,  _ = integrate.quad(ball_distance, TUNNEL_POINT, PLATE_TIME,   args=(p1, p2))
    tdr = (early / late) if late > 0 else np.nan
    return round(tdr, 4), round(early, 4), round(late, 4)

# ── MLB API (from get_tunnel_data.py) ─────────────────────────────────────────
def get_play_ids_for_game(game_pk):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    feed = requests.get(url, timeout=30).json()
    rows = []
    for play in feed["liveData"]["plays"]["allPlays"]:
        ab = play["about"]["atBatIndex"] + 1  # API is 0-based; statcast is 1-based
        for event in play["playEvents"]:
            if event.get("isPitch"):
                rows.append({
                    "game_pk":      game_pk,
                    "at_bat_number": ab,
                    "pitch_number": event["pitchNumber"],
                    "play_id":      event["playId"],
                })
    return pd.DataFrame(rows)

# ── Video download (from get_tunnel_data.py) ──────────────────────────────────
def resolve_video_url(play_id, video_type=VIDEO_TYPE):
    page = SESSION.get(
        f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}", timeout=30
    )
    page.raise_for_status()
    soup = BeautifulSoup(page.text, "html.parser")
    for source in soup.find_all("source"):
        src = source.get("src", "")
        if video_type.upper() in src.upper():
            return src
    first = soup.find("source", src=True)
    return first["src"] if first else None

def download_video(play_id, out_path, video_type=VIDEO_TYPE):
    url = resolve_video_url(play_id, video_type)
    if not url:
        return False
    r = SESSION.get(url, timeout=90, stream=True)
    r.raise_for_status()
    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    return True

# ── Pipeline steps ────────────────────────────────────────────────────────────
def step1_fetch_statcast(target_date: str) -> pd.DataFrame:
    print(f"\n[1] Fetching Statcast data for {target_date}...")
    df = pb.statcast(target_date, target_date)
    print(f"    {len(df)} pitches fetched")

    # Keep only at-bats that end in a terminal event
    terminal_keys = (
        df[df["events"].isin(TERMINAL_EVENTS)][["game_pk", "at_bat_number"]]
        .drop_duplicates()
    )
    df = df.merge(terminal_keys, on=["game_pk", "at_bat_number"])

    # Keep at-bats in the desired pitch count range
    ab_sizes = (
        df.groupby(["game_pk", "at_bat_number"])
        .size()
        .reset_index(name="pitch_count")
    )
    valid_abs = ab_sizes[ab_sizes["pitch_count"].between(MIN_PITCHES, MAX_PITCHES)][
        ["game_pk", "at_bat_number"]
    ]
    df = df.merge(valid_abs, on=["game_pk", "at_bat_number"]).dropna(subset=PHYSICS_COLS)

    # Attach play_ids from the MLB live-feed API
    game_pks = df["game_pk"].unique()
    lookup = pd.concat(
        [get_play_ids_for_game(pk) for pk in game_pks], ignore_index=True
    )
    df = df.merge(lookup, on=["game_pk", "at_bat_number", "pitch_number"], how="left")

    n_abs = df.groupby(["game_pk", "at_bat_number"]).ngroups
    print(f"    {len(df)} pitches across {n_abs} at-bats after filtering")
    return df


def step2_find_tunnel_pairs(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[2] Computing TDR for all consecutive pitch pairs...")
    pairs = []
    for (game_pk, ab), ab_df in df.groupby(["game_pk", "at_bat_number"]):
        ab_df = ab_df.sort_values("pitch_number")
        rows = list(ab_df.itertuples())
        for r1, r2 in pairwise(rows):
            p1 = row_to_physics(ab_df.loc[r1.Index])
            p2 = row_to_physics(ab_df.loc[r2.Index])
            tdr, early, late = compute_tdr(p1, p2)
            pairs.append({
                "game_pk":        game_pk,
                "at_bat_number":  ab,
                "pitcher":        r1.pitcher,
                "batter":         r1.batter,
                "pitch1_number":  r1.pitch_number,
                "pitch1_type":    r1.pitch_type,
                "pitch1_play_id": r1.play_id,
                "pitch2_number":  r2.pitch_number,
                "pitch2_type":    r2.pitch_type,
                "pitch2_play_id": r2.play_id,
                "early_sep":      early,
                "late_sep":       late,
                "TDR":            tdr,
            })

    pairs_df = pd.DataFrame(pairs).dropna(subset=["TDR"])
    top = pairs_df[pairs_df["TDR"] >= MIN_TDR].nlargest(TOP_N_PAIRS, "TDR")
    print(f"    {len(pairs_df)} pairs computed, {len(top)} above TDR={MIN_TDR}")
    if len(top):
        print(top[["pitch1_type", "pitch2_type", "early_sep", "late_sep", "TDR"]].to_string(index=False))
    return top


def step3_download_videos(pairs_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    print(f"\n[3] Downloading videos for {len(pairs_df)} pairs...")
    pair_dirs = []

    for _, row in pairs_df.iterrows():
        label = (
            f"{int(row['game_pk'])}_ab{int(row['at_bat_number'])}"
            f"_p{int(row['pitch1_number'])}vs{int(row['pitch2_number'])}"
        )
        pair_dir = out_dir / label
        pair_dir.mkdir(parents=True, exist_ok=True)

        for pitch_num, play_id, pitch_type in [
            (row["pitch1_number"], row["pitch1_play_id"], row["pitch1_type"]),
            (row["pitch2_number"], row["pitch2_play_id"], row["pitch2_type"]),
        ]:
            fname = f"pitch{int(pitch_num):02d}_{pitch_type}_{str(play_id)[:8]}.mp4"
            out_path = pair_dir / fname
            if out_path.exists():
                print(f"  [skip] {label}/{fname}")
            else:
                try:
                    ok = download_video(play_id, out_path)
                    kb = out_path.stat().st_size // 1024 if ok else 0
                    status = "OK" if ok else "miss"
                    print(f"  [{status}] {label}/{fname} ({kb} KB)")
                except Exception as e:
                    print(f"  [ERR] {label}/{fname}: {e}")
            time.sleep(1.2)

        pair_dirs.append(pair_dir)

    return pair_dirs


def step4_generate_overlays(pair_dirs: list[Path], out_dir: Path):
    print(f"\n[4] Loading YOLOv4 model from {MODEL_PATH}...")
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    saved_model = tf.saved_model.load(str(MODEL_PATH), tags=[tag_constants.SERVING])
    infer = saved_model.signatures["serving_default"]

    for pair_dir in pair_dirs:
        videos = sorted(v for v in pair_dir.iterdir() if v.suffix == ".mp4")
        if len(videos) < 2:
            print(f"\n  [skip] {pair_dir.name} — only {len(videos)} video(s) downloaded")
            continue

        print(f"\n  Processing {pair_dir.name}...")
        pitch_frames = []
        width = height = fps = None
        for video_path in videos:
            try:
                frames, w, h, f = get_pitch_frames(
                    str(video_path), infer, 416, 0.45, 0.5, sharpening=True
                )
                pitch_frames.append(frames)
                width, height, fps = w, h, f
                print(f"    {video_path.name}: {len(frames)} frames")
            except Exception as e:
                print(f"    [ERR] {video_path.name}: {e}")

        if len(pitch_frames) < 2:
            print(f"    [skip] not enough successful detections")
            continue

        output_path = out_dir / f"{pair_dir.name}_overlay.mp4"
        generate_overlay(pitch_frames, width, height, fps, str(output_path), registration_type="orb")
        print(f"    Saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pitch tunneling overlay pipeline")
    parser.add_argument(
        "--date",
        default=str(date.today()),
        help="Date to process in YYYY-MM-DD format (default: today)",
    )
    args = parser.parse_args()

    out_dir = OUT_BASE / args.date
    out_dir.mkdir(parents=True, exist_ok=True)

    statcast_df = step1_fetch_statcast(args.date)
    if statcast_df.empty:
        print("No data found for this date. Exiting.")
        return

    top_pairs = step2_find_tunnel_pairs(statcast_df)
    if top_pairs.empty:
        print(f"No pairs found above TDR={MIN_TDR}. Exiting.")
        return

    top_pairs.to_csv(out_dir / "top_pairs.csv", index=False)
    print(f"\n    Pairs saved → {out_dir / 'top_pairs.csv'}")

    pair_dirs = step3_download_videos(top_pairs, out_dir)
    step4_generate_overlays(pair_dirs, out_dir)

    print(f"\nDone. All output in {out_dir}")


if __name__ == "__main__":
    main()
