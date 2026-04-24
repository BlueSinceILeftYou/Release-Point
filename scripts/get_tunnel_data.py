import numpy as np
import pandas as pd
import pybaseball as pb
import time
import os
import random
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# Keep settings from previous get data here
VIDEO_TYPE = "HOME"                 
START_DATE = "2025-09-01"             
END_DATE = "2025-09-28"
NUM_AT_BATS = 12

# At bat info
MIN_PITCHES = 3
MAX_PITCHES = 8
TERMINAL_EVENTS    = [              
    "strikeout", "home_run", "walk",
    "single", "double", "triple",
]

# OS
OUT_DIR = Path("../data/atbat_sequences")
OUT_DIR.mkdir(parents=True, exist_ok=True)


#PARAMS = ["vx0", "vy0", "vz0", "ax", "ay", "az", "release_pos_x", "relase_extension", "release_pos_z"]

"""
These are just Ethan's functions from get data. I know just copying 
and pasting isn't the most elegant but just want to test for now
"""
def get_lookup_for_game(game_pk):
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    feed = requests.get(url).json()

    rows = []
    for play in feed["liveData"]["plays"]["allPlays"]:
        ab = play["about"]["atBatIndex"] + 1            # API is 0-based; statcast at_bat_number is 1-based
        for event in play["playEvents"]:
            if event.get("isPitch"):
                rows.append({
                    "game_pk": game_pk,
                    "at_bat_number": ab,
                    "pitch_number": event["pitchNumber"],
                    "play_id": event["playId"],
                })
    return pd.DataFrame(rows)

def resolve_video_url(play_id: str, video_type: str = "HOME") -> str | None:
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


def download_video(play_id: str, out_path: Path, video_type: str = "HOME") -> bool:
    video_url = resolve_video_url(play_id, video_type)
    if not video_url:
        return False
    r = SESSION.get(video_url, timeout=90, stream=True)
    r.raise_for_status()
    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    return True


SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36"
})

print(f"Output directory: {OUT_DIR.resolve()}")

df = pb.statcast(START_DATE, END_DATE)

terminal_ab_keys = (
    df[df["events"].isin(TERMINAL_EVENTS)][["game_pk", "at_bat_number"]]
    .drop_duplicates()
)
df = df.merge(terminal_ab_keys, on=["game_pk", "at_bat_number"])
 
# Count pitches per at-bat and filter to the desired length range
ab_sizes = (
    df.groupby(["game_pk", "at_bat_number", "pitcher", "batter"])
    .size()
    .reset_index(name="pitch_count")
)
ab_sizes = ab_sizes[ab_sizes["pitch_count"].between(MIN_PITCHES, MAX_PITCHES)]

# Sample some at bats
sampled_ab_keys = ab_sizes.sample(n=NUM_AT_BATS, random_state=42)[["game_pk", "at_bat_number"]]

sampled_pitches = (
	df.merge(sampled_ab_keys, on=["game_pk", "at_bat_number"])
	.sort_values(["game_pk", "at_bat_number", "pitch_number"])
	.copy()
)

print(f"Sampled {sampled_pitches['at_bat_number'].nunique()} at-bats "
	  f"({len(sampled_pitches)} total pitches)")


game_pks = sampled_pitches["game_pk"].unique()
lookup_df = pd.concat([get_lookup_for_game(pk) for pk in game_pks], ignore_index=True)

sampled_pitches = sampled_pitches.merge(
	lookup_df,
	on=["game_pk", "at_bat_number", "pitch_number"],
	how="left",
)

missing = sampled_pitches["play_id"].isna().sum()
print(f"Merged : {len(sampled_pitches)} rows | Missing play_id : {missing}")

downloaded, failed = [], []

for _, row in sampled_pitches.iterrows():
	if pd.isna(row["play_id"]):
		failed.append(row.get("play_id"))
		continue

	# give each at-bat its own subfolder
	ab_dir = OUT_DIR / f"{row['game_pk']}_ab{int(row['at_bat_number'])}_{row['pitcher']}_vs_{row['batter']}"
	ab_dir.mkdir(parents=True,exist_ok=True)

	safe_type = row["pitch_type"] if row["pitch_type"] else "XX"
	fname = f"pitch{int(row['pitch_number']):02d}_{safe_type}_{row['play_id'][:8]}.mp4"
	out_path = ab_dir / fname

	if out_path.exists():
	        print(f"  [skip] {ab_dir.name}/{fname}")
	        downloaded.append(row["play_id"])
	        continue
	try:
		ok = download_video(row["play_id"], out_path, VIDEO_TYPE)
		if ok:
			kb = out_path.stat().st_size // 1024
			print(f" [OK] {ab_dir.name}/fname	({kb} KB)")
			downloaded.append(row["play_id"])
		else:
			print(f" [miss] {row['play_id']} - video URL not found")
			failed.append(row["play_id"])
	except Exception as e:
		print(f" [ERR] {row['play_id']}: {e}")
		failed.append(row["play_id"])

	time.sleep(1.2)

print(f"\nDone.  Downloaded: {len(downloaded)}  |  Failed / missing: {len(failed)}")

sampled_pitches["downloaded"] = sampled_pitches["play_id"].isin(downloaded)
sampled_pitches["ab_folder"]  = sampled_pitches.apply(
    lambda r: f"{r['game_pk']}_ab{int(r['at_bat_number'])}_{r['pitcher']}_vs_{r['batter']}",
    axis=1,
)
sampled_pitches["filename"] = sampled_pitches.apply(
    lambda r: f"pitch{int(r['pitch_number']):02d}_{r['pitch_type'] or 'XX'}_{r['play_id'][:8]}.mp4"
    if pd.notna(r["play_id"]) else "",
    axis=1,
)
 
csv_path = OUT_DIR / "metadata.csv"
sampled_pitches.to_csv(csv_path, index=False)
print(f"Metadata saved → {csv_path}")