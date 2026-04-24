import pandas as pd
import numpy as np 
from scipy import integrate 
from pathlib import Path
from itertools import pairwise

METADATA_PATH = Path("../data/atbat_sequences/metadata.csv")
OUT_PATH = Path("../data/atbat_sequences/tunneling.csv")

TUNNEL_POINT = 0.125 # This was the value in the TDR article
PLATE_TIME = 0.330

df = pd.read_csv(METADATA_PATH)

PARAMS = ["vx0", "vy0", "vz0", "ax", "ay", "az", "release_pos_x", "release_pos_y", "release_pos_z"]

def pos(t, p0, v0, a):
	return a*t**2 + v0*t + p0

def d(t, p1: dict, p2: dict) -> float:
	dx = pos(t, p2["px0"], p2["vx0"], p2["ax"]) - pos(t, p1["px0"], p1["vx0"], p1["ax"])
	dy = pos(t, p2["py0"], p2["vy0"], p2["ay"]) - pos(t, p1["py0"], p1["vy0"], p1["ay"])
	dz = pos(t, p2["pz0"], p2["vz0"], p2["az"]) - pos(t, p1["pz0"], p1["vz0"], p1["az"])
	return np.sqrt(dx**2 + dy**2 + dz**2)

def row_to_dict(row) -> dict:
	return {
		"px0": row["release_pos_x"],
		"py0": row["release_pos_y"],
		"pz0": row["release_pos_z"],
		"vx0": row["vx0"],
		"vy0": row["vy0"],
		"vz0": row["vz0"],
		"ax" : row["ax"],
		"ay" : row["ay"],
		"az" : row["az"],
	}


def integrate_d(p1: dict, p2: dict, ti: float, tf: float) -> float:
	res, _ = integrate.quad(d, ti, tf, args=(p1, p2))
	return res 

df = df.dropna(subset=PARAMS)

print(f"Loaded {len(df)} pitches across "
	  f"{df.groupby(['game_pk', 'at_bat_number']).ngroups} at-bats")


tunneling_info = []

for (game_pk, at_bat_number), ab_df in df.groupby(["game_pk", "at_bat_number"]):
	ab_df = ab_df.sort_values("pitch_number")
	pitches = list(ab_df.itertuples())

	for p1_row, p2_row in pairwise(pitches):
		p1 = row_to_dict(ab_df.loc[p1_row.Index])
		p2 = row_to_dict(ab_df.loc[p2_row.Index])

		i1 = integrate_d(p1, p2, 0, TUNNEL_POINT)
		i2 = integrate_d(p1, p2, TUNNEL_POINT, PLATE_TIME)

		tdr = i1 / i2 if i2 > 0 else np.nan

		tunneling_info.append({
			"game_pk": game_pk,
			"at_bat_number": at_bat_number,
			"pitcher": p1_row.pitcher,
			"batter": p1_row.batter,
			"pitch1_number": p1_row.pitch_number,
			"pitch1_type": p1_row.pitch_type,
			"pitch1_play_id": p1_row.play_id,
			"pitch2_number": p2_row.pitch_number,
			"pitch2_type": p2_row.pitch_type,
			"pitch2_play_id": p2_row.play_id,
			"early_separation": round(i1, 4),
			"late_separation": round(i2, 4),
			"TDR": round(tdr, 4),
		})

res_df = pd.DataFrame(tunneling_info)

print(f"\nComputed TDR for {len(res_df)} consecutive pitch pairs\n")
print(res_df[["pitch1_type", "pitch2_type", "early_separation",
               "late_separation", "TDR"]].to_string(index=False))
 
print(f"\nTop tunneling pairs (highest TDR):")
print(res_df.nlargest(5, "TDR")[
    ["pitcher", "pitch1_type", "pitch2_type", "early_separation", "late_separation", "TDR"]
].to_string(index=False))

 
res_df.to_csv(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")
 
