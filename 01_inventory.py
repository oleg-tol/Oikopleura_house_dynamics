#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01 — Inventory: every recording, deduplicated, with annotations, groups and tracking QC.

Outputs (OUT/):
    01_metadata_animals.csv      one row per animal
    01_inventory_all_files.csv   every csv copy found
    01_annotation_events.csv     abnormal start / failed insertion / twitching (native frames)
    01_inventory_overview.png/svg
"""
import re
import hashlib
from common import *

# =============================================================================
# CONFIG
# =============================================================================
SOURCES = {                         # name: (root folder, priority — lower wins)
    "NEW_HE":     (f"{HD}/New_HE_data/OneDrive_1_10-8-2026", 1),
    "B1_28_28":   (f"{HD}/28_28_partial_overlap/OneDrive_1_10-8-2026(1)", 2),
    "OHDA_DOP":   (f"{HD}/OHDA_DOP", 3),
    "OLD_DATA":   (f"{HD}/Old_data", 4),
    "HE_CSV_OLD": (f"{HD}/csv files for house expansion dopamine_ctrl and 6-OHDA", 5),
}
NEW_TXT = f"{SOURCES['NEW_HE'][0]}/detailed info on house expansion frames and abnormal behaviour.txt"
OLD_TXT = f"{HD}/Old_data/detailed info on house expansion frames.txt"

COND_FOLDERS = {"control": "control", "6-ohda": "6-OHDA", "6-ohda+dopamine": "6-OHDA+DA"}
HOUSED_AT_START = {                 # collaborator: inside a house at video start
    "20250527_111618", "20250527_111622", "20250602_141107", "20250602_144303", "20250623_102937",
    "20250623_102940", "20250623_104657", "20250623_104658", "20250623_110702", "20250623_152145",   # control
    "20250527_150040", "20250602_153854", "20250602_155811"}                                         # 6-OHDA
EXCLUDE = {"20250818_135423": "condition conflict (6-OHDA in HE folder, 6-OHDA+DA in OHDA_DOP) + poor tracking"}
NOTES = {"B2026_6-OHDA": "re-recorded until house building started: onset latency not interpretable"}
BP_KEY = ["bodypart1", "bodypart2", "bodypart3", "bodypart5", "bodypart12", "bodypart6"]
RX_ID = re.compile(r"(\d{8}_\d{6})")

def batch_of(e):
    """B1 = May–Jun 2025, B2 = Aug–Sep 2025, B2026 = new recordings."""
    return "B2026" if e.startswith("2026") else ("B1" if e[4:6] in ("05", "06") else "B2")

# =============================================================================
# A. find every csv
# =============================================================================
files = []
for src, (root, prio) in SOURCES.items():
    for dp, dns, fns in os.walk(root):
        if dp == root:                                   # only descend into condition folders
            dns[:] = [d for d in dns if d.lower() in COND_FOLDERS]
        parts = os.path.relpath(dp, root).split(os.sep)
        cond = COND_FOLDERS.get(parts[0].lower()) if parts[0] != "." else None
        for fn in fns:
            m = RX_ID.match(fn)
            if cond and fn.lower().endswith(".csv") and m:
                snap = re.search(r"snapshot_(.+?)_el", fn)
                files.append(dict(source=src, priority=prio, Experiment_ID=m.group(1), cond=cond,
                                  snapshot=snap.group(1) if snap else "?", path=os.path.join(dp, fn)))
files = pd.DataFrame(files)
print("A. csv files found per source:")
print(files.groupby(["source", "cond"]).size().unstack(fill_value=0).to_string())

# =============================================================================
# B. fingerprint every copy
# =============================================================================
def fingerprint(fp, n=3000):
    raw = pd.read_csv(fp, header=None, skiprows=4, nrows=n, low_memory=False)
    a = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return hashlib.md5(np.round(np.nan_to_num(a), 1).tobytes()).hexdigest()[:12]

files["fingerprint"] = [fingerprint(p) for p in files["path"]]
same_content = files.groupby("fingerprint")["Experiment_ID"].nunique()
if (same_content > 1).any():
    print("\n!! identical data under DIFFERENT IDs:")
    print(files[files["fingerprint"].isin(same_content[same_content > 1].index)]
          [["fingerprint", "Experiment_ID", "source", "cond"]].to_string(index=False))

# =============================================================================
# C. one row per animal
# =============================================================================
def summarise_copies(g):
    g = g.sort_values("priority")
    conds = sorted(set(g["cond"]))
    return pd.Series(dict(cond=g["cond"].iloc[0], cond_all="|".join(conds), cond_conflict=len(conds) > 1,
                          snapshot=g["snapshot"].iloc[0], canonical_source=g["source"].iloc[0],
                          path=g["path"].iloc[0], sources="|".join(g["source"]), n_copies=len(g),
                          copies_identical=g["fingerprint"].nunique() == 1))

meta = files.groupby("Experiment_ID").apply(summarise_copies).reset_index()
meta["batch"] = meta["Experiment_ID"].map(batch_of)
print(f"\nC. unique animals: {len(meta)} (from {len(files)} files)")
print("   animals in more than one source:", int((meta["n_copies"] > 1).sum()),
      "| copies NOT identical:", int((~meta["copies_identical"]).sum()))
if meta["cond_conflict"].any():
    print("!! condition conflicts:\n", meta.loc[meta["cond_conflict"], ["Experiment_ID", "cond_all", "sources"]]
          .to_string(index=False))

# =============================================================================
# D. annotations
# =============================================================================
def parse_new_txt(fp):
    """Detailed annotation: HE, inside house, escape, quality, abnormal events."""
    rx_he, rx_in = re.compile(r"Frames\s+(\d+)\s+to\s+(\d+)"), re.compile(r"INSIDE HOUSE:\s*(\d+)\s+to\s+(end|\d+)", re.I)
    rx_esc = re.compile(r"ESCAPE:\s*(?:NO ESCAPE|Frame\s+(\d+))", re.I)
    rx_q = re.compile(r"(GOOD QUALITY|BLURRED \(VISIBLE TAIL\)|BLURRED)")
    rx_ev = re.compile(r"(abnormal start|failed insertion|insertion fail|twitching)\s*\(frames\s+(\d+)\s*-\s*(\d+)", re.I)
    ann, evs = [], []
    for line in open(fp, encoding="utf-8", errors="replace"):
        if not line.strip().startswith("-ID"):
            continue
        e = RX_ID.search(line).group(1)
        he, inh, esc, q = rx_he.search(line), rx_in.search(line), rx_esc.search(line), rx_q.search(line)
        ann.append(dict(Experiment_ID=e,
                        he_start=int(he.group(1)) if he else np.nan, he_end=int(he.group(2)) if he else np.nan,
                        in_start=int(inh.group(1)) if inh else np.nan,
                        in_end=np.nan if (inh is None or inh.group(2) == "end") else int(inh.group(2)),
                        escape=int(esc.group(1)) if (esc and esc.group(1)) else np.nan,
                        quality=q.group(1) if q else "?", dragging="dragging" in line.lower(),
                        bubble_flag="house bubble" in line.lower()))
        for m in rx_ev.finditer(line):
            evs.append(dict(Experiment_ID=e, event=m.group(1).lower().replace("insertion fail", "failed insertion"),
                            start=int(m.group(2)), end=int(m.group(3))))
    return pd.DataFrame(ann), pd.DataFrame(evs, columns=["Experiment_ID", "event", "start", "end"])

def parse_old_txt(fp):
    """Older annotation: only ID + first 'a to b' / 'a-b' frame range after it is trusted."""
    rows, cur = [], None
    rx_fr = re.compile(r"(\d+)\s*(?:to|-)\s*(\d+)")
    for line in open(fp, encoding="utf-8", errors="replace"):
        m = RX_ID.search(line)
        if m:
            cur = m.group(1)
            line = line[m.end():]
        if cur and not any(r["Experiment_ID"] == cur for r in rows):
            f = rx_fr.search(line)
            if f:
                rows.append(dict(Experiment_ID=cur, old_he_start=int(f.group(1)), old_he_end=int(f.group(2))))
    return pd.DataFrame(rows, columns=["Experiment_ID", "old_he_start", "old_he_end"])

ann_new, events = parse_new_txt(NEW_TXT)
ann_old = parse_old_txt(OLD_TXT) if os.path.isfile(OLD_TXT) else pd.DataFrame(columns=["Experiment_ID", "old_he_start", "old_he_end"])
print(f"\nD. annotations: NEW {len(ann_new)} animals, {len(events)} events | OLD {len(ann_old)} animals")
meta = meta.merge(ann_new, on="Experiment_ID", how="left").merge(ann_old, on="Experiment_ID", how="left")
meta["annot_source"] = np.where(meta["he_start"].notna(), "NEW", np.where(meta["old_he_start"].notna(), "OLD", "none"))
fill = meta["annot_source"] == "OLD"                     # secondary source fills gaps only
meta.loc[fill, "he_start"], meta.loc[fill, "he_end"] = meta.loc[fill, "old_he_start"], meta.loc[fill, "old_he_end"]
both = meta["he_start"].notna() & meta["old_he_start"].notna() & (meta["annot_source"] == "NEW")
meta["old_vs_new_onset_diff_fr"] = np.where(both, meta["old_he_start"] - meta["he_start"], np.nan)
for evn in ["abnormal start", "failed insertion", "twitching"]:
    meta[f"n_{evn.replace(' ', '_')}"] = meta["Experiment_ID"].map(
        events[events["event"] == evn].groupby("Experiment_ID").size()).fillna(0).astype(int)
print("   NEW annotations without a csv:", sorted(set(ann_new["Experiment_ID"]) - set(meta["Experiment_ID"])) or "none")

# =============================================================================
# E. groups
# =============================================================================
def assign_group(r):
    c = "C" if r["cond"] == "control" else "O"
    if r["cond"] == "6-OHDA+DA":
        return "R"
    if pd.notna(r["he_start"]):
        return c + "+"
    if r["Experiment_ID"] in HOUSED_AT_START:
        return c + "h"
    return c + ("-" if r["batch"] == "B1" else "?")

meta["group"] = meta.apply(assign_group, axis=1)
meta["exclude_reason"] = meta["Experiment_ID"].map(EXCLUDE).fillna("")
meta["note"] = np.where((meta["batch"] == "B2026") & (meta["cond"] == "6-OHDA"), NOTES["B2026_6-OHDA"], "")
print("\nE. housed-at-start IDs not found:", sorted(HOUSED_AT_START - set(meta["Experiment_ID"])) or "none")

# =============================================================================
# F. tracking QC (canonical copy)
# =============================================================================
def tracking_qc(fp):
    bps, X, Y, L = read_dlc(fp)
    j = {b: bps.index(b) for b in BP_KEY}
    good = {b: L[:, j[b]] >= LK_MIN for b in BP_KEY}
    ok = good["bodypart1"] & good["bodypart2"]
    trunk = np.hypot(X[ok, j["bodypart1"]] - X[ok, j["bodypart2"]], Y[ok, j["bodypart1"]] - Y[ok, j["bodypart2"]])
    return dict(n_frames=X.shape[0], trunk_len_px=float(np.median(trunk)) if ok.any() else np.nan,
                core_lk=float(np.all([good[b] for b in ["bodypart1", "bodypart2", "bodypart3", "bodypart5"]], axis=0).mean()),
                **{f"cov_{b.replace('bodypart', 'bp')}": float(g.mean()) for b, g in good.items()})

print("\nF. tracking QC …")
meta = pd.concat([meta, pd.DataFrame([tracking_qc(p) for p in meta["path"]])], axis=1)
meta["dur_min"] = meta["n_frames"] / FPS / 60
meta["track_ok"] = meta["core_lk"] >= 0.30              # flag only; exclusion decided per analysis

# =============================================================================
# G. summary + overview plot
# =============================================================================
order = ["C+", "C-", "C?", "Ch", "O+", "O-", "O?", "Oh", "R"]
tab = pd.crosstab(meta["group"], meta["batch"]).reindex(order, fill_value=0)
print("\nG. animals per group x batch:"); print(tab.to_string())
print("   excluded:", meta.loc[meta["exclude_reason"] != "", ["Experiment_ID", "group"]].values.tolist() or "none")
print(meta.groupby("group")[["core_lk", "cov_bp12", "cov_bp6", "trunk_len_px", "dur_min"]].median()
      .reindex(order).round(2).to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
ax = axes[0]
batches = ["B1", "B2", "B2026"]
bottom = np.zeros(3)
for g in order:
    v = tab.reindex(columns=batches, fill_value=0).loc[g].to_numpy()
    ax.bar(batches, v, bottom=bottom, color=GROUP_COL[g], edgecolor="white", label=g)
    for i, (b, h) in enumerate(zip(bottom, v)):
        if h:
            ax.text(i, b + h / 2, f"{g} {int(h)}", ha="center", va="center", fontsize=7)
    bottom += v
ax.set_ylabel("animals"); ax.set_title("Animals per batch and group", fontsize=10)
ax = axes[1]
for i, g in enumerate(order):
    d = meta[meta["group"] == g]
    ax.scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.2, 0.2, len(d)), d["core_lk"],
               color=GROUP_COL[g], s=18, edgecolor="k", linewidth=0.3)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
ax.axhline(0.3, color="red", lw=0.8, ls="--")
ax.set_ylabel("fraction of frames: trunk + tail base tracked"); ax.set_title("Tracking quality", fontsize=10)
ax = axes[2]
he = meta[meta["he_start"].notna()]
for (c, src), d in he.groupby(["cond", "annot_source"]):
    ax.scatter(d["he_start"] / FPS / 60, (d["he_end"] - d["he_start"]) / FPS, label=f"{c} ({src})",
               marker="o" if src == "NEW" else "^", color=GROUP_COL["C+" if c == "control" else "O+"], s=30, alpha=0.8)
ax.set_xlabel("HE onset (min into video)"); ax.set_ylabel("HE duration (s)")
ax.set_title("Annotated house expansions", fontsize=10); ax.legend(fontsize=7, frameon=False)
plt.tight_layout()
save_fig(fig, OUT, "01_inventory_overview", dpi=200)

# =============================================================================
# H. save
# =============================================================================
meta = meta.sort_values(["group", "batch", "Experiment_ID"]).reset_index(drop=True)
meta.to_csv(f"{OUT}/01_metadata_animals.csv", index=False)
files.to_csv(f"{OUT}/01_inventory_all_files.csv", index=False)
events.to_csv(f"{OUT}/01_annotation_events.csv", index=False)
print(f"\nsaved → {OUT}/01_* ({len(meta)} animals, {len(files)} files)")
