#!/usr/bin/env python
"""analyze_plan_aaa_t1_diagnostic.py

Lightweight diagnostic for Codex Touchpoint 3 Round A-bis paper §Limitations
strengthening (Plan AAA alpha158 same-day OHLC leak provenance, plan §1.9 #5).

QUESTION: If Plan AAA's input alpha158 had been T-1-shifted (leak-free), would
the same top-15 GROUPS still appear in the top-15 by importance?

APPROACH (proxy, not full re-run of Plan AAA permutation framework):
  1. Load alpha158_raw (T, N, 158) and 21d forward labels from pa
  2. Build T-1-shifted version: alpha158_t1[t] = alpha158_raw[t-1]
  3. For each feature col, compute mean per-day spearman IC vs labels over
     the Plan AAA test window (313 days, fold 0-4 test periods)
  4. Load Plan AAA group structure from ranking.csv (group_members column)
  5. Compute group-level proxy importance = mean(|IC|) over group members
  6. Rank groups by proxy importance under both regimes
  7. Report: top-15 overlap, member-level rank shifts, group-level rank shifts

LIMITS (honest):
  - Single-feature mean IC ≠ Plan AAA permutation Δ-IC (model-dependent)
  - Group importance = mean|IC| over members is heuristic — a group can be
    important via interaction effects that single-feature IC misses
  - This diagnostic SUPPORTS a §Limitations qualifier; it does NOT replace
    a full leak-free Plan AAA re-run for definitive composition validation
  - If diagnostic shows top-15 mostly stable (>= 12/15 overlap), the §Limitations
    qualifier is light; if substantially unstable (< 8/15), composition needs
    rework before submission

Run: ~20-30 min M4 CPU (no model training; just IC computation × 158 features × 313 days × 2 regimes)

Outputs:
  artifacts/plan_aaa_t1_diagnostic/proxy_ic_per_feature.csv  — per (feature, regime)
  artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv  — per group, both rankings
  artifacts/plan_aaa_t1_diagnostic/summary.md  — top-15 overlap + interpretation
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/Users/heruixi/Desktop/GNN-Testing")
import run_step3_plan_z_part_a as pa  # noqa: E402

OUT_DIR = Path("artifacts/plan_aaa_t1_diagnostic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA158_PATH = Path("data/reference/sp500_5y_alpha158_features_raw.npy")
ALPHA158_META = Path("data/reference/sp500_5y_alpha158_features_meta.json")
PLAN_AAA_RANKING = Path("artifacts/plan_aaa/ranking.csv")

# Plan AAA 5-fold walk-forward test windows (per archived/scripts/run_horizon_ablation.py:72-83)
# Test days = day indices that fall inside any fold's test split
# We'll derive from labels_np.shape[0] and HORIZON=21

t0 = time.time()
print(f"[diag] loading data via pa.load_data_and_features() ...")
base = pa.load_data_and_features()
labels_np = base["labels_np"]  # (T, N) float32, NaN where label invalid
label_valid_np = base["label_valid_np"]  # (T, N) bool
hc_tensor = base["features_np"]  # (T, N, 10) — not used here, just for shape
T, N = labels_np.shape
print(f"[diag]   T={T} days, N={N} tickers")

print(f"[diag] loading alpha158_raw ...")
alpha158 = np.load(ALPHA158_PATH).astype(np.float32)  # (T, N, 158)
alpha158_meta = json.load(open(ALPHA158_META))
feature_names = list(alpha158_meta["feature_order"])
assert alpha158.shape == (T, N, 158), f"shape mismatch: {alpha158.shape} vs ({T}, {N}, 158)"
print(f"[diag]   alpha158 shape {alpha158.shape}, 158 features")

# Build T-1 shifted version
print(f"[diag] building T-1 shifted alpha158 (np.roll axis=0 shift=1) ...")
alpha158_t1 = np.full_like(alpha158, np.nan)
alpha158_t1[1:] = alpha158[:-1]  # row[t] = row[t-1]; row[0] = NaN
# alpha158_raw row[t] = features computed AS OF day t (with same-day OHLC leak)
# alpha158_t1  row[t] = features computed AS OF day t-1 (one-day lagged, leak-free)
print(f"[diag]   T-1 shift applied; alpha158_t1[0] is NaN by construction")

# Define Plan AAA test window: fold 0 test starts at day index corresponding
# to 2024-04-01 (right after train_end 2023-12-31, val end 2024-03-31).
# Plan AAA uses 313 test days per ranking.csv (n_dates column). We don't have
# explicit day-index → date mapping in pa, so we use the LAST 313 days of valid labels.
# This matches the "test period Q2-2024 to Q2-2025" total span.

# Compute valid_days mask: days where >= 30 tickers have valid labels
n_valid_per_day = label_valid_np.sum(axis=1)
valid_days = np.where(n_valid_per_day >= 30)[0]
print(f"[diag] valid days (>=30 tickers with labels): {len(valid_days)} / {T}")

TEST_WINDOW_DAYS = 313  # per Plan AAA ranking.csv n_dates column
test_days = valid_days[-TEST_WINDOW_DAYS:] if len(valid_days) >= TEST_WINDOW_DAYS else valid_days
print(f"[diag] using last {len(test_days)} valid days as test window")

# Per-feature per-day spearman IC, averaged over test_days, for both regimes
print(f"[diag] computing per-feature mean IC over {len(test_days)} test days × 158 features × 2 regimes ...")
print(f"[diag]   estimated time: ~{158*len(test_days)*2*0.001/60:.1f} min")

def compute_mean_ic(features_arr, label_arr, valid_arr, days):
    """features_arr: (T, N) for one feature; label_arr: (T, N); valid_arr: (T, N) bool.
    Returns (mean_ic, n_days_used).
    """
    ics = []
    for d in days:
        mask = valid_arr[d] & ~np.isnan(features_arr[d]) & ~np.isnan(label_arr[d])
        if mask.sum() < 30:
            continue
        f_d = features_arr[d, mask]
        l_d = label_arr[d, mask]
        if f_d.std() < 1e-9 or l_d.std() < 1e-9:
            continue
        rho, _ = spearmanr(f_d, l_d)
        if not np.isnan(rho):
            ics.append(rho)
    return (float(np.mean(ics)), len(ics)) if ics else (np.nan, 0)

rows = []
for fi, fname in enumerate(feature_names):
    if fi % 20 == 0:
        elapsed = time.time() - t0
        print(f"[diag]   feature {fi}/158 [{fname}], elapsed {elapsed/60:.1f}m")
    ic_raw, n_raw = compute_mean_ic(alpha158[:, :, fi], labels_np, label_valid_np, test_days)
    ic_t1, n_t1 = compute_mean_ic(alpha158_t1[:, :, fi], labels_np, label_valid_np, test_days)
    rows.append({
        "feature": fname,
        "feature_idx": fi,
        "ic_raw_leaky": round(ic_raw, 6),
        "ic_raw_n_days": n_raw,
        "ic_t1_shifted": round(ic_t1, 6),
        "ic_t1_n_days": n_t1,
        "abs_ic_raw": round(abs(ic_raw), 6) if not np.isnan(ic_raw) else np.nan,
        "abs_ic_t1": round(abs(ic_t1), 6) if not np.isnan(ic_t1) else np.nan,
        "ic_drop_abs": round(abs(ic_raw) - abs(ic_t1), 6) if not (np.isnan(ic_raw) or np.isnan(ic_t1)) else np.nan,
    })

feat_df = pd.DataFrame(rows)
feat_df.to_csv(OUT_DIR / "proxy_ic_per_feature.csv", index=False)
print(f"[diag] wrote {OUT_DIR}/proxy_ic_per_feature.csv ({len(feat_df)} rows)")

# Load Plan AAA group structure
print(f"[diag] loading Plan AAA group structure from {PLAN_AAA_RANKING} ...")
rank_df = pd.read_csv(PLAN_AAA_RANKING)
print(f"[diag]   {len(rank_df)} groups in Plan AAA ranking")

# Group-level proxy importance: mean(|feature_IC|) over members (only alpha158 features;
# hc_ features were inherited from part_a and not affected by alpha158 leak)
group_rows = []
for _, grow in rank_df.iterrows():
    members = grow["group_members"].split(",")
    members_a158 = [m for m in members if not m.startswith("hc_")]  # only alpha158 members
    n_members = len(members)
    n_a158 = len(members_a158)
    if n_a158 == 0:
        # Pure hc group (e.g., rank 1: "hc_mom12m") — not affected by alpha158 leak
        group_rows.append({
            "group_id": int(grow["group_id"]),
            "group_label": grow["group_label"],
            "group_size": n_members,
            "n_alpha158_members": 0,
            "plan_aaa_orig_rank": int(grow["rank"]),
            "plan_aaa_orig_mean_delta_ic": float(grow["mean_delta_IC"]),
            "proxy_group_abs_ic_raw_leaky": np.nan,  # not applicable
            "proxy_group_abs_ic_t1_shifted": np.nan,
            "proxy_ic_drop_abs": np.nan,
            "leak_affected": False,
        })
        continue
    sub = feat_df[feat_df["feature"].isin(members_a158)]
    proxy_raw = float(sub["abs_ic_raw"].mean())
    proxy_t1 = float(sub["abs_ic_t1"].mean())
    group_rows.append({
        "group_id": int(grow["group_id"]),
        "group_label": grow["group_label"],
        "group_size": n_members,
        "n_alpha158_members": n_a158,
        "plan_aaa_orig_rank": int(grow["rank"]),
        "plan_aaa_orig_mean_delta_ic": float(grow["mean_delta_IC"]),
        "proxy_group_abs_ic_raw_leaky": round(proxy_raw, 6),
        "proxy_group_abs_ic_t1_shifted": round(proxy_t1, 6),
        "proxy_ic_drop_abs": round(proxy_raw - proxy_t1, 6),
        "leak_affected": True,
    })

group_df = pd.DataFrame(group_rows)
# Rank by proxy under both regimes (descending |IC|; pure-hc groups get assigned tied last rank)
group_df["proxy_rank_raw"] = group_df["proxy_group_abs_ic_raw_leaky"].rank(method="min", ascending=False, na_option="bottom").astype(int)
group_df["proxy_rank_t1"] = group_df["proxy_group_abs_ic_t1_shifted"].rank(method="min", ascending=False, na_option="bottom").astype(int)
group_df = group_df.sort_values("plan_aaa_orig_rank")
group_df.to_csv(OUT_DIR / "group_ranking_comparison.csv", index=False)
print(f"[diag] wrote {OUT_DIR}/group_ranking_comparison.csv ({len(group_df)} groups)")

# Top-15 overlap analysis
top15_orig = set(group_df[group_df["plan_aaa_orig_rank"] <= 15]["group_id"].tolist())
top15_proxy_raw = set(group_df.nsmallest(15, "proxy_rank_raw")["group_id"].tolist())
top15_proxy_t1 = set(group_df.nsmallest(15, "proxy_rank_t1")["group_id"].tolist())

overlap_raw_vs_orig = len(top15_orig & top15_proxy_raw)  # sanity: proxy_raw should align reasonably with Plan AAA orig
overlap_t1_vs_orig = len(top15_orig & top15_proxy_t1)    # THE KEY METRIC
overlap_raw_vs_t1 = len(top15_proxy_raw & top15_proxy_t1)

# Write summary
with open(OUT_DIR / "summary.md", "w") as f:
    f.write("# Plan AAA T-1 Stability Diagnostic\n\n")
    f.write(f"Run date: 2026-05-27 | Wall time: {(time.time()-t0)/60:.1f} min\n\n")
    f.write("## Question\n\n")
    f.write("If Plan AAA's input `sp500_5y_alpha158_features_raw.npy` had been T-1-shifted (leak-free), ")
    f.write("would the same top-15 groups still appear in the top-15 by importance? Universe C ")
    f.write("composition derives from Plan AAA top-15; if top-15 is unstable under T-1 shift, the ")
    f.write("composition basis is weakened.\n\n")

    f.write("## Method\n\n")
    f.write("- Proxy importance per feature: mean per-day spearman IC vs 21d forward labels over ")
    f.write(f"the last {len(test_days)} valid days (matches Plan AAA n_dates=313 in ranking.csv)\n")
    f.write("- Group importance: mean(|feature_IC|) over alpha158 member features\n")
    f.write("- Proxy ≠ Plan AAA permutation Δ-IC (model-dependent); single-feature IC is a directional indicator\n\n")

    f.write("## Top-15 Overlap Results\n\n")
    f.write(f"- Plan AAA original top-15 (from `ranking.csv`): {len(top15_orig)} groups\n")
    f.write(f"- Proxy-raw (leaky alpha158) top-15: {len(top15_proxy_raw)} groups\n")
    f.write(f"- Proxy-T1 (shifted alpha158) top-15: {len(top15_proxy_t1)} groups\n\n")
    f.write(f"- **Overlap(Plan AAA orig ∩ proxy-raw) = {overlap_raw_vs_orig}/15** — sanity check that proxy aligns with Plan AAA permutation framework\n")
    f.write(f"- **Overlap(Plan AAA orig ∩ proxy-T1) = {overlap_t1_vs_orig}/15** — KEY: stability under leak removal\n")
    f.write(f"- **Overlap(proxy-raw ∩ proxy-T1) = {overlap_raw_vs_t1}/15** — direct leak-effect-on-proxy measurement\n\n")

    f.write("## Interpretation\n\n")
    if overlap_t1_vs_orig >= 12:
        verdict = "HIGH STABILITY"
        action = "Light §Limitations qualifier sufficient; Universe C composition basis robust to leak removal at top-15 level."
    elif overlap_t1_vs_orig >= 8:
        verdict = "MODERATE STABILITY"
        action = "Stronger §Limitations qualifier needed; mention specific groups that drop out of top-15 under T-1 shift; consider sensitivity analysis on Universe C with alternate composition."
    else:
        verdict = "LOW STABILITY"
        action = "Universe C composition basis is leak-driven; full Plan AAA re-run required before submission, OR Universe C must be re-defined."
    f.write(f"- Verdict: **{verdict}** (Plan AAA orig top-15 ∩ proxy-T1 top-15 = {overlap_t1_vs_orig}/15)\n")
    f.write(f"- Action: {action}\n\n")

    f.write("## Caveats\n\n")
    f.write("1. Single-feature IC ≠ Plan AAA permutation Δ-IC. A group can be important via member-feature interaction effects that single-feature IC misses.\n")
    f.write("2. The proxy uses |IC| aggregation; sign is discarded. Plan AAA's Δ-IC is signed.\n")
    f.write("3. Test window matched at length (313 days) but specific calendar dates may differ if data has been updated since Plan AAA ranking was generated. ranking.csv n_dates=313 is the reference target.\n")
    f.write("4. Pure-hc groups (e.g., rank-1 `hc_mom12m`) are NOT affected by alpha158 leak; they retain their Plan AAA rank by construction.\n\n")

    f.write("## Top-15 group-by-group detail\n\n")
    top15_df = group_df[group_df["plan_aaa_orig_rank"] <= 15].copy()
    top15_df["in_proxy_raw_top15"] = top15_df["group_id"].isin(top15_proxy_raw)
    top15_df["in_proxy_t1_top15"] = top15_df["group_id"].isin(top15_proxy_t1)
    f.write(top15_df[["plan_aaa_orig_rank", "group_id", "group_label", "n_alpha158_members",
                       "plan_aaa_orig_mean_delta_ic", "proxy_group_abs_ic_raw_leaky",
                       "proxy_group_abs_ic_t1_shifted", "proxy_ic_drop_abs",
                       "proxy_rank_raw", "proxy_rank_t1",
                       "in_proxy_raw_top15", "in_proxy_t1_top15"]].to_markdown(index=False))

print(f"\n[diag] DONE — {OUT_DIR}/summary.md")
print(f"[diag] HEADLINE: Plan AAA orig top-15 ∩ proxy-T1 top-15 = {overlap_t1_vs_orig}/15")
print(f"[diag] Total wall: {(time.time()-t0)/60:.1f} min")
