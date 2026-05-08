#!/usr/bin/env python3
"""Reference-paper comparison analysis for LLM Behavioral Coherence.

Generates figures, statistical tests, and results tables from core_fresh experiments.

Usage:
    python paper_analysis.py --mode all
    python paper_analysis.py --mode figures --focal-model 2
    python paper_analysis.py --mode tests
    python paper_analysis.py --mode table
    python paper_analysis.py --mode extras
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    DEFAULT_RESULTS_SUBDIR,
    build_output_path,
    ensure_dir,
    load_model_registry,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOPIC_CONTENTIOUSNESS = {0: 3, 1: 3, 2: 3, 3: 2, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1}
TOPIC_NAMES = [
    "taxes", "immigration", "healthcare", "electric_scooters",
    "student_athletes", "remote_work", "spring_vs_fall",
    "beaches_vs_mountains", "coke_vs_pepsi",
]
# Analyze every model in configs/models.yaml by default. If a completed
# experiment lacks artifacts for a model, the dataframe builder naturally skips it.
EXCLUDE_MODELS = set()
FOCAL_MODEL = 8  # Gemma-3-12b

P_THRESHOLD = 0.01
BOOTSTRAP_N = 10000
SEED = 42

# Matplotlib defaults
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 14
ANNOTATION_TEXT_SIZE = 13
INSET_TEXT_SIZE = 13
TITLE_SIZE = 15

QUAL_FIGURE_NAMES = [
    "result_pref_base",
    "result_pref_no_disagreement",
    "result_pref_agree_less",
    "result_pref_topic",
    "result_bias_asymmetry",
    "result_open_base",
    "result_open_subsample",
]

REQUIRED_PNG_PDF_FIGURES = {
    "result_bias_asymmetry",
    "result_open_subsample",
    "result_pref_agree_less",
    "result_pref_base",
    "result_pref_topic",
}

FIGURE_STYLE_VERSIONS = {
    "result_pref_base": "v3",
    "result_pref_no_disagreement": "v1",
    "result_pref_agree_less": "v4",
    "result_pref_topic": "v3",
    "result_bias_asymmetry": "v4",
    "result_open_base": "v1",
    "result_open_subsample": "v4",
}

FIGURE_HASH_COLUMNS = {
    "result_pref_base": ["pref_gap", "agreement"],
    "result_pref_no_disagreement": ["pref_gap", "agreement", "bias1", "bias2"],
    "result_pref_agree_less": ["pref1", "pref2", "agreement"],
    "result_pref_topic": ["pref_gap", "pref1", "pref2", "contentiousness", "agreement"],
    "result_bias_asymmetry": ["pref_gap", "bias1", "bias2", "agreement"],
    "result_open_base": ["combined_openness", "agreement"],
    "result_open_subsample": ["pref1", "pref2", "open1", "open2", "agreement"],
}

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": INSET_TEXT_SIZE,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Step 1: Data loading
# ---------------------------------------------------------------------------
def _safe_jsonl_rows(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_master_dataframe(output_root, experiment_name):
    """Build a flat DataFrame where each row = one conversation."""
    results_root = build_output_path(output_root, DEFAULT_RESULTS_SUBDIR, experiment_name)
    all_rows = []

    models = load_model_registry()
    model_ids = [int(m["id"]) for m in models if int(m["id"]) not in EXCLUDE_MODELS]

    for model_id in model_ids:
        for topic_id in range(9):
            for combined_bias_id in range(9):
                stage_dir = os.path.join(results_root, str(model_id), str(topic_id), str(combined_bias_id))
                conv_path = os.path.join(stage_dir, "conversations.jsonl")
                judge_path = os.path.join(stage_dir, "judge_scores.npy")

                conv_rows = _safe_jsonl_rows(conv_path)
                if not conv_rows or not os.path.exists(judge_path):
                    continue

                judge_scores = np.load(judge_path, allow_pickle=True)

                for row in conv_rows:
                    pb1 = row["pair_bin_1"]
                    pb2 = row["pair_bin_2"]
                    sidx = row["sample_idx"]

                    # Extract final-window judge score
                    try:
                        agreement = int(judge_scores[pb1, pb2, sidx, -1])
                    except (IndexError, ValueError):
                        continue

                    pref1 = row["agent_1_topic_response"]
                    pref2 = row["agent_2_topic_response"]
                    open1 = row["agent_1_persuadability"]
                    open2 = row["agent_2_persuadability"]

                    all_rows.append({
                        "model_id": model_id,
                        "topic_id": topic_id,
                        "topic_name": TOPIC_NAMES[topic_id],
                        "combined_bias_id": combined_bias_id,
                        "bias1": combined_bias_id // 3,
                        "bias2": combined_bias_id % 3,
                        "pair_bin_1": pb1,
                        "pair_bin_2": pb2,
                        "sample_idx": sidx,
                        "pref1": pref1,
                        "pref2": pref2,
                        "pref_gap": abs(pref1 - pref2),
                        "open1": open1,
                        "open2": open2,
                        "combined_openness": open1 + open2,
                        "agreement": agreement,
                        "contentiousness": TOPIC_CONTENTIOUSNESS[topic_id],
                        "conversation_quality_flag": row.get("conversation_quality_flag", False),
                    })

    df = pd.DataFrame(all_rows)
    # Filter invalid judge scores
    df = df[df["agreement"] >= 0].copy()
    return df


def get_or_build_dataframe(output_root, experiment_name, cache_dir):
    """Load cached parquet or build from scratch."""
    cache_path = os.path.join(cache_dir, "master_df.parquet")
    if os.path.exists(cache_path):
        print(f"Loading cached dataframe from {cache_path}")
        return pd.read_parquet(cache_path)

    print("Building master dataframe from raw data...")
    df = build_master_dataframe(output_root, experiment_name)
    ensure_dir(cache_dir)
    df.to_parquet(cache_path, index=False)
    print(f"Cached dataframe ({len(df)} rows) to {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Bootstrap utilities
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(data, n_boot=BOOTSTRAP_N, ci=0.95, rng=None):
    """Return (mean, lo, hi) via bootstrap."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return float(np.mean(data)), float(lo), float(hi)


def bootstrap_statistic(data, stat_fn, n_boot=BOOTSTRAP_N, ci=0.95, rng=None):
    """Return (stat, lo, hi) via bootstrap for arbitrary statistic."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    stat = stat_fn(data)
    boot_stats = np.array([
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_stats, [alpha, 1 - alpha])
    return float(stat), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Step 3: Qualitative figures (focal model)
# ---------------------------------------------------------------------------
def _pref_pair_label(p1, p2):
    return f"({min(p1,p2)},{max(p1,p2)})"


def _save_fig(fig, output_dir, name, formats):
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {name}")


def _normalize_formats(formats):
    seen = set()
    ordered = []
    for fmt in formats:
        fmt_l = fmt.lower()
        if fmt_l not in seen:
            seen.add(fmt_l)
            ordered.append(fmt_l)
    return ordered


def _parse_figure_format_map(raw_entries):
    fmt_map = {}
    for entry in raw_entries or []:
        if "=" not in entry:
            raise ValueError(f"Invalid --figure-format-map entry: {entry}")
        fig_name, fmt_text = entry.split("=", 1)
        fig_name = fig_name.strip()
        if fig_name not in QUAL_FIGURE_NAMES:
            raise ValueError(f"Unknown figure in --figure-format-map: {fig_name}")
        fmts = [v.strip().lower() for v in fmt_text.split(",") if v.strip()]
        if not fmts:
            raise ValueError(f"No formats provided for figure: {fig_name}")
        fmt_map[fig_name] = _normalize_formats(fmts)
    return fmt_map


def _resolve_formats_for_figure(figure_name, default_formats, figure_format_map):
    formats = figure_format_map.get(figure_name, default_formats)
    formats = _normalize_formats(formats)
    if figure_name in REQUIRED_PNG_PDF_FIGURES:
        if "png" not in formats:
            formats = formats + ["png"]
        if "pdf" not in formats:
            formats = formats + ["pdf"]
    return formats


def _figure_cache_manifest_path(cache_dir):
    return os.path.join(cache_dir, "figure_cache_manifest.json")


def _load_figure_cache_manifest(cache_dir):
    ensure_dir(cache_dir)
    path = _figure_cache_manifest_path(cache_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {}


def _save_figure_cache_manifest(cache_dir, manifest):
    ensure_dir(cache_dir)
    path = _figure_cache_manifest_path(cache_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _stable_dataframe_hash(df, columns):
    if df.empty:
        return "empty"
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return "no-columns"
    normalized = df[cols].copy()
    normalized = normalized.sort_values(cols).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(normalized, index=False)
    return hashlib.sha256(hashed.values.tobytes()).hexdigest()


def _build_figure_cache_key(figure_name, df, formats, focal_model):
    payload = {
        "figure_name": figure_name,
        "focal_model": int(focal_model),
        "formats": sorted(formats),
        "style_version": FIGURE_STYLE_VERSIONS.get(figure_name, "v1"),
        "row_count": int(len(df)),
        "data_hash": _stable_dataframe_hash(df, FIGURE_HASH_COLUMNS.get(figure_name, [])),
    }
    digest_input = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def _figure_cache_hit(manifest, figure_name, cache_key, output_dir, formats, use_cache):
    if not use_cache:
        return False
    entry = manifest.get(figure_name)
    if not entry or entry.get("key") != cache_key:
        return False
    for fmt in formats:
        if not os.path.exists(os.path.join(output_dir, f"{figure_name}.{fmt}")):
            return False
    return True


def _style_target_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=1.2, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def fig_pref_base(df, output_dir, formats):
    """Figure 3a: Preference gap vs agreement."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    gaps = sorted(df["pref_gap"].unique())
    means, los, his = [], [], []
    for g in gaps:
        subset = df[df["pref_gap"] == g]["agreement"].values
        m, lo, hi = bootstrap_mean_ci(subset)
        means.append(m)
        los.append(m - lo)
        his.append(hi - m)

    ax.bar(gaps, means, yerr=[los, his], capsize=4, color="steelblue", edgecolor="black", width=0.6)
    _style_target_axes(ax, "Preference Gap", "Mean Agreement Score")
    ax.set_xticks(gaps)
    ax.set_ylim(0, 6)

    # Down-right expected-relationship cue around the upper middle of the panel.
    x_min, x_max = min(gaps) - 0.4, max(gaps) + 0.4
    x_start = x_min + 0.45 * (x_max - x_min)
    x_end = x_start + 0.2 * (x_max - x_min)
    y_start = 4.3
    y_end = 3.4
    ax.annotate(
        "",
        xy=(x_end, y_end),
        xytext=(x_start, y_start),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.4),
    )
    ax.text(
        (x_start + x_end) / 2.0,
        y_start + 0.15,
        "Expected Relationship",
        ha="center",
        va="bottom",
        fontsize=ANNOTATION_TEXT_SIZE,
        color="black",
    )

    fig.tight_layout()
    _save_fig(fig, output_dir, "result_pref_base", formats)


def fig_pref_no_disagreement(df, output_dir, formats):
    """Figure 3b: Disagreement dampening — observed vs expected distributions."""
    gaps = sorted(df["pref_gap"].unique())
    gap0_scores = df[df["pref_gap"] == 0]["agreement"].values
    gap0_dist = np.bincount(gap0_scores, minlength=7)[1:6]  # scores 1-5
    gap0_dist_norm = gap0_dist / gap0_dist.sum() if gap0_dist.sum() > 0 else gap0_dist

    # Inverted gap=0 distribution (score k -> score 6-k)
    inverted_dist = gap0_dist_norm[::-1]

    # For linear interpolation of expected means
    obs_mean_0 = np.mean(gap0_scores) if len(gap0_scores) > 0 else 3.0
    inv_mean = np.sum(inverted_dist * np.arange(1, 6))

    n_cols = 3
    display_gaps = [g for g in gaps if g > 0]
    n_rows = len(display_gaps)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.5 * n_rows), squeeze=False)

    score_range = np.arange(1, 6)

    for row_idx, gap in enumerate(display_gaps):
        obs_scores = df[df["pref_gap"] == gap]["agreement"].values
        obs_dist = np.bincount(obs_scores, minlength=7)[1:6]
        obs_dist_norm = obs_dist / obs_dist.sum() if obs_dist.sum() > 0 else obs_dist
        obs_mean = np.mean(obs_scores) if len(obs_scores) > 0 else 0

        # Linear shift expected
        t = gap / 4.0
        expected_linear = (1 - t) * gap0_dist_norm + t * inverted_dist
        expected_linear_mean = np.sum(expected_linear * score_range)

        # Non-linear (sigmoidal) shift
        t_sig = 1 / (1 + np.exp(-6 * (gap / 4.0 - 0.5)))
        expected_sig = (1 - t_sig) * gap0_dist_norm + t_sig * inverted_dist
        expected_sig_mean = np.sum(expected_sig * score_range)

        # Bias=2 subsample
        df_b2 = df[(df["pref_gap"] == gap) & (df["bias1"] == 2) & (df["bias2"] == 2)]
        obs_b2 = df_b2["agreement"].values
        obs_b2_dist = np.bincount(obs_b2, minlength=7)[1:6] if len(obs_b2) > 0 else np.zeros(5)
        obs_b2_norm = obs_b2_dist / obs_b2_dist.sum() if obs_b2_dist.sum() > 0 else obs_b2_dist
        obs_b2_mean = np.mean(obs_b2) if len(obs_b2) > 0 else 0

        for col_idx, (exp_d, exp_m, obs_d_plot, obs_m_plot, title_suffix) in enumerate([
            (expected_linear, expected_linear_mean, obs_dist_norm, obs_mean, "Linear Shift"),
            (expected_sig, expected_sig_mean, obs_dist_norm, obs_mean, "Non-Linear Shift"),
            (expected_linear, expected_linear_mean, obs_b2_norm, obs_b2_mean, "$B_i=2$"),
        ]):
            ax = axes[row_idx, col_idx]
            w = 0.35
            ax.bar(score_range - w / 2, obs_d_plot, width=w, label="Observed", color="steelblue", edgecolor="black")
            ax.bar(score_range + w / 2, exp_d, width=w, label="Expected", color="salmon", edgecolor="black")
            ax.axvline(obs_m_plot, color="steelblue", linestyle="--", linewidth=1)
            ax.axvline(exp_m, color="salmon", linestyle="--", linewidth=1)

            if row_idx == 0:
                ax.set_title(title_suffix)
            if col_idx == 0:
                ax.set_ylabel(f"Gap={gap}")
            ax.set_xticks(score_range)
            ax.set_ylim(0, 1.0)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

    fig.tight_layout()
    _save_fig(fig, output_dir, "result_pref_no_disagreement", formats)


def fig_pref_agree_less(df, output_dir, formats):
    """Figure 3c: Sentiment asymmetry — violin plots."""
    # Anchor=1: pairs (1,1), (1,2), (1,3), (1,4)
    # Anchor=5: pairs (5,5), (4,5), (3,5), (2,5)
    anchor1_pairs = [(1, 1), (1, 2), (1, 3), (1, 4)]
    anchor5_pairs = [(5, 5), (4, 5), (3, 5), (2, 5)]

    plot_data = []
    for gap_idx, (pair1, pair5) in enumerate(zip(anchor1_pairs, anchor5_pairs)):
        for pair, anchor_label in [(pair1, "Anchor=1"), (pair5, "Anchor=5")]:
            p_lo, p_hi = min(pair), max(pair)
            subset = df[
                ((df["pref1"] == pair[0]) & (df["pref2"] == pair[1])) |
                ((df["pref1"] == pair[1]) & (df["pref2"] == pair[0]))
            ]
            for val in subset["agreement"].values:
                plot_data.append({
                    "gap": gap_idx,
                    "anchor": anchor_label,
                    "pair_label": f"{pair[0]}-{pair[1]}",
                    "agreement": val,
                })

    pdf = pd.DataFrame(plot_data)
    if pdf.empty:
        print("  WARNING: No data for pref_agree_less figure")
        return

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), sharey=True)
    for gap_idx, ax in enumerate(axes):
        sub = pdf[pdf["gap"] == gap_idx]
        if sub.empty:
            continue
        sns.violinplot(data=sub, x="pair_label", y="agreement", hue="anchor",
                       split=True, ax=ax, inner="quartile", palette={"Anchor=1": "steelblue", "Anchor=5": "salmon"},
                       density_norm="width", cut=0)
        ax.set_xlabel("")
        if gap_idx == 0:
            ax.set_ylabel("Agreement Score", fontsize=AXIS_LABEL_SIZE)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE, width=1.2, length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        if ax.legend_:
            ax.legend_.remove()

    axes[0].text(
        0.78,
        0.5,
        "Should mirror\nleft-distribution",
        transform=axes[0].transAxes,
        ha="center",
        va="center",
        fontsize=INSET_TEXT_SIZE,
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.95),
    )

    fig.supxlabel("Preference Pairs", fontsize=AXIS_LABEL_SIZE)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save_fig(fig, output_dir, "result_pref_agree_less", formats)


def fig_pref_topic(df, output_dir, formats):
    """Figure 3d: Contentiousness at gap=0 — grouped bars by C level.

    Full sample at gap=0, grouped bars for C=1,2,3 per preference pair.
    Directly visualizes Test 4: at gap=0, C shouldn't matter, but it does.
    """
    gap0 = df[df["pref_gap"] == 0].copy()
    if gap0.empty:
        print("  WARNING: No gap=0 data for pref_topic figure")
        return

    gap0["pair_label"] = gap0.apply(
        lambda r: f"({int(r['pref1'])},{int(r['pref2'])})", axis=1
    )

    fig, ax = plt.subplots(figsize=(5, 4.5))

    contentiousness_levels = [1, 2, 3]
    colors = {1: "forestgreen", 2: "darkorange", 3: "firebrick"}

    pair_labels = sorted(gap0["pair_label"].unique())
    x_pos = np.arange(len(pair_labels))
    n_groups = len(contentiousness_levels)
    bar_width = 0.25
    offsets = {c: (i - (n_groups - 1) / 2) * bar_width
               for i, c in enumerate(contentiousness_levels)}
    pair_tops = {pl: 0.0 for pl in pair_labels}

    for c_level in contentiousness_levels:
        c_data = gap0[gap0["contentiousness"] == c_level]
        means, errs_lo, errs_hi = [], [], []
        for pl in pair_labels:
            vals = c_data[c_data["pair_label"] == pl]["agreement"].values
            if len(vals) > 0:
                m, lo, hi = bootstrap_mean_ci(vals)
            else:
                m, lo, hi = np.nan, np.nan, np.nan
            means.append(m)
            errs_lo.append(m - lo if not np.isnan(m) else 0)
            errs_hi.append(hi - m if not np.isnan(m) else 0)

        ax.bar(x_pos + offsets[c_level], means, width=bar_width,
               yerr=[errs_lo, errs_hi], capsize=3,
               label=f"C={c_level}", color=colors[c_level],
               edgecolor="black", linewidth=0.5)
        for idx, pl in enumerate(pair_labels):
            m = means[idx]
            top = (m + errs_hi[idx]) if not np.isnan(m) else 0.0
            pair_tops[pl] = max(pair_tops[pl], float(top))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(pair_labels, fontsize=TICK_LABEL_SIZE)
    _style_target_axes(ax, "Preference Pair (gap=0)", "Mean Agreement Score")
    ax.legend(fontsize=INSET_TEXT_SIZE)
    ax.set_ylim(0, 6)

    if "(2,2)" in pair_labels:
        idx = pair_labels.index("(2,2)")
        xc = x_pos[idx]
        x_left = xc + offsets[1] - bar_width / 2
        x_right = xc + offsets[3] + bar_width / 2
        y_bracket = min(5.75, pair_tops["(2,2)"] + 0.18)
        y_drop = y_bracket - 0.1
        ax.plot([x_left, x_right], [y_bracket, y_bracket], color="black", linewidth=1.8)
        ax.plot([x_left, x_left], [y_drop, y_bracket], color="black", linewidth=1.8)
        ax.plot([x_right, x_right], [y_drop, y_bracket], color="black", linewidth=1.8)
        ax.text(
            (x_left + x_right) / 2.0,
            min(5.95, y_bracket + 0.06),
            "should_be_equal",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_TEXT_SIZE,
            color="black",
        )

    fig.tight_layout()
    _save_fig(fig, output_dir, "result_pref_topic", formats)


def fig_bias_asymmetry(df, output_dir, formats):
    """Two-bar comparison: B=2 effect at gap=0 vs gap=max.

    Shows only the two gaps that matter for Test 2, with bootstrap CIs
    and expected-direction annotations. Skips gaps with no B=2 data.
    """
    max_gap = df["pref_gap"].max()
    if pd.isna(max_gap):
        print("  WARNING: No pref_gap data for bias_asymmetry figure")
        return

    rng = np.random.default_rng(SEED)

    def _b2_effect_ci(gap_val):
        """Compute B=2 effect size (B2_mean − other_mean) with bootstrap CI."""
        gdata = df[df["pref_gap"] == gap_val]
        b2 = gdata[(gdata["bias1"] == 2) & (gdata["bias2"] == 2)]["agreement"].values
        other = gdata[~((gdata["bias1"] == 2) & (gdata["bias2"] == 2))]["agreement"].values
        if len(b2) < 2 or len(other) < 2:
            return None
        effect = np.mean(b2) - np.mean(other)
        boot_effects = []
        for _ in range(BOOTSTRAP_N):
            b2_s = rng.choice(b2, size=len(b2), replace=True)
            ot_s = rng.choice(other, size=len(other), replace=True)
            boot_effects.append(np.mean(b2_s) - np.mean(ot_s))
        boot_effects = np.array(boot_effects)
        lo, hi = np.quantile(boot_effects, [0.025, 0.975])
        return effect, lo, hi, len(b2), len(other)

    # Compute effects for gap=0 and gap=max
    bar_data = []
    for gap_val, label in [
        (0, "Gap = 0"),
        (max_gap, f"Gap = {int(max_gap)}"),
    ]:
        result = _b2_effect_ci(gap_val)
        if result is not None:
            effect, lo, hi, n_b2, n_other = result
            bar_data.append({
                "label": label, "effect": effect, "lo": lo, "hi": hi,
                "n_b2": n_b2, "n_other": n_other,
            })

    if not bar_data:
        print("  WARNING: No B=2 data at gap=0 or gap=max for bias_asymmetry figure")
        return

    fig, ax = plt.subplots(figsize=(5, 4))

    x_pos = np.arange(len(bar_data))
    effects = [d["effect"] for d in bar_data]
    errs_lo = [d["effect"] - d["lo"] for d in bar_data]
    errs_hi = [d["hi"] - d["effect"] for d in bar_data]
    bar_colors = ["steelblue" if e >= 0 else "salmon" for e in effects]
    labels = [d["label"] for d in bar_data]

    ax.bar(x_pos, effects, yerr=[errs_lo, errs_hi], capsize=6,
           color=bar_colors, edgecolor="black", width=0.5, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    arrow_color = "black"
    if len(bar_data) >= 1:
        ax.annotate(
            "",
            xy=(0, 0.1),
            xytext=(0, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=2.0),
        )
        ax.text(
            0,
            0.12,
            "Expected\nDirection",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_TEXT_SIZE,
            color="black",
        )
    if len(bar_data) >= 2:
        ax.annotate(
            "",
            xy=(1, -0.1),
            xytext=(1, 0.0),
            arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=2.0),
        )
        ax.text(
            1,
            -0.12,
            "Expected\nDirection",
            ha="center",
            va="top",
            fontsize=ANNOTATION_TEXT_SIZE,
            color="black",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=TICK_LABEL_SIZE)
    _style_target_axes(ax, "", "B=2 Effect ($\\bar{x}_{B=2} - \\bar{x}_{other}$)")
    ax.set_ylim(-0.4, 0.6)

    fig.tight_layout()
    _save_fig(fig, output_dir, "result_bias_asymmetry", formats)


def fig_open_base(df, output_dir, formats):
    """Figure 3e: Combined openness vs agreement."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    openness_vals = sorted(df["combined_openness"].unique())
    means, los, his = [], [], []
    for o in openness_vals:
        vals = df[df["combined_openness"] == o]["agreement"].values
        m, lo, hi = bootstrap_mean_ci(vals)
        means.append(m)
        los.append(m - lo)
        his.append(hi - m)

    ax.bar(openness_vals, means, yerr=[los, his], capsize=3, color="steelblue",
           edgecolor="black", width=0.7)
    ax.set_xlabel("Combined Openness (sum)")
    ax.set_ylabel("Mean Agreement Score")
    ax.set_xticks(openness_vals)
    ax.set_ylim(0, 6)
    fig.tight_layout()
    _save_fig(fig, output_dir, "result_open_base", formats)


def fig_open_subsample(df, output_dir, formats):
    """Figure 3f: Openness fails at max divergence — horizontal bars for pref pair (1,5)."""
    sub = df[
        ((df["pref1"] == 1) & (df["pref2"] == 5)) |
        ((df["pref1"] == 5) & (df["pref2"] == 1))
    ]
    if sub.empty:
        print("  WARNING: No data for open_subsample figure")
        return

    # Group by combined openness to keep the chart compact and directly comparable.
    grouped = sub.copy()
    grouped["combined_open"] = grouped["open1"] + grouped["open2"]
    grouped = grouped.groupby("combined_open", as_index=False).agg(
        mean_agreement=("agreement", "mean"),
        count=("agreement", "count"),
    )
    grouped = grouped[(grouped["combined_open"] >= 5) & (grouped["combined_open"] <= 17)].copy()
    if grouped.empty:
        print("  WARNING: No combined openness data in [5, 17] for open_subsample figure")
        return

    baseline_level = 5 if (grouped["combined_open"] == 5).any() else int(grouped["combined_open"].min())
    baseline_mean = float(grouped[grouped["combined_open"] == baseline_level]["mean_agreement"].iloc[0])
    grouped["diff"] = grouped["mean_agreement"] - baseline_mean
    grouped = grouped.sort_values("combined_open")

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    colors = ["steelblue" if d > 0 else "salmon" for d in grouped["diff"]]
    ax.barh(range(len(grouped)), grouped["diff"].values, color=colors, edgecolor="black")
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(grouped["combined_open"].astype(int).astype(str).tolist(), fontsize=TICK_LABEL_SIZE)
    ax.invert_yaxis()
    _style_target_axes(
        ax,
        f"Mean Agreement Difference\nfrom Combined Openness {baseline_level}",
        "Combined Openness\n(open1 + open2)",
    )
    ax.axvline(0, color="black", linewidth=0.8)
    x_left = min(-0.5, float(grouped["diff"].min()) - 0.05)
    ax.set_xlim(x_left, 0.5)

    y_arrow = (len(grouped) - 1) / 2.0
    x_start = 0.0
    x_end = 0.16
    ax.annotate(
        "",
        xy=(x_end, y_arrow),
        xytext=(x_start, y_arrow),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=4, mutation_scale=18),
    )
    ax.text(
        x_end + 0.015,
        y_arrow,
        "Expected Distribution",
        ha="left",
        va="center",
        rotation=90,
        fontsize=ANNOTATION_TEXT_SIZE,
        color="black",
    )

    fig.tight_layout()
    _save_fig(fig, output_dir, "result_open_subsample", formats)


def generate_all_figures(
    df,
    focal_model,
    output_dir,
    formats,
    target_figures=None,
    figure_format_map=None,
    figure_cache_dir=None,
    use_figure_cache=True,
):
    """Generate qualitative figures for the focal model with optional caching."""
    fdf = df[df["model_id"] == focal_model].copy()
    print(f"Generating figures for model {focal_model} ({len(fdf)} rows)...")
    ensure_dir(output_dir)
    default_formats = _normalize_formats(formats)
    figure_format_map = figure_format_map or {}
    target_figures = target_figures or list(QUAL_FIGURE_NAMES)

    unknown_targets = [f for f in target_figures if f not in QUAL_FIGURE_NAMES]
    if unknown_targets:
        raise ValueError(f"Unknown target figures: {unknown_targets}")

    cache_dir = figure_cache_dir or os.path.join(output_dir, ".cache")
    manifest = _load_figure_cache_manifest(cache_dir)

    figure_specs = [
        ("result_pref_base", fig_pref_base),
        ("result_pref_no_disagreement", fig_pref_no_disagreement),
        ("result_pref_agree_less", fig_pref_agree_less),
        ("result_pref_topic", fig_pref_topic),
        ("result_bias_asymmetry", fig_bias_asymmetry),
        ("result_open_base", fig_open_base),
        ("result_open_subsample", fig_open_subsample),
    ]

    for figure_name, figure_fn in figure_specs:
        if figure_name not in target_figures:
            continue
        resolved_formats = _resolve_formats_for_figure(figure_name, default_formats, figure_format_map)
        cache_key = _build_figure_cache_key(figure_name, fdf, resolved_formats, focal_model)
        if _figure_cache_hit(
            manifest, figure_name, cache_key, output_dir, resolved_formats, use_figure_cache
        ):
            print(f"  Cache hit for {figure_name}; skipping")
            continue
        figure_fn(fdf, output_dir, resolved_formats)
        manifest[figure_name] = {"key": cache_key, "formats": sorted(resolved_formats)}

    _save_figure_cache_manifest(cache_dir, manifest)


# ---------------------------------------------------------------------------
# Step 4: Statistical tests
# ---------------------------------------------------------------------------
def _test1_pref_gap_agreement(df):
    """Pearson r(pref_gap, agreement): expect r < 0, p < 0.01."""
    r, p = stats.pearsonr(df["pref_gap"], df["agreement"])
    passed = (r < 0) and (p < P_THRESHOLD)
    return {"test": 1, "name": "Pref gap → lower agreement",
            "r": r, "p": p, "passed": passed}


def _test2_bias_asymmetry(df):
    """Bias instruction asymmetry: B=2 should amplify agreement at gap=0 and
    amplify disagreement at gap=max. Pass requires both directions at p < 0.01
    (Bonferroni-corrected, 2 comparisons)."""
    max_gap = df["pref_gap"].max()
    if pd.isna(max_gap) or max_gap == 0:
        return {"test": 2, "name": "Bias instruction asymmetry",
                "p_gap0": np.nan, "p_gapmax": np.nan, "passed": "N/A"}

    # gap=0: B=2 should increase agreement (one-sided: B=2 > other)
    gap0 = df[df["pref_gap"] == 0]
    gap0_b2 = gap0[(gap0["bias1"] == 2) & (gap0["bias2"] == 2)]["agreement"].values
    gap0_other = gap0[~((gap0["bias1"] == 2) & (gap0["bias2"] == 2))]["agreement"].values

    # gap=max: B=2 should decrease agreement (one-sided: B=2 < other)
    gapmax = df[df["pref_gap"] == max_gap]
    gapmax_b2 = gapmax[(gapmax["bias1"] == 2) & (gapmax["bias2"] == 2)]["agreement"].values
    gapmax_other = gapmax[~((gapmax["bias1"] == 2) & (gapmax["bias2"] == 2))]["agreement"].values

    if len(gap0_b2) < 2 or len(gap0_other) < 2:
        p_gap0 = np.nan
    else:
        _, p_gap0 = stats.mannwhitneyu(gap0_b2, gap0_other, alternative="greater")

    if len(gapmax_b2) < 2 or len(gapmax_other) < 2:
        p_gapmax = np.nan
    else:
        _, p_gapmax = stats.mannwhitneyu(gapmax_b2, gapmax_other, alternative="less")

    # Bonferroni correction (2 comparisons)
    n_comparisons = 2
    p_gap0_corr = min(p_gap0 * n_comparisons, 1.0) if not np.isnan(p_gap0) else np.nan
    p_gapmax_corr = min(p_gapmax * n_comparisons, 1.0) if not np.isnan(p_gapmax) else np.nan

    both_testable = not np.isnan(p_gap0_corr) and not np.isnan(p_gapmax_corr)
    passed = both_testable and (p_gap0_corr < P_THRESHOLD) and (p_gapmax_corr < P_THRESHOLD)

    return {"test": 2, "name": "Bias instruction asymmetry",
            "p_gap0": p_gap0_corr, "p_gapmax": p_gapmax_corr, "passed": passed}


def _test3_sentiment_consistency(df):
    """Mann-Whitney U: (1,1) > (2,5), (3,5), (4,5). Bonferroni corrected."""
    def _get_pair(p1, p2):
        return df[
            ((df["pref1"] == p1) & (df["pref2"] == p2)) |
            ((df["pref1"] == p2) & (df["pref2"] == p1))
        ]["agreement"].values

    scores_11 = _get_pair(1, 1)
    comparison_pairs = [(2, 5), (3, 5), (4, 5)]
    n_comparisons = len(comparison_pairs)
    all_pass = True
    details = []

    for p1, p2 in comparison_pairs:
        scores_other = _get_pair(p1, p2)
        if len(scores_11) == 0 or len(scores_other) == 0:
            all_pass = False
            details.append({"pair": f"{p1}-{p2}", "p": np.nan, "passed": False})
            continue
        # One-sided: (1,1) > other
        u_stat, p = stats.mannwhitneyu(scores_11, scores_other, alternative="greater")
        corrected_p = min(p * n_comparisons, 1.0)
        pair_pass = corrected_p < P_THRESHOLD
        if not pair_pass:
            all_pass = False
        details.append({"pair": f"{p1}-{p2}", "p": corrected_p, "passed": pair_pass})

    return {"test": 3, "name": "Sentiment consistency",
            "details": details, "passed": all_pass}


def _test4_contentiousness_at_shared_pref(df):
    """At gap=0, contentiousness should not affect agreement.
    Kruskal-Wallis across C=1,2,3. Pass if p ≥ 0.01 (cannot reject)."""
    gap0 = df[df["pref_gap"] == 0]
    if gap0.empty:
        return {"test": 4, "name": "Contentiousness at shared pref",
                "h_stat": np.nan, "p": np.nan, "passed": False}

    groups = [gap0[gap0["contentiousness"] == c]["agreement"].values
              for c in sorted(gap0["contentiousness"].unique())]
    groups = [g for g in groups if len(g) >= 2]

    if len(groups) < 2:
        return {"test": 4, "name": "Contentiousness at shared pref",
                "h_stat": np.nan, "p": np.nan, "passed": False}

    h_stat, p = stats.kruskal(*groups)
    passed = p >= P_THRESHOLD  # cannot reject that distributions are the same

    return {"test": 4, "name": "Contentiousness at shared pref",
            "h_stat": h_stat, "p": p, "passed": passed}


def _test5_openness_agreement(df):
    """Pearson r(combined_openness, agreement): expect r > 0, p < 0.01."""
    r, p = stats.pearsonr(df["combined_openness"], df["agreement"])
    passed = (r > 0) and (p < P_THRESHOLD)
    return {"test": 5, "name": "Openness → higher agreement",
            "r": r, "p": p, "passed": passed}


def _test6_low_openness_high_gap(df):
    """Mann-Whitney U: (0,0) openness with (1,5) pref < other pairings. Bonferroni."""
    pref15 = df[
        ((df["pref1"] == 1) & (df["pref2"] == 5)) |
        ((df["pref1"] == 5) & (df["pref2"] == 1))
    ]
    if pref15.empty:
        return {"test": 6, "name": "Low openness + high gap → lowest",
                "passed": False}

    # Find lowest available openness pairing
    open_groups = pref15.groupby(["open1", "open2"]).size().reset_index(name="count")
    open_groups["combined"] = open_groups["open1"] + open_groups["open2"]
    min_combined = open_groups["combined"].min()
    min_row = open_groups[open_groups["combined"] == min_combined].iloc[0]
    baseline_o1, baseline_o2 = int(min_row["open1"]), int(min_row["open2"])
    baseline = pref15[(pref15["open1"] == baseline_o1) & (pref15["open2"] == baseline_o2)]["agreement"].values

    other_pairings = pref15[
        ~((pref15["open1"] == baseline_o1) & (pref15["open2"] == baseline_o2))
    ]
    unique_open = other_pairings.groupby(["open1", "open2"]).size().reset_index()

    if len(baseline) == 0 or len(unique_open) == 0:
        return {"test": 6, "name": "Low openness + high gap → lowest",
                "passed": False}

    p_values = []
    for _, row in unique_open.iterrows():
        other = pref15[
            (pref15["open1"] == row["open1"]) & (pref15["open2"] == row["open2"])
        ]["agreement"].values
        if len(other) < 2:
            continue
        # baseline should be < other (i.e., other > baseline)
        _, p = stats.mannwhitneyu(baseline, other, alternative="less")
        p_values.append(p)

    if not p_values:
        return {"test": 6, "name": "Low openness + high gap → lowest",
                "n_tests": 0, "passed": False}

    n_tests = len(p_values)
    corrected = [min(p * n_tests, 1.0) for p in p_values]
    all_pass = all(cp < P_THRESHOLD for cp in corrected)
    n_sig = sum(cp < P_THRESHOLD for cp in corrected)

    return {"test": 6, "name": "Low openness + high gap → lowest",
            "n_tests": n_tests, "n_significant": n_sig, "passed": all_pass}


def run_all_tests(df, model_id):
    """Run all 6 tests for a single model."""
    mdf = df[df["model_id"] == model_id]
    if mdf.empty:
        return []
    results = [
        _test1_pref_gap_agreement(mdf),
        _test2_bias_asymmetry(mdf),
        _test3_sentiment_consistency(mdf),
        _test4_contentiousness_at_shared_pref(mdf),
        _test5_openness_agreement(mdf),
        _test6_low_openness_high_gap(mdf),
    ]
    for r in results:
        r["model_id"] = model_id
    return results


# ---------------------------------------------------------------------------
# Step 5: Results table
# ---------------------------------------------------------------------------
def generate_results_table(df, tables_dir):
    """Run tests for all models and output CSV + LaTeX table."""
    ensure_dir(tables_dir)
    models = load_model_registry()
    model_ids = sorted([int(m["id"]) for m in models if int(m["id"]) not in EXCLUDE_MODELS])

    # Build model info lookup
    model_info = {}
    for m in models:
        mid = int(m["id"])
        model_info[mid] = {
            "hf_name": m["hf_name"],
            "family": m["family"],
            "size_b": m.get("size_b"),
            "short_name": m["hf_name"].split("/")[-1],
        }

    all_results = []
    summary_rows = []

    for mid in model_ids:
        if mid not in df["model_id"].unique():
            continue
        results = run_all_tests(df, mid)
        all_results.extend(results)
        row = {
            "model_id": mid,
            "model_name": model_info[mid]["short_name"],
            "family": model_info[mid]["family"],
            "size_b": model_info[mid]["size_b"],
        }
        for r in results:
            row[f"test{r['test']}"] = r["passed"]
        row["n_passed"] = sum(1 for r in results if r["passed"] == True and r["passed"] != "N/A")
        summary_rows.append(row)

    # CSV
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(tables_dir, "test_results.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved test_results.csv ({len(summary_rows)} models)")

    # LaTeX
    latex_path = os.path.join(tables_dir, "test_results.tex")
    _write_latex_table(summary_rows, model_info, latex_path)
    print(f"  Saved test_results.tex")

    return summary_df


def _write_latex_table(summary_rows, model_info, path):
    """Write LaTeX table matching tab:size-hyp-6 format."""
    families = {
        "qwen3": "Qwen3",
        "llama3.2": "Llama-3.x",
        "llama3.1": "Llama-3.x",
        "gemma3": "Gemma-3",
    }

    lines = []
    lines.append(r"\begin{table}[t!]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabularx}{1.0\textwidth}{cl|Y|Y|Y|Y|Y|Y}")
    lines.append(r"\toprule")
    lines.append(r"& & \multicolumn{4}{c}{Preference} & \multicolumn{2}{c}{Openness} \\")
    lines.append(r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}")
    lines.append(r"& & \multicolumn{1}{c}{Surface-level} & \multicolumn{3}{c}{In-depth} & \multicolumn{1}{c}{Surface-level} & \multicolumn{1}{c}{In-depth} \\")
    lines.append(r"\cmidrule(r){3-3}\cmidrule(lr){4-6}\cmidrule(l){7-7}\cmidrule(lr){8-8}")
    lines.append(r"& & Test 1 & Test 2 & Test 3 & Test 4 & Test 5 & Test 6 \\")
    lines.append(r"\midrule")

    # Group by family
    grouped = {}
    for row in summary_rows:
        fam = families.get(row["family"], row["family"])
        grouped.setdefault(fam, []).append(row)

    for fam_name in ["Qwen3", "Llama-3.x", "Gemma-3"]:
        members = grouped.get(fam_name, [])
        if not members:
            continue
        n = len(members)
        rot = r"\parbox[t]{2mm}{\multirow{" + str(n) + r"}{*}{\rotatebox[origin=c]{90}{\small{" + fam_name + r"}}}}"
        for i, row in enumerate(sorted(members, key=lambda r: r.get("size_b") or 0)):
            prefix = rot if i == 0 else ""
            marks = []
            for t in range(1, 7):
                val = row.get(f"test{t}", False)
                if val == "N/A":
                    marks.append("N/A")
                elif val:
                    marks.append(r"\checkmark")
                else:
                    marks.append(r"\times")
            model_label = row["model_name"]
            lines.append(f"{prefix} &{model_label} & {' & '.join(marks)} \\\\")
        lines.append(r"\midrule")

    lines.append(r"\end{tabularx}")
    lines.append(r"\caption{Significance testing results for each model across six tests. "
                 r"A ($\checkmark$) indicates the model passed the test, while ($\times$) indicates failure.}")
    lines.append(r"\label{tab:size-hyp-6}")
    lines.append(r"\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Step 6: Additional analyses
# ---------------------------------------------------------------------------
def fig_scaling_analysis(df, output_dir, formats):
    """Scaling analysis: tests passed vs model size, plus effect sizes."""
    models = load_model_registry()
    model_info = {}
    for m in models:
        mid = int(m["id"])
        if mid not in EXCLUDE_MODELS:
            model_info[mid] = {
                "size_b": m.get("size_b", 0),
                "short_name": m["hf_name"].split("/")[-1],
                "family": m["family"],
            }

    model_ids = sorted(model_info.keys())
    results_by_model = {}
    for mid in model_ids:
        if mid not in df["model_id"].unique():
            continue
        results_by_model[mid] = run_all_tests(df, mid)

    if not results_by_model:
        print("  WARNING: No models with data for scaling analysis")
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    # Panel 1: Total tests passed vs size
    ax = axes[0, 0]
    sizes = [model_info[mid]["size_b"] for mid in results_by_model]
    n_passed = [sum(1 for r in results_by_model[mid] if r["passed"] == True and r["passed"] != "N/A") for mid in results_by_model]
    family_colors = {"qwen3": "steelblue", "llama3.2": "forestgreen", "llama3.1": "forestgreen", "gemma3": "firebrick"}

    for mid, sz, np_ in zip(results_by_model.keys(), sizes, n_passed):
        fam = model_info[mid]["family"]
        ax.scatter(sz, np_, color=family_colors.get(fam, "gray"), s=80, zorder=3,
                   label=model_info[mid]["short_name"])
    ax.set_xscale("log")
    ax.set_xlabel("Model Size (B)")
    ax.set_ylabel("Tests Passed (of 6)")
    ax.set_ylim(-0.5, 6.5)
    ax.set_title("Tests Passed vs Size")
    ax.legend(fontsize=6, loc="upper left")

    # Panels 2-7: Effect size per test
    test_names = [
        "Pref gap corr (r)", "Bias asymmetry", "Sentiment",
        "Content. shared pref", "Openness corr (r)", "Low open+high gap",
    ]
    for t_idx in range(6):
        row, col = divmod(t_idx + 1, 4)
        if t_idx + 1 >= 4:
            row, col = 1, (t_idx + 1) - 4
        else:
            row, col = 0, t_idx + 1
        ax = axes[row, col]

        for mid in results_by_model:
            res = results_by_model[mid][t_idx]
            sz = model_info[mid]["size_b"]
            fam = model_info[mid]["family"]

            # Extract effect size where available
            if "r" in res:
                effect = abs(res["r"])
            elif "h_stat" in res:
                effect = res.get("h_stat", 0)
            elif "p_gap0" in res and "p_gapmax" in res:
                # Bias asymmetry: use -log10 of worse (larger) p as effect
                p_vals = [v for v in [res["p_gap0"], res["p_gapmax"]] if not np.isnan(v)]
                worst_p = max(p_vals) if p_vals else 1.0
                effect = -np.log10(worst_p) if worst_p > 0 else 10.0
            elif "n_significant" in res and "n_tests" in res and res["n_tests"] > 0:
                effect = res["n_significant"] / res["n_tests"]
            else:
                effect = 0

            marker = "^" if res["passed"] else "v"
            ax.scatter(sz, effect, color=family_colors.get(fam, "gray"), s=60,
                       marker=marker, zorder=3)

        ax.set_xscale("log")
        ax.set_xlabel("Size (B)")
        ax.set_ylabel("Effect Size")
        ax.set_title(f"Test {t_idx + 1}: {test_names[t_idx]}")

    # Hide last unused panel if needed
    if len(axes.flat) > 8:
        for extra_ax in axes.flat[8:]:
            extra_ax.set_visible(False)
    axes[1, 3].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, output_dir, "scaling_analysis", formats)


def fig_per_topic_breakdown(df, focal_model, output_dir, formats):
    """9-panel figure: pref_gap vs agreement per topic for focal model."""
    fdf = df[df["model_id"] == focal_model]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharey=True)

    for topic_id in range(9):
        row, col = divmod(topic_id, 3)
        ax = axes[row, col]
        tdf = fdf[fdf["topic_id"] == topic_id]
        gaps = sorted(tdf["pref_gap"].unique()) if not tdf.empty else list(range(5))
        means, los, his = [], [], []
        for g in gaps:
            vals = tdf[tdf["pref_gap"] == g]["agreement"].values
            m, lo, hi = bootstrap_mean_ci(vals)
            means.append(m)
            los.append(m - lo if not np.isnan(m) else 0)
            his.append(hi - m if not np.isnan(m) else 0)

        ax.bar(gaps, means, yerr=[los, his], capsize=3, color="steelblue",
               edgecolor="black", width=0.6)
        ax.set_title(TOPIC_NAMES[topic_id].replace("_", " ").title(), fontsize=10)
        ax.set_xticks(gaps)
        ax.set_ylim(0, 6)
        if col == 0:
            ax.set_ylabel("Mean Agreement")
        if row == 2:
            ax.set_xlabel("Preference Gap")

    fig.suptitle(f"Per-Topic Breakdown (Model {focal_model})", y=1.01)
    fig.tight_layout()
    _save_fig(fig, output_dir, "per_topic_breakdown", formats)


def fig_agreement_distribution(df, output_dir, formats):
    """Agreement score distribution per model."""
    models = load_model_registry()
    model_info = {int(m["id"]): m["hf_name"].split("/")[-1] for m in models if int(m["id"]) not in EXCLUDE_MODELS}
    model_ids = sorted([mid for mid in model_info if mid in df["model_id"].unique()])

    n_models = len(model_ids)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for idx, mid in enumerate(model_ids):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        scores = df[df["model_id"] == mid]["agreement"].values
        ax.hist(scores, bins=np.arange(0.5, 6.5, 1), color="steelblue", edgecolor="black",
                density=True)
        mean_val = np.mean(scores)
        mode_val = float(stats.mode(scores, keepdims=False).mode)
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean={mean_val:.2f}")
        ax.axvline(mode_val, color="orange", linestyle=":", label=f"Mode={mode_val:.0f}")
        ax.set_title(model_info[mid], fontsize=9)
        ax.set_xlabel("Agreement Score")
        ax.set_ylabel("Density")
        ax.set_xticks(range(1, 6))
        ax.legend(fontsize=7)

    # Hide unused panels
    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, output_dir, "agreement_distribution", formats)


def generate_extras(df, focal_model, output_dir, formats):
    """Generate the three additional analysis figures."""
    ensure_dir(output_dir)
    fig_scaling_analysis(df, output_dir, formats)
    fig_per_topic_breakdown(df, focal_model, output_dir, formats)
    fig_agreement_distribution(df, output_dir, formats)


# ---------------------------------------------------------------------------
# Step 7: CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Reference-paper comparison analysis for LLM Behavioral Coherence.")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Artifact root (default: /lustre/fs0/scratch/jmooney/latent_profile)")
    parser.add_argument("--experiment-name", type=str, default="core_fresh",
                        help="Experiment subdirectory under results/")
    parser.add_argument("--focal-model", type=int, default=FOCAL_MODEL,
                        help="Model ID for qualitative figures")
    parser.add_argument("--fig-format", nargs="+", default=["png", "pdf"],
                        help="Output figure formats")
    parser.add_argument(
        "--target-figures",
        nargs="+",
        default=list(QUAL_FIGURE_NAMES),
        choices=QUAL_FIGURE_NAMES,
        help="Subset of qualitative figures to generate",
    )
    parser.add_argument(
        "--figure-format-map",
        nargs="*",
        default=[],
        help="Per-figure format override as figure=fmt1,fmt2 (e.g., result_pref_base=pdf)",
    )
    parser.add_argument(
        "--no-figure-cache",
        action="store_true",
        help="Disable figure cache and force render of selected figures",
    )
    parser.add_argument("--output-dir", type=str, default="analysis_outputs/paper_reference",
                        help="Base output directory for derived figures and tables")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "figures", "tests", "table", "extras"],
                        help="Which analyses to run")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force rebuild of cached dataframe")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    tables_dir = os.path.join(args.output_dir, "tables")
    cache_dir = os.path.join(args.output_dir, ".cache")

    if args.rebuild_cache:
        cache_path = os.path.join(cache_dir, "master_df.parquet")
        if os.path.exists(cache_path):
            os.remove(cache_path)
        fig_cache_path = _figure_cache_manifest_path(cache_dir)
        if os.path.exists(fig_cache_path):
            os.remove(fig_cache_path)

    df = get_or_build_dataframe(args.output_root, args.experiment_name, cache_dir)
    print(f"Master dataframe: {len(df)} rows, {df['model_id'].nunique()} models")

    try:
        figure_format_map = _parse_figure_format_map(args.figure_format_map)
    except ValueError as exc:
        parser.error(str(exc))

    if args.mode in ("all", "figures"):
        generate_all_figures(
            df,
            args.focal_model,
            fig_dir,
            args.fig_format,
            target_figures=args.target_figures,
            figure_format_map=figure_format_map,
            figure_cache_dir=cache_dir,
            use_figure_cache=not args.no_figure_cache,
        )

    if args.mode in ("all", "tests", "table"):
        generate_results_table(df, tables_dir)

    if args.mode in ("all", "tests"):
        print("\n--- Test Results (focal model) ---")
        results = run_all_tests(df, args.focal_model)
        for r in results:
            mark = "PASS" if r["passed"] else "FAIL"
            print(f"  Test {r['test']}: {mark} — {r['name']}")

    if args.mode in ("all", "extras"):
        generate_extras(df, args.focal_model, fig_dir, args.fig_format)

    print("\nDone.")


if __name__ == "__main__":
    main()
