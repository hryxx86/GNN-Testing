# Phase D+: 增强诊断实验设计

## 目标
在做任何GNN改进之前，建立信号水位线：
1. 信号在哪里？（分桶诊断）
2. 非GNN方法能到什么水平？（baseline矩阵）
3. LLM特征上限是多少？（小样本验证）

## Notebook 结构: `phase_d_diagnosis.ipynb`

---

### Cell 1 [Markdown]: Experiment Purpose

```markdown
# Phase D+: Signal Diagnosis & Baseline Establishment
**Hypothesis**: Signal exists but is sparse — most events are unpredictable noise,
but a subset with strong sentiment/momentum carries exploitable signal.
**Key Question**: Does selective filtering improve AUC for ANY method, not just GNN?
```

### Cell 2 [Code]: Imports & Setup

```python
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Paths — adjust to your actual project structure
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("docs")
OUTPUT_DIR.mkdir(exist_ok=True)
```

### Cell 3b [Code]: News Deduplication (before any analysis)

```python
def deduplicate_news(df, method='same_day_ticker'):
    """
    Aggregate duplicate/rewritten news for the same stock on the same day.
    Financial newswires frequently republish/rewrite the same story.

    Method: For same (date, ticker), keep one row with:
    - FinBERT embedding: mean of all events
    - Sentiment: max absolute sentiment (strongest signal)
    - Headline: keep the one with highest sentiment confidence
    - Label: same (it's the same stock-day return)
    """
    original_count = len(df)

    # Group by (date, ticker) and aggregate
    agg_funcs = {
        'finbert_pos': 'mean',
        'finbert_neg': 'mean',
        'finbert_neu': 'mean',
        'label': 'first',       # Same stock-day → same label
        'return_pct': 'first',  # Same stock-day → same return
        'vix': 'first',
        'sector': 'first',
        'event_id': 'count',    # Track how many events were merged
    }
    # Add momentum/volatility cols if present
    for col in df.columns:
        if 'momentum' in col or 'volatility' in col:
            agg_funcs[col] = 'first'

    deduped = df.groupby(['date', 'ticker']).agg(agg_funcs).reset_index()
    deduped = deduped.rename(columns={'event_id': 'n_events_merged'})

    deduped_count = len(deduped)
    print(f"News deduplication: {original_count:,} events → {deduped_count:,} stock-days")
    print(f"  Reduction: {(1 - deduped_count/original_count)*100:.1f}%")
    print(f"  Median events per stock-day: {deduped['n_events_merged'].median():.0f}")
    print(f"  Max events per stock-day: {deduped['n_events_merged'].max()}")

    return deduped

# df_deduped = deduplicate_news(df)
# Run ALL subsequent analyses on df_deduped
# Also report AUC before vs after dedup to quantify impact
```

### Cell 3 [Code]: PARAMETERS (Sacred Cell)

```python
PARAMS = {
    # Data splits (walk-forward, with dev-holdout for LLM validation)
    "fold_1_train_end": "2023-09-30",      # Train ends Sep 2023
    "dev_holdout_start": "2023-10-01",      # Dev-holdout: Oct-Dec 2023
    "dev_holdout_end": "2023-12-31",        # (used for D.3 LLM validation ONLY)
    "fold_1_test_start": "2024-01-01",
    "fold_1_test_end": "2024-12-31",
    "fold_2_train_end": "2024-12-31",       # Fold 2 train includes 2024
    "fold_2_test_start": "2025-01-01",
    "fold_2_test_end": "2025-12-31",        # LOCKBOX — never touch until final report

    # Label noise threshold
    "noise_threshold_pct": 0.5,  # |return| < 0.5% considered noise

    # Selective prediction coverage levels
    "coverage_levels": [0.05, 0.10, 0.20, 0.50, 1.00],

    # Diagnosis buckets
    "return_buckets": [0, 0.005, 0.01, 0.02, float('inf')],
    "return_bucket_labels": ["<0.5%", "0.5-1%", "1-2%", ">2%"],
    "vix_buckets": [0, 15, 25, float('inf')],
    "vix_bucket_labels": ["Low (<15)", "Medium (15-25)", "High (>25)"],

    # LLM small sample
    "llm_sample_frac": 0.10,

    # Baselines
    "xgb_params": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": SEED,
        "eval_metric": "auc",
    },
}
```

### Cell 4 [Code]: Data Loading

```python
def load_data():
    """
    Load preprocessed event data with features and labels.
    Expected columns:
    - event_id, date, ticker, sector
    - finbert_pos, finbert_neg, finbert_neu (FinBERT sentiment)
    - finbert_confidence (max of pos/neg)
    - label (1=up, 0=down)
    - return_pct (actual return, for diagnosis only — NOT a feature)
    - vix (VIX level on event date)
    - momentum_5d, momentum_10d, momentum_21d (rolling returns)
    - volatility_5d, volatility_10d, volatility_21d (rolling std)
    """
    # TODO: Replace with actual data loading from your pipeline
    # df = pd.read_parquet(DATA_DIR / "events_with_features.parquet")

    # Placeholder structure for development
    print("TODO: Load actual data. Expected shape: ~1.7M rows")
    print("Required columns: date, ticker, sector, finbert_*, label, return_pct, vix, momentum_*, volatility_*")
    return None

def temporal_split(df, train_end, test_start, test_end):
    """Strict temporal split — no future leakage."""
    train = df[df['date'] <= train_end].copy()
    test = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
    return train, test

df = load_data()
```

### Cell 5 [Code]: D.1 — Label Noise Analysis

```python
def label_noise_analysis(df):
    """Quantify how many events are in the 'noise zone' (tiny returns)."""
    abs_return = df['return_pct'].abs()
    noise_mask = abs_return < PARAMS['noise_threshold_pct']

    stats = {
        "total_events": len(df),
        "noise_events": noise_mask.sum(),
        "noise_pct": noise_mask.mean() * 100,
        "median_abs_return": abs_return.median(),
        "mean_abs_return": abs_return.mean(),
    }

    # Distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: return distribution with noise zone
    axes[0].hist(df['return_pct'].clip(-5, 5), bins=100, alpha=0.7, edgecolor='black')
    axes[0].axvline(-PARAMS['noise_threshold_pct'], color='red', linestyle='--', label=f'Noise zone (±{PARAMS["noise_threshold_pct"]}%)')
    axes[0].axvline(PARAMS['noise_threshold_pct'], color='red', linestyle='--')
    axes[0].set_xlabel('Return (%)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Return Distribution — {stats["noise_pct"]:.1f}% in noise zone')
    axes[0].legend()

    # Right: AUC by |return| bucket
    df_copy = df.copy()
    df_copy['abs_return_bucket'] = pd.cut(
        abs_return,
        bins=PARAMS['return_buckets'],
        labels=PARAMS['return_bucket_labels']
    )
    bucket_aucs = []
    for bucket in PARAMS['return_bucket_labels']:
        mask = df_copy['abs_return_bucket'] == bucket
        subset = df_copy[mask]
        if len(subset) > 100 and subset['label'].nunique() == 2:
            # Use FinBERT sentiment as a simple predictor
            auc = roc_auc_score(subset['label'], subset['finbert_pos'] - subset['finbert_neg'])
            bucket_aucs.append({'bucket': bucket, 'AUC': auc, 'n': len(subset)})

    bucket_df = pd.DataFrame(bucket_aucs)
    axes[1].bar(bucket_df['bucket'], bucket_df['AUC'], color='steelblue', edgecolor='black')
    axes[1].axhline(0.5, color='red', linestyle='--', label='Random')
    axes[1].set_ylabel('AUC (FinBERT sentiment)')
    axes[1].set_title('AUC by |Return| Magnitude')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_label_noise.png', dpi=150)
    plt.show()

    return stats

# noise_stats = label_noise_analysis(df)
```

### Cell 6 [Code]: D.2 — Non-GNN Baseline Matrix

```python
def compute_selective_auc(y_true, y_pred_proba, coverages):
    """
    Compute AUC at different coverage levels.
    Selection criterion: |predicted_prob - 0.5| (confidence ranking).
    """
    confidence = np.abs(y_pred_proba - 0.5)
    results = {}
    for cov in coverages:
        if cov >= 1.0:
            results[f"AUC@{int(cov*100)}%"] = roc_auc_score(y_true, y_pred_proba)
            continue
        k = int(len(y_true) * cov)
        if k < 50:
            results[f"AUC@{int(cov*100)}%"] = np.nan
            continue
        top_idx = np.argsort(confidence)[-k:]  # highest confidence
        results[f"AUC@{int(cov*100)}%"] = roc_auc_score(y_true[top_idx], y_pred_proba[top_idx])
    return results


def run_baseline_matrix(train_df, test_df, fold_name="Fold"):
    """Run all non-GNN baselines and compute full + selective AUC."""

    results = []
    coverages = PARAMS['coverage_levels']

    # --- Feature preparation ---
    sentiment_cols = ['finbert_pos', 'finbert_neg', 'finbert_neu']
    momentum_cols = [c for c in train_df.columns if 'momentum' in c or 'volatility' in c]
    all_feature_cols = sentiment_cols + momentum_cols

    X_train_sent = train_df[sentiment_cols].values
    X_test_sent = test_df[sentiment_cols].values
    X_train_all = train_df[all_feature_cols].fillna(0).values
    X_test_all = test_df[all_feature_cols].fillna(0).values
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    # --- 1. Random baseline ---
    random_preds = np.random.RandomState(SEED).rand(len(y_test))
    r = compute_selective_auc(y_test, random_preds, coverages)
    r['Baseline'] = 'Random'
    results.append(r)

    # --- 2. Rule-based (sentiment > 0 → up) ---
    rule_preds = (test_df['finbert_pos'] - test_df['finbert_neg']).values
    rule_preds = 1 / (1 + np.exp(-rule_preds))  # sigmoid to [0,1]
    r = compute_selective_auc(y_test, rule_preds, coverages)
    r['Baseline'] = 'Rule-based (sentiment)'
    results.append(r)

    # --- 3. Logistic Regression (sentiment only) ---
    lr = LogisticRegression(random_state=SEED, max_iter=1000)
    lr.fit(X_train_sent, y_train)
    lr_preds = lr.predict_proba(X_test_sent)[:, 1]
    r = compute_selective_auc(y_test, lr_preds, coverages)
    r['Baseline'] = 'LR (sentiment)'
    results.append(r)

    # --- 4. XGBoost (sentiment + momentum) ---
    xgb = XGBClassifier(**PARAMS['xgb_params'])
    xgb.fit(X_train_all, y_train, verbose=False)
    xgb_preds = xgb.predict_proba(X_test_all)[:, 1]
    r = compute_selective_auc(y_test, xgb_preds, coverages)
    r['Baseline'] = 'XGBoost (sent + momentum)'
    results.append(r)

    # --- 5. GNN v1 (load saved predictions) ---
    # TODO: Load Phase C predictions
    # gnn_preds = np.load("results/phase_c_predictions.npy")
    # r = compute_selective_auc(y_test, gnn_preds, coverages)
    # r['Baseline'] = 'GNN v1 (Phase C)'
    # results.append(r)

    # --- Format results ---
    results_df = pd.DataFrame(results)
    cols = ['Baseline'] + [c for c in results_df.columns if c != 'Baseline']
    results_df = results_df[cols]

    print(f"\n{'='*60}")
    print(f"  Baseline Matrix — {fold_name}")
    print(f"{'='*60}")
    print(results_df.to_string(index=False, float_format='{:.4f}'.format))

    return results_df


def run_walk_forward_baselines(df):
    """Run baselines across both walk-forward folds."""
    fold_results = []

    for fold_num, (train_end, test_start, test_end) in enumerate([
        (PARAMS['fold_1_train_end'], PARAMS['fold_1_test_start'], PARAMS['fold_1_test_end']),
        (PARAMS['fold_2_train_end'], PARAMS['fold_2_test_start'], PARAMS['fold_2_test_end']),
    ], 1):
        train, test = temporal_split(df, train_end, test_start, test_end)
        print(f"\nFold {fold_num}: Train={len(train):,} events, Test={len(test):,} events")
        result = run_baseline_matrix(train, test, fold_name=f"Fold {fold_num}")
        result['Fold'] = fold_num
        fold_results.append(result)

    # Aggregate: mean ± std across folds
    combined = pd.concat(fold_results)
    auc_cols = [c for c in combined.columns if 'AUC' in c]
    summary = combined.groupby('Baseline')[auc_cols].agg(['mean', 'std']).round(4)
    print(f"\n{'='*60}")
    print("  Walk-Forward Summary (mean ± std)")
    print(f"{'='*60}")
    print(summary)

    return combined, summary

# combined_results, summary = run_walk_forward_baselines(df)
```

### Cell 7 [Code]: D.3 — LLM Small Sample Verification (on Dev-Holdout)

```python
# === LLM API call framework ===
# CRITICAL: Use dev-holdout (2023-Q4), NOT test set, to avoid leakage.
# The test set (2024/2025) must remain untouched until final reporting.

LLM_PROMPT_TEMPLATE = """Analyze this financial news headline for stock {ticker}.

News: "{headline}"

Respond ONLY with a JSON object (no other text):
{{
  "impact_level": "high" or "medium" or "low",
  "direction": "positive" or "negative" or "neutral",
  "confidence": <float 0.0-1.0>,
  "reasoning_type": "earnings" or "macro" or "sentiment" or "technical" or "other"
}}"""


def encode_llm_features(llm_response: dict) -> np.ndarray:
    """
    Convert LLM structured output to feature vector.
    Returns 7-dim vector:
    - impact: 3-dim one-hot (high/medium/low)
    - direction: encoded as scalar (-1/0/1)
    - confidence: scalar [0,1]
    - reasoning_type: 5-dim one-hot
    Total: 3 + 1 + 1 + 5 = 10 dims (vs FinBERT's 3 dims)
    """
    impact_map = {"high": [1,0,0], "medium": [0,1,0], "low": [0,0,1]}
    direction_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    reasoning_map = {
        "earnings": [1,0,0,0,0], "macro": [0,1,0,0,0],
        "sentiment": [0,0,1,0,0], "technical": [0,0,0,1,0], "other": [0,0,0,0,1]
    }

    features = (
        impact_map.get(llm_response.get("impact_level", "low"), [0,0,1])
        + [direction_map.get(llm_response.get("direction", "neutral"), 0.0)]
        + [float(llm_response.get("confidence", 0.5))]
        + reasoning_map.get(llm_response.get("reasoning_type", "other"), [0,0,0,0,1])
    )
    return np.array(features)


def compare_finbert_vs_llm(test_df, llm_features):
    """
    Compare FinBERT (3-dim) vs LLM (10-dim) as features for LR/XGBoost.
    llm_features: np.ndarray of shape (n_llm_samples, 10)
    """
    # Subsample test_df to match LLM sample
    sample_idx = test_df.index[:len(llm_features)]
    sample_df = test_df.loc[sample_idx]
    y = sample_df['label'].values

    # Need a train set for the LR/XGB — use remaining test data or a held-out portion
    # For quick validation, do 50/50 split within the LLM sample
    mid = len(y) // 2

    results = {}
    for feat_name, X in [
        ("FinBERT 3-dim", sample_df[['finbert_pos','finbert_neg','finbert_neu']].values),
        ("LLM 10-dim", llm_features),
    ]:
        lr = LogisticRegression(random_state=SEED, max_iter=500)
        lr.fit(X[:mid], y[:mid])
        preds = lr.predict_proba(X[mid:])[:, 1]
        auc = roc_auc_score(y[mid:], preds)
        results[feat_name] = auc
        print(f"  {feat_name}: AUC = {auc:.4f}")

    return results

# After running LLM API calls and caching results:
# llm_features = np.array([encode_llm_features(r) for r in llm_responses])
# compare_finbert_vs_llm(test_df, llm_features)
```

### Cell 8 [Code]: D.4 — Conditional Bucket Diagnosis

```python
def bucket_diagnosis(df, pred_col='finbert_sentiment_score'):
    """
    Compute AUC across different conditioning variables.
    pred_col: column to use as prediction (e.g., finbert_pos - finbert_neg).
    """
    if pred_col not in df.columns:
        df = df.copy()
        df[pred_col] = df['finbert_pos'] - df['finbert_neg']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- A. By sentiment confidence ---
    ax = axes[0, 0]
    df['sent_conf'] = df[['finbert_pos', 'finbert_neg']].max(axis=1)
    df['sent_conf_bucket'] = pd.qcut(df['sent_conf'], q=4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
    aucs = []
    for bucket in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
        subset = df[df['sent_conf_bucket'] == bucket]
        if len(subset) > 100 and subset['label'].nunique() == 2:
            auc = roc_auc_score(subset['label'], subset[pred_col])
            aucs.append({'Bucket': bucket, 'AUC': auc, 'N': len(subset)})
    auc_df = pd.DataFrame(aucs)
    ax.bar(auc_df['Bucket'], auc_df['AUC'], color='steelblue', edgecolor='black')
    ax.axhline(0.5, color='red', linestyle='--')
    ax.set_title('AUC by Sentiment Confidence Quartile')
    ax.set_ylabel('AUC')

    # --- B. By sector ---
    ax = axes[0, 1]
    sector_aucs = []
    for sector in sorted(df['sector'].unique()):
        subset = df[df['sector'] == sector]
        if len(subset) > 200 and subset['label'].nunique() == 2:
            auc = roc_auc_score(subset['label'], subset[pred_col])
            sector_aucs.append({'Sector': sector, 'AUC': auc, 'N': len(subset)})
    sector_df = pd.DataFrame(sector_aucs).sort_values('AUC', ascending=False)
    ax.barh(sector_df['Sector'], sector_df['AUC'], color='darkorange', edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--')
    ax.set_title('AUC by GICS Sector')
    ax.set_xlabel('AUC')

    # --- C. By |return| magnitude ---
    ax = axes[1, 0]
    df['return_bucket'] = pd.cut(
        df['return_pct'].abs(),
        bins=PARAMS['return_buckets'],
        labels=PARAMS['return_bucket_labels']
    )
    ret_aucs = []
    for bucket in PARAMS['return_bucket_labels']:
        subset = df[df['return_bucket'] == bucket]
        if len(subset) > 100 and subset['label'].nunique() == 2:
            auc = roc_auc_score(subset['label'], subset[pred_col])
            ret_aucs.append({'Bucket': bucket, 'AUC': auc, 'N': len(subset)})
    ret_df = pd.DataFrame(ret_aucs)
    ax.bar(ret_df['Bucket'], ret_df['AUC'], color='seagreen', edgecolor='black')
    ax.axhline(0.5, color='red', linestyle='--')
    ax.set_title('AUC by |Return| Magnitude (Oracle — diagnosis only)')
    ax.set_ylabel('AUC')

    # --- D. By VIX regime ---
    ax = axes[1, 1]
    df['vix_bucket'] = pd.cut(
        df['vix'],
        bins=PARAMS['vix_buckets'],
        labels=PARAMS['vix_bucket_labels']
    )
    vix_aucs = []
    for bucket in PARAMS['vix_bucket_labels']:
        subset = df[df['vix_bucket'] == bucket]
        if len(subset) > 100 and subset['label'].nunique() == 2:
            auc = roc_auc_score(subset['label'], subset[pred_col])
            vix_aucs.append({'Regime': bucket, 'AUC': auc, 'N': len(subset)})
    vix_df = pd.DataFrame(vix_aucs)
    ax.bar(vix_df['Regime'], vix_df['AUC'], color='crimson', edgecolor='black')
    ax.axhline(0.5, color='gray', linestyle='--')
    ax.set_title('AUC by VIX Regime')
    ax.set_ylabel('AUC')

    plt.suptitle('Phase D+: Signal Diagnosis by Condition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_bucket_diagnosis.png', dpi=150)
    plt.show()

# bucket_diagnosis(test_df)
```

### Cell 9 [Code]: D.4 bonus — Sector × Confidence Heatmap

```python
def sector_confidence_heatmap(df, pred_col='finbert_sentiment_score'):
    """
    Heatmap of AUC across sector × sentiment confidence.
    This directly becomes Figure 3 in the paper.
    """
    if pred_col not in df.columns:
        df = df.copy()
        df[pred_col] = df['finbert_pos'] - df['finbert_neg']

    df['sent_conf_q'] = pd.qcut(
        df[['finbert_pos', 'finbert_neg']].max(axis=1),
        q=3, labels=['Low', 'Medium', 'High']
    )

    sectors = sorted(df['sector'].unique())
    conf_levels = ['Low', 'Medium', 'High']

    heatmap_data = pd.DataFrame(index=sectors, columns=conf_levels, dtype=float)
    count_data = pd.DataFrame(index=sectors, columns=conf_levels, dtype=int)

    for sector in sectors:
        for conf in conf_levels:
            subset = df[(df['sector'] == sector) & (df['sent_conf_q'] == conf)]
            count_data.loc[sector, conf] = len(subset)
            if len(subset) > 100 and subset['label'].nunique() == 2:
                heatmap_data.loc[sector, conf] = roc_auc_score(
                    subset['label'], subset[pred_col]
                )
            else:
                heatmap_data.loc[sector, conf] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heatmap_data.astype(float), annot=True, fmt='.3f',
        cmap='RdYlGn', center=0.5, vmin=0.45, vmax=0.58,
        linewidths=0.5, ax=ax
    )
    ax.set_title('AUC Heatmap: Sector × Sentiment Confidence')
    ax.set_xlabel('Sentiment Confidence')
    ax.set_ylabel('GICS Sector')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_sector_confidence_heatmap.png', dpi=150)
    plt.show()

    return heatmap_data

# heatmap = sector_confidence_heatmap(test_df)
```

### Cell 10 [Markdown]: Observations & Decision Template

```markdown
## Phase D+ Observations

### D.1 Label Noise
- Noise events (|return| < 0.5%): XX.X%
- AUC in noise zone: X.XXX (expect ~0.50)
- AUC for |return| > 2%: X.XXX (expect > 0.50 if signal exists)

### D.2 Baseline Matrix
| Baseline | Full AUC | Sel@10% | Sel@5% |
|----------|----------|---------|--------|
| Random   | 0.500    | 0.50X   | 0.50X  |
| Rule     | 0.XXX    | 0.XXX   | 0.XXX  |
| LR       | 0.XXX    | 0.XXX   | 0.XXX  |
| XGBoost  | 0.XXX    | 0.XXX   | 0.XXX  |
| GNN v1   | 0.507    | 0.XXX   | 0.XXX  |

### D.3 LLM vs FinBERT (10% sample)
- FinBERT → LR AUC: X.XXX
- LLM 10-dim → LR AUC: X.XXX
- Delta: +X.XXX → [LLM worth pursuing / marginal / not worth it]

### D.4 Bucket Diagnosis
- Strongest sector: XXXX (AUC = X.XXX)
- Weakest sector: XXXX (AUC = X.XXX)
- High VIX regime AUC: X.XXX vs Low VIX: X.XXX
- High sentiment confidence AUC: X.XXX vs Low: X.XXX

### Key Decision
Based on D+ results:
- [ ] Signal exists in high-confidence subset → proceed with selective prediction (Phase F)
- [ ] XGBoost >= GNN → fix graph structure first (Phase E priority)
- [ ] LLM >> FinBERT → adjust feature base before GNN training
- [ ] No signal anywhere → fundamental rethink needed

### Go/Pivot/Stop Criteria (hard thresholds)

| Judgment | Condition | Action |
|----------|-----------|--------|
| **GO** | Any baseline in any conditional bucket: AUC > 0.52 | Proceed E → F as planned |
| **PIVOT** | Full AUC ≈ 0.50 but |return| > 2% subset AUC > 0.54 | Narrow scope to high-impact events only |
| **STOP** | All conditions, all baselines ≈ 0.50 | Redefine problem or write negative result paper (EMH evidence) |

**Date of decision: ________**
**Decision: GO / PIVOT / STOP**
**Evidence: ________**
```

---

## 并行测试效率建议

1.7M事件上跑LR/XGBoost很快（分钟级）。关键瓶颈是GNN predictions的加载。

建议执行顺序:
1. 先跑 Cell 5 (标签噪声) — 5分钟
2. 跑 Cell 6 (baseline矩阵) — 15分钟 (含XGBoost训练)
3. 跑 Cell 8-9 (诊断图表) — 10分钟
4. 根据1-3的结果决定是否值得花$3-5跑LLM (Cell 7)
5. 填写 Cell 10 的决策模板，更新 progress.md

总耗时: ~30分钟（不含LLM API调用）
