# Phase F.2: Volatility-Calibrated SelectiveNet — 设计文档

## 动机

标准 SelectiveNet (ICML'19) 假设数据分布平稳: selection head 学习 g(h) → [0,1]。
但金融市场的 P(y|x) 在不同 regime 下完全不同:
- 低波动期: 信号相对稳定，模型可以更大胆
- 高波动期: 噪声激增，模型应更保守

**核心创新**: Selection head 不仅看 GNN hidden state，还看 market regime。
这是一个金融领域特有的归纳偏置，standard SelectiveNet 没有。

---

## 架构

```
                    ┌──────────────────────────────────────────┐
                    │           DynHetGNN Backbone              │
                    │  (SAGEConv/HGT on monthly HeteroGraph)   │
                    └─────────────────┬────────────────────────┘
                                      │
                                   h ∈ R^d  (event-level GNN hidden state)
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                   │
                    ▼                 ▼                   ▼
            ┌──────────────┐  ┌─────────────────┐  ┌──────────────┐
            │ Prediction   │  │ Selection Head   │  │ Auxiliary     │
            │ Head         │  │ (NOVEL)          │  │ Head          │
            │              │  │                  │  │              │
            │ MLP(h)→σ→p  │  │ MLP([h;m])→σ→s  │  │ MLP(h)→σ→a  │
            │              │  │                  │  │              │
            │ p ∈ [0,1]   │  │ s ∈ [0,1]       │  │ a ∈ [0,1]   │
            │ (up prob)    │  │ (select prob)    │  │ (regularizer)│
            └──────────────┘  └─────────────────┘  └──────────────┘
                    │                 │
                    ▼                 ▼
              prediction loss    coverage constraint
                    └────────┬────────┘
                             ▼
                     Selective Loss
```

### Market Context Vector m

```python
# ALL features use STRICTLY T-1 CLOSE VALUES — no exceptions.
# Using T-0 (same day) values introduces look-ahead bias because
# same-day VIX correlates with same-day returns.
market_context = {
    "vix":              float,   # VIX close on T-1
    "spy_drawdown":     float,   # SPY drawdown from 52-week high, as of T-1
    "realized_vol_30d": float,   # SPY 30-day realized volatility, as of T-1
    "market_breadth":   float,   # % of S&P 500 stocks above 20-day MA, as of T-1
}
# Concatenated: m ∈ R^4
```

**Ablation required**: T-0 vs T-1 context comparison. If T-0 significantly
outperforms T-1, this is evidence of information leakage, not model quality.

---

## Loss Function

### Standard SelectiveNet Loss (baseline for comparison)

```
L_standard = (1/n) * Σ [s_i * CE(p_i, y_i)] / mean(s) + λ * max(0, c_target - mean(s))²
```

### Our Volatility-Calibrated Loss

```
L_ours = L_selective + λ_cov * L_coverage + λ_aux * L_auxiliary

where:
  L_selective = (1/n) * Σ [s_i * CE(p_i, y_i)] / mean(s)
  L_coverage  = per-month coverage penalty (see below)
  L_auxiliary = (1/n) * Σ CE(a_i, y_i)   [standard auxiliary head]
```

### Per-Month Coverage Constraint

```python
def per_month_coverage_loss(select_probs, month_ids, min_coverage=0.05):
    """
    Force minimum coverage per calendar month.
    Prevents model from abstaining entirely during volatile periods.
    """
    loss = 0.0
    unique_months = month_ids.unique()
    for m in unique_months:
        mask = (month_ids == m)
        monthly_coverage = select_probs[mask].mean()
        # Penalize if monthly coverage drops below minimum
        loss += torch.relu(min_coverage - monthly_coverage) ** 2
    return loss / len(unique_months)
```

**Why per-month, not global**: Global coverage target allows the model to "hide" —
predict 30% in calm 2021 and 0% in volatile 2022, averaging to target.
Per-month forces stable behavior across regimes.

---

## PyTorch Implementation Sketch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VolatilityCalibratedSelectiveNet(nn.Module):
    """
    SelectiveNet with market-regime-aware selection head.

    Key difference from standard SelectiveNet:
    selection = g(h, m) instead of g(h), where m = market context vector.
    """

    def __init__(self, input_dim, hidden_dim=128, market_dim=4, dropout=0.3):
        super().__init__()

        # Shared backbone output: h ∈ R^input_dim
        # (In practice, this comes from the GNN; here we define the heads only)

        # Prediction head: h → p ∈ [0,1]
        self.pred_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Selection head: [h; m] → s ∈ [0,1]  ← THIS IS THE NOVELTY
        self.select_head = nn.Sequential(
            nn.Linear(input_dim + market_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Auxiliary head: h → a ∈ [0,1]  (regularizer, standard)
        self.aux_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h, market_context):
        """
        Args:
            h: GNN hidden states, shape (batch, input_dim)
            market_context: market regime features, shape (batch, market_dim)
        Returns:
            pred: prediction probabilities (batch,)
            select: selection probabilities (batch,)
            aux: auxiliary predictions (batch,)
        """
        pred = torch.sigmoid(self.pred_head(h)).squeeze(-1)

        # Selection head sees BOTH hidden state AND market context
        select_input = torch.cat([h, market_context], dim=-1)
        select = torch.sigmoid(self.select_head(select_input)).squeeze(-1)

        aux = torch.sigmoid(self.aux_head(h)).squeeze(-1)

        return pred, select, aux


class SelectiveLoss(nn.Module):
    """
    Volatility-calibrated selective prediction loss.
    """

    def __init__(self, target_coverage=0.5, min_monthly_coverage=0.05,
                 lambda_coverage=10.0, lambda_aux=0.5):
        super().__init__()
        self.target_coverage = target_coverage
        self.min_monthly_coverage = min_monthly_coverage
        self.lambda_coverage = lambda_coverage
        self.lambda_aux = lambda_aux

    def forward(self, pred, select, aux, labels, month_ids):
        """
        Args:
            pred: prediction probs (batch,)
            select: selection probs (batch,)
            aux: auxiliary preds (batch,)
            labels: ground truth (batch,)
            month_ids: calendar month identifier per event (batch,)
        """
        # Empirical coverage
        coverage = select.mean()

        # Selective prediction loss (weighted by selection)
        bce = F.binary_cross_entropy(pred, labels.float(), reduction='none')
        selective_loss = (select * bce).sum() / (coverage * len(labels) + 1e-8)

        # Global coverage target
        coverage_loss = torch.relu(self.target_coverage - coverage) ** 2

        # Per-month coverage stability
        monthly_loss = torch.tensor(0.0, device=pred.device)
        for m in month_ids.unique():
            mask = (month_ids == m)
            m_cov = select[mask].mean()
            monthly_loss += torch.relu(self.min_monthly_coverage - m_cov) ** 2
        monthly_loss /= max(month_ids.unique().numel(), 1)

        # Auxiliary loss (standard regularizer)
        aux_loss = F.binary_cross_entropy(aux, labels.float())

        # Total
        total = (selective_loss
                 + self.lambda_coverage * (coverage_loss + monthly_loss)
                 + self.lambda_aux * aux_loss)

        # Logging
        metrics = {
            'selective_loss': selective_loss.item(),
            'coverage': coverage.item(),
            'monthly_cov_min': min(select[month_ids == m].mean().item()
                                   for m in month_ids.unique()),
            'aux_loss': aux_loss.item(),
            'total_loss': total.item(),
        }

        return total, metrics
```

---

## Training Protocol

### Primary Specification: Staged Training (防止训练目标冲突)

**Problem**: In end-to-end training, the prediction head is incentivized to
deliberately lower confidence on difficult samples (pushing them to be rejected),
artificially inflating AUC on the selected subset. This is NOT the same as the
"trivial strategy" concern — it's a training dynamics bias.

**Solution**: Two-stage training.

```python
# Stage 1: Train GNN + prediction head + auxiliary head (NO selection)
for epoch in range(stage1_epochs):
    h = gnn_backbone(graph)
    pred = prediction_head(h)
    aux = auxiliary_head(h)
    loss = BCE(pred, labels) + lambda_aux * BCE(aux, labels)
    loss.backward()
    optimizer_stage1.step()

# Stage 2: Freeze backbone + prediction head, train ONLY selection head
for param in gnn_backbone.parameters():
    param.requires_grad = False
for param in prediction_head.parameters():
    param.requires_grad = False

for epoch in range(stage2_epochs):
    h = gnn_backbone(graph)  # frozen
    pred = prediction_head(h)  # frozen
    market_ctx = get_market_context(dates)  # T-1 values
    select = selection_head(torch.cat([h, market_ctx], dim=-1))

    # Selection loss on frozen predictions
    bce = F.binary_cross_entropy(pred, labels, reduction='none')
    selective_loss = (select * bce).sum() / (select.mean() * len(labels) + 1e-8)
    coverage_loss = per_month_coverage_loss(select, month_ids)

    loss = selective_loss + lambda_cov * coverage_loss
    loss.backward()
    optimizer_stage2.step()
```

### Secondary Specification: End-to-End Training

- Train all components jointly (as in standard SelectiveNet)
- **Must additionally report**: prediction head AUC on rejected subset,
  compared with/without selection head
- If end-to-end AUC >> staged AUC → suspect training dynamics bias

---

## Evaluation Plan

### Baselines to compare against:

| Method | Description | What it tests |
|--------|-------------|---------------|
| F.0 Random | Random X% selection | Is selective better than random? |
| F.1 Confidence Threshold | Post-hoc |p-0.5| ranking | Is end-to-end better than post-hoc? |
| F.2a Standard SelectiveNet | g(h) without market context | Does market context help? |
| F.2b Ours (VC-SelectiveNet), staged | g(h,m) with market context, staged training | Full method (PRIMARY) |
| F.2c Ours (VC-SelectiveNet), e2e | g(h,m) with market context, end-to-end | Check for training dynamics bias |
| F.2d T-0 context ablation | g(h,m) but m uses T-0 instead of T-1 | Leak detection: if T-0 >> T-1, it's bias |

### Key analyses:

1. **Coverage-AUC curve**: For each method, plot AUC vs coverage [5%, 10%, ..., 100%]
2. **Jaccard(F.1, F.2b)**: How different are the selected sets?
3. **Per-month coverage stability**: Variance of monthly coverage for F.2a vs F.2b
4. **Regime-conditional AUC**: Does F.2b maintain AUC in high-VIX periods where F.2a degrades?
5. **ECE plot**: Are confidence scores well-calibrated?
6. **Selected event characteristics**: Sector distribution, avg |return|, market regime distribution
7. **Rejected subset AUC** [NEW]: Compare prediction head AUC on rejected events with/without selection head training — detects "prediction head gaming"
8. **Staged vs E2E gap** [NEW]: If e2e AUC >> staged AUC, flag as potential training dynamics bias
9. **T-0 vs T-1 leak test** [NEW]: If T-0 context >> T-1, the market context is leaking label info

### Expected narrative:

"Standard SelectiveNet (F.2a) achieves AUC X at Y% coverage, but its coverage collapses
during volatile periods (2022 Q1-Q2 coverage drops to Z%). Our volatility-calibrated
selection head (F.2b) maintains stable coverage across regimes while achieving comparable
or better AUC, because the market context signal allows the model to adaptively adjust
its confidence threshold rather than panicking uniformly."

---

## Hyperparameter Search Space

| Parameter | Range | Default |
|-----------|-------|---------|
| target_coverage | [0.1, 0.2, 0.3, 0.5] | 0.3 |
| min_monthly_coverage | [0.03, 0.05, 0.10] | 0.05 |
| lambda_coverage | [1.0, 5.0, 10.0, 50.0] | 10.0 |
| lambda_aux | [0.1, 0.5, 1.0] | 0.5 |
| selection_head_hidden | [64, 128] | 128 |

**Strategy**: Grid search on Fold 1, validate on Fold 2.
