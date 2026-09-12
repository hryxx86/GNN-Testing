---
reviewer: codex
date: 2026-04-29
touchpoint: discussion_C
topic: alternative_loss_noise_reduction
---

# Research Discussion C: Alternative Losses & Noise Reduction

Project anchors from the required files:

- The preregistered Stage 1 verdict is fixed: 600 cells, 0/8 co-primary rejection versus MSE on IC and Sharpe. Any new result is supplementary.
- The current loss API is simple: `loss(pred, target, mask, **kwargs)` registered in `LOSS_REGISTRY`, with `dispatch_loss()` called once per day in `train_one_v2()`. Current losses are pointwise MSE, ListMLE, pairwise hinge-margin, and ApproxNDCG.
- The labels are cross-sectionally standardized 21d forward returns. Per-day universe size is about 500 stocks. The objective is rank IC and long-short/top-bottom portfolio behavior, not exact return value.
- The non-negotiable diagnostic is ListMLE fold-4 collapse: 6/6 architecture x feature combinations collapse on fold 4 with IC in roughly [-0.36, -0.28], fold standard deviation about 3x MSE. Pairwise hinge does not collapse the same way, but it compresses prediction scale in 4/4 contrasts under cluster bootstrap.
- The working noise split, about 70% noise and 30% overfitting residual, is plausible from the fold-2 ListMLE pilot val/test sign flip: val IC about +0.115, test IC about -0.045.

Notation used below: on day `t`, valid stocks are `V_t`, labels are `y_i`, scores are `s_i`, residuals are `r_i=s_i-y_i`, and `n=|V_t|`. All formulas are per day unless stated otherwise.

## Q1: Cross-Domain Ranking Losses

### Highest-ROI candidates

My recommended 3-5 are:

1. **Robust pointwise regression: Huber, Tukey biweight, truncated MSE.**
2. **Anchored Bradley-Terry / RankNet / BPR logistic pairwise loss**, not the current pure hinge.
3. **Tempered top-one ListNet**, as a controlled listwise test, not as a full ListMLE replacement.
4. **Soft Spearman IC via torchsort**, as a metric-aligned diagnostic rather than first-line production loss.
5. **GCE-BT pairwise plus Group-DRO**, only after the simple anchored pairwise and robust pointwise baselines are run.

I would not spend first-pass budget on Triplet, Circle, or ArcFace. They are good metric-learning losses for embedding spaces with class/identity semantics, but our model outputs one scalar per stock per day. Converting a scalar cross-sectional ranking problem into an embedding metric problem creates extra degrees of freedom without a natural "same class" definition. Triplet loss, e.g. Schroff et al. CVPR 2015, optimizes

$$
L_{triplet}=\sum_a [d(f_a,f_p)-d(f_a,f_n)+m]_+,
$$

which requires an anchor-positive-negative structure. In this setting positives and negatives would be induced from noisy future ranks, so it degenerates into a pairwise rank loss with more sampling variance. Circle loss (Sun et al., CVPR 2020, arXiv:2002.10857) and ArcFace (Deng et al., CVPR 2019, arXiv:1801.07698) are angular/classification-margin losses; the class labels are absent here. I would only revisit them if the architecture is changed to learn stock embeddings for a downstream nearest-neighbor or sector-neutral relative-value task.

### Candidate 1: Robust pointwise losses

Evidence basis: Huber's robust loss (Huber, Annals of Mathematical Statistics, 1964) and Tukey biweight M-estimation are classical robust regression tools. GCE (Zhang and Sabuncu, NeurIPS 2018, arXiv:1805.07836), SCE (Wang et al., ICCV 2019, arXiv:1908.06112), and Active Negative Loss (Ye et al., NeurIPS 2023) are classification losses, but their shared principle is useful: reduce the gradient contribution of likely mislabeled examples. For continuous 21d labels, Huber/Tukey/truncated MSE are the direct analogs.

Formula:

$$
L_{Huber,\delta}(r)=
\begin{cases}
0.5r^2, & |r|\le \delta,\\
\delta(|r|-0.5\delta), & |r|>\delta.
\end{cases}
$$

$$
L_{Tukey,c}(r)=
\begin{cases}
\frac{c^2}{6}\left[1-\left(1-\left(\frac{r}{c}\right)^2\right)^3\right], & |r|\le c,\\
\frac{c^2}{6}, & |r|>c.
\end{cases}
$$

$$
L_{trunc,c}(r)=\min(r^2,c^2).
$$

Why it fits our setup: MSE is already the baseline winner or at least non-inferior. If about 70% of the 21d cross-sectional labels are noise-dominant, the cheapest improvement is not more ranking structure, but controlling the gradient from large, likely regime-specific residuals. This directly addresses the "noise dominant" part without inheriting ListMLE's global permutation likelihood.

Failure mode overlap with ListMLE fold-4: low. These losses are pointwise and do not impose a full-list likelihood. They can still fail if fold 4 is a true sign-reversal regime, but they should fail like MSE, not catastrophically invert the entire ranking. The main risk is over-clipping true extremes, which are exactly the names that drive top/bottom portfolios.

PyTorch integration:

```python
def huber_loss(pred, target, mask, delta: float = 1.0, **kwargs):
    p = pred[mask]
    t = target[mask]
    return F.huber_loss(p, t, delta=delta, reduction="mean")

def tukey_biweight_loss(pred, target, mask, c: float = 2.0, **kwargs):
    p = pred[mask]
    t = target[mask]
    r = p - t
    u = r / c
    inside = (u.abs() < 1.0)
    loss = torch.empty_like(r)
    loss[inside] = (c * c / 6.0) * (1.0 - (1.0 - u[inside].pow(2)).pow(3))
    loss[~inside] = c * c / 6.0
    return loss.mean()

def truncated_mse_loss(pred, target, mask, c: float = 2.0, **kwargs):
    r2 = (pred[mask] - target[mask]).pow(2)
    return torch.clamp(r2, max=c * c).mean()

LOSS_REGISTRY.update({
    "huber": huber_loss,
    "tukey": tukey_biweight_loss,
    "trunc_mse": truncated_mse_loss,
})
```

### Candidate 2: Anchored Bradley-Terry / RankNet / BPR

Evidence basis: RankNet (Burges et al., ICML 2005) models pairwise preferences with a logistic probability. BPR (Rendle et al., UAI 2009) uses the same core `-log sigmoid(s_i-s_j)` idea for implicit preference ranking. The 2025 stock-ranking benchmark by Kwiatkowski and Chudziak (CIKM 2025, DOI:10.1145/3746252.3760812; arXiv:2510.14156) reports strong Sharpe for Margin/ListNet and best IC for RankNet, but their setup is materially easier/different: daily horizon and a smaller/top-cap universe versus our 21d horizon and about 500 stocks.

Formula, with a pointwise anchor:

$$
P(i \succ j|s)=\sigma((s_i-s_j)/\tau),
$$

$$
L_{BT}=\frac{\sum_{(i,j)\in P_t} w_{ij}\log(1+\exp(-(s_i-s_j)/\tau))}
{\sum_{(i,j)\in P_t} w_{ij}},
$$

where

$$
P_t=\{(i,j): y_i-y_j>\epsilon_y,\ i\in top_q(y)\ \text{or}\ j\in bottom_q(y)\},
\quad
w_{ij}=\min(|y_i-y_j|,w_{max}).
$$

Use it as:

$$
L=(1-\alpha)L_{Huber}+ \alpha L_{BT}+\lambda_{\sigma}[\max(0,\sigma_{min}-std(s))]^2.
$$

Why it fits our setup: the current pairwise hinge saturates once the small margin is met and empirically collapses prediction scale. Logistic BT/RankNet gives nonzero gradients around the boundary and can be limited to top-bottom economically relevant pairs. The Huber anchor is not optional; without it, pairwise translation/scale invariances plus weight decay can recreate the current scale-collapse failure.

Failure mode overlap with ListMLE fold-4: medium. It shares pairwise preference labels with ListMLE but not the full Plackett-Luce list likelihood. It should be less vulnerable to one anomalous full-list regime, but can still learn a sign-inverted preference map if the training/validation split rewards that map. The explicit viability gate should be: fold-4 IC must not be below MSE by more than 0.05 in any architecture x feature pilot, and median `pred_cs_std` must stay above a pre-specified floor such as 0.05.

PyTorch integration:

```python
def anchored_ranknet_loss(
    pred, target, mask,
    tau: float = 0.25,
    top_frac: float = 0.20,
    y_gap: float = 0.10,
    alpha: float = 0.50,
    w_max: float = 3.0,
    huber_delta: float = 1.0,
    sigma_min: float = 0.05,
    sigma_penalty: float = 0.05,
    **kwargs,
):
    p = pred[mask]
    t = target[mask]
    n = p.numel()
    if n < 10:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    k = max(1, int(top_frac * n))
    order = torch.argsort(t, descending=True)
    keep = torch.zeros(n, dtype=torch.bool, device=p.device)
    keep[order[:k]] = True
    keep[order[-k:]] = True

    dt = t.unsqueeze(1) - t.unsqueeze(0)
    pair_mask = (dt > y_gap) & (keep.unsqueeze(1) | keep.unsqueeze(0))
    if pair_mask.sum() == 0:
        return F.huber_loss(p, t, delta=huber_delta)

    ds = p.unsqueeze(1) - p.unsqueeze(0)
    weights = dt.clamp(min=0.0, max=w_max).detach()
    rank_loss = F.softplus(-ds[pair_mask] / tau)
    rank_loss = (rank_loss * weights[pair_mask]).sum() / weights[pair_mask].sum().clamp(min=1.0)

    anchor = F.huber_loss(p, t, delta=huber_delta)
    scale_guard = F.relu(sigma_min - p.std()).pow(2)
    return (1.0 - alpha) * anchor + alpha * rank_loss + sigma_penalty * scale_guard

LOSS_REGISTRY["anchored_ranknet"] = anchored_ranknet_loss
```

### Candidate 3: Tempered top-one ListNet

Evidence basis: ListNet (Cao et al., ICML 2007) introduced listwise learning through top-one/list probabilities. It is a weaker listwise objective than ListMLE (Xia et al., ICML 2008), because it does not multiply sequential conditional probabilities down the full permutation. The CIKM 2025 benchmark reports strong Sharpe for ListNet, but again under daily/shorter-universe conditions.

Formula:

$$
P_y(i)=\frac{\exp(\tilde y_i/\tau_y)}{\sum_j \exp(\tilde y_j/\tau_y)},\quad
P_s(i)=\frac{\exp((s_i-\bar s)/\tau_s)}{\sum_j \exp((s_j-\bar s)/\tau_s)}
$$

$$
L_{ListNet}=-\sum_i P_y(i)\log P_s(i).
$$

Here `tilde y` should be label-clipped, e.g. `clip(y, -2, 2)`, and `tau_y >= 1` should be used to avoid letting one noisy 21d winner dominate the target distribution.

Why it fits our setup: it is the cleanest test of "maybe listwise is useful, but ListMLE was too sharp." It keeps whole-list information but removes ListMLE's repeated tail log-sum-exp terms. It is also easy to integrate.

Failure mode overlap with ListMLE fold-4: high. It is still a softmax listwise loss, so if fold 4 is a regime where the learned order is systematically inverted, ListNet can fail in the same direction. I would only run this with aggressive temperature and label clipping, and I would stop it immediately if fold-4 pilot IC falls below -0.15 or fold standard deviation exceeds 2x MSE.

PyTorch integration:

```python
def tempered_listnet_loss(
    pred, target, mask,
    tau_y: float = 1.5,
    tau_s: float = 1.0,
    y_clip: float = 2.0,
    **kwargs,
):
    p = pred[mask]
    t = target[mask]
    if p.numel() < 10:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    t_clip = torch.clamp(t.detach(), -y_clip, y_clip)
    py = torch.softmax(t_clip / tau_y, dim=0)
    ps_log = torch.log_softmax((p - p.mean()) / tau_s, dim=0)
    return -(py * ps_log).sum()

LOSS_REGISTRY["listnet_temp"] = tempered_listnet_loss
```

### Candidate 4: Soft Spearman IC

Evidence basis: Blondel et al. (ICML 2020, arXiv:2002.08871) propose fast differentiable sorting and ranking; the `torchsort` package exposes `soft_rank` and shows a differentiable Spearman example. SoDeep (Engilberge et al., CVPR 2019, arXiv:1904.04272) is another learned differentiable sorter, but torchsort is simpler and more reproducible. Numerai forum implementations are useful engineering references, not peer-reviewed evidence.

Formula:

$$
\rho_{soft}(s,y)=corr(\operatorname{soft\_rank}(s),\operatorname{rank}(y)),
\quad
L_{IC}= -\rho_{soft}(s,y).
$$

Why it fits our setup: IC is the primary predictive metric. Directly optimizing a smooth approximation removes the surrogate mismatch of MSE/listwise/pairwise losses.

Failure mode overlap with ListMLE fold-4: medium. It is global-rank based, but not likelihood based and not exponential in the top items. The risk is different: the gradient can be very sensitive to the smoothing strength. If too sharp, it becomes high-variance and unstable around noisy rank ties; if too smooth, it behaves like a weak correlation loss and may underfit.

PyTorch integration:

```python
def soft_spearman_ic_loss(
    pred, target, mask,
    regularization: str = "l2",
    regularization_strength: float = 1.0,
    **kwargs,
):
    import torchsort

    p = pred[mask]
    t = target[mask].detach()
    if p.numel() < 10:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pr = torchsort.soft_rank(
        p.unsqueeze(0),
        regularization=regularization,
        regularization_strength=regularization_strength,
    ).squeeze(0)
    tr = torch.argsort(torch.argsort(t)).float() + 1.0

    pr = pr - pr.mean()
    tr = tr - tr.mean()
    pr = pr / pr.norm().clamp(min=1e-6)
    tr = tr / tr.norm().clamp(min=1e-6)
    return -(pr * tr).sum()

LOSS_REGISTRY["soft_spearman"] = soft_spearman_ic_loss
```

### Candidate 5: GCE-BT pairwise

Evidence basis: GCE was designed for classification under noisy labels (Zhang and Sabuncu, NeurIPS 2018). ANL (Ye et al., NeurIPS 2023) improves active/passive noisy-label losses for classification, but I do not see a directly validated regression/ranking analog. The defensible adaptation is a bounded pairwise logistic loss, not a literal multiclass ANL port.

Formula:

$$
q_{ij}=\sigma((s_i-s_j)/\tau),\quad y_i>y_j,
$$

$$
L_{GCE-BT}=\frac{\sum_{(i,j)\in P_t} w_{ij}\frac{1-q_{ij}^{q}}{q}}
{\sum_{(i,j)\in P_t}w_{ij}},\quad q\in(0,1].
$$

As `q -> 0`, this approaches cross-entropy; larger `q` bounds the loss more aggressively.

Why it fits our setup: it directly targets noisy pairwise preferences. Under a 21d horizon, many pairwise orders are effectively noise. Bounded pairwise loss prevents a few extreme wrong pairs from dominating.

Failure mode overlap with ListMLE fold-4: medium-high if used alone. The bounded loss can ignore genuinely hard regimes, which may make fold-4 worse even while average validation improves. This is why I would only test it with an anchor and then Group-DRO/CVaR if the simple version is not unstable.

PyTorch integration:

```python
def gce_bt_loss(
    pred, target, mask,
    tau: float = 0.25,
    gce_q: float = 0.7,
    alpha: float = 0.50,
    top_frac: float = 0.20,
    y_gap: float = 0.10,
    **kwargs,
):
    p = pred[mask]
    t = target[mask]
    n = p.numel()
    if n < 10:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    k = max(1, int(top_frac * n))
    order = torch.argsort(t, descending=True)
    keep = torch.zeros(n, dtype=torch.bool, device=p.device)
    keep[order[:k]] = True
    keep[order[-k:]] = True

    dt = t.unsqueeze(1) - t.unsqueeze(0)
    pair_mask = (dt > y_gap) & (keep.unsqueeze(1) | keep.unsqueeze(0))
    if pair_mask.sum() == 0:
        return F.huber_loss(p, t, delta=1.0)

    prob = torch.sigmoid((p.unsqueeze(1) - p.unsqueeze(0)) / tau).clamp(1e-6, 1.0)
    weights = dt.clamp(min=0.0, max=3.0).detach()
    gce = (1.0 - prob[pair_mask].pow(gce_q)) / gce_q
    rank_loss = (gce * weights[pair_mask]).sum() / weights[pair_mask].sum().clamp(min=1.0)
    anchor = F.huber_loss(p, t, delta=1.0)
    return (1.0 - alpha) * anchor + alpha * rank_loss

LOSS_REGISTRY["gce_bt"] = gce_bt_loss
```

## Q2: Direct IC Optimization

Direct Spearman optimization is scientifically attractive but not my first production bet for this specific 21d task. The core tradeoff is bias versus variance:

- MSE/Huber is biased for rank IC but has low gradient variance and preserves prediction scale.
- Pairwise/listwise losses reduce metric mismatch but amplify noisy pair/list labels.
- Soft Spearman removes metric mismatch but creates a high-variance global-rank gradient on about 500 stocks per day.

Blondel et al. (ICML 2020, arXiv:2002.08871) is the strongest foundation here because it gives exact differentiable sorting/ranking operators with efficient complexity. SoDeep (Engilberge et al., CVPR 2019, arXiv:1904.04272) is less appealing in this project because it adds a learned surrogate module and a second model component; the extra approximation is hard to defend when the core result is already about loss-function failure under regime shift.

Formula:

$$
L_{softIC}(t)=-corr(\operatorname{soft\_rank}(s_t;\lambda), \operatorname{rank}(y_t)).
$$

For our setup I would combine it with a pointwise anchor:

$$
L=(1-\alpha)L_{Huber}+\alpha L_{softIC},\quad \alpha \in \{0.25,0.50\}.
$$

The anchor matters because rank-only losses are invariant to monotone transforms and can again leave prediction scale uncontrolled, which matters for diagnostics and portfolio construction.

PyTorch integration:

```python
def anchored_soft_ic_loss(pred, target, mask, alpha: float = 0.50, **kwargs):
    ic_loss = soft_spearman_ic_loss(pred, target, mask, **kwargs)
    anchor = F.huber_loss(pred[mask], target[mask], delta=1.0)
    return (1.0 - alpha) * anchor + alpha * ic_loss

LOSS_REGISTRY["anchored_soft_ic"] = anchored_soft_ic_loss
```

Expected OOS generalization: I expect Huber or anchored RankNet to generalize better than pure soft Spearman on the full 5-fold OOS. The reason is first-principles rather than a finance-specific published result: with 21d overlapping labels, adjacent ranks are very noisy, and Spearman treats all rank swaps equally after ranking. A robust pointwise loss lets small/noisy rank swaps remain small residuals. Soft Spearman can spend capacity rearranging middle ranks that never enter the portfolio. If we use soft IC, restrict the objective to top/bottom weights or combine it with Huber.

Failure mode analysis: soft IC is less likely than ListMLE to produce the exact fold-4 softmax-likelihood collapse, but it can overfit the validation-era rank operator. The viability gate should include:

- per-fold IC table, not just aggregate;
- fold-4 IC versus MSE;
- `pred_cs_std` versus MSE;
- IC missingness, because constant predictions make Spearman undefined.

## Q3: DRO for Fold-4 Collapse

The best match is **Group-DRO over time/regime groups**, with CVaR-DRO as a fallback if groups are hard to define. I would not start with Wasserstein-DRO or generic KL/chi-square f-DRO.

Evidence basis:

- Group-DRO for neural networks under group shifts is directly studied by Sagawa et al. (ICLR 2020, "Distributionally Robust Neural Networks for Group Shifts"). A key lesson is that Group-DRO only works with sufficient regularization/early stopping; naive overparameterized Group-DRO can still overfit.
- Duchi and Namkoong's DRO work ("Learning Models with Uniform Performance via Distributionally Robust Optimization", Annals of Statistics 2021; arXiv:1810.08750) is a strong basis for tail/worst-subpopulation robustness.
- Wasserstein-DRO adversarial training (Sinha, Namkoong, Duchi, ICLR 2018, arXiv:1710.10571) is more natural for input perturbation robustness. Our problem is not small perturbations of features; it is a calendar/regime label relationship shift.

Formula for Group-DRO:

$$
L_g(\theta)=\frac{1}{|D_g|}\sum_{(x,y)\in D_g}\ell_\theta(x,y),\quad
\min_\theta \max_g L_g(\theta).
$$

Stochastic exponentiated-gradient implementation:

$$
q_g \leftarrow \frac{q_g\exp(\eta L_g)}{\sum_h q_h\exp(\eta L_h)},\quad
L_{DRO}=\sum_g q_gL_g.
$$

Formula for CVaR-DRO over daily losses:

$$
L_{CVaR,\alpha}(\theta)=\min_\eta\left[\eta+\frac{1}{1-\alpha}E(\ell_\theta-\eta)_+\right].
$$

Concrete recommendation: define groups within each training fold using only training data:

- calendar quarter or half-year;
- high/low market dispersion group, using same-day cross-sectional label dispersion or past realized volatility available at training time;
- optionally high/low turnover/volume regime if S8/Alpha158 includes volume-derived factors.

Do not define a group using fold-4 test membership. That would be leakage. The goal is to protect against a fold-4-like regime by upweighting historical regimes with high loss or similar dispersion, not to train on fold 4.

Training-loop pseudocode:

```python
# Precompute group_id_by_day for train_days only, no test information.
# Example: quarter id crossed with high/low realized-vol regime.
group_ids = sorted(set(group_id_by_day[int(d)] for d in train_days))
group_to_idx = {g: i for i, g in enumerate(group_ids)}
q = torch.ones(len(group_ids), device=pa.DEVICE) / len(group_ids)
dro_eta = 0.05

for ep in range(hparams["epochs"]):
    model.train()
    losses_by_group = {g: [] for g in group_ids}

    for d in train_days[np.random.permutation(len(train_days))]:
        x = features_t[d].to(pa.DEVICE)
        pred = model(x, full_ei)
        mask = label_valid_t[d].to(pa.DEVICE)
        target = labels_t[d].to(pa.DEVICE)
        if mask.sum() < 10:
            continue
        base = dispatch_loss(loss_type, pred, target, mask, loss_kwargs, rng_state=rng_state)
        losses_by_group[group_id_by_day[int(d)]].append(base)

    group_losses = []
    for g in group_ids:
        if losses_by_group[g]:
            group_losses.append(torch.stack(losses_by_group[g]).mean())
        else:
            group_losses.append(torch.tensor(0.0, device=pa.DEVICE))
    group_losses = torch.stack(group_losses)

    with torch.no_grad():
        q *= torch.exp(dro_eta * group_losses.detach())
        q /= q.sum().clamp(min=1e-12)

    opt.zero_grad()
    dro_loss = (q.detach() * group_losses).sum()
    dro_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

Failure mode analysis:

- Group-DRO can overfit the worst training group if the group has little data. With only 5 years and about 60 test days per fold, quarter-level groups are already small. Use coarse groups first.
- CVaR can chase noisy high-loss days. For a 70% noise hypothesis, pure CVaR over days may overweight label noise. Use CVaR on group-averaged losses, not individual stock residuals.
- f-DRO with KL/chi-square balls is less interpretable for a paper story because "fold 4 anomalous regime" is temporal, not an abstract distribution ball.
- Wasserstein-DRO is likely the wrong failure model: small feature perturbations did not explain the earlier fold-4 anomaly; the issue is relationship shift.

## Q4: SAM / mSAM

Evidence basis:

- SAM (Foret et al., ICLR 2021, arXiv:2010.01412) optimizes a local worst-case loss:

$$
\min_w \max_{\|\epsilon\|_2\le \rho} L(w+\epsilon).
$$

The first-order perturbation is:

$$
\epsilon=\rho\frac{\nabla_w L(w)}{\|\nabla_w L(w)\|_2}.
$$

- The 2025 "no overhead" method appears to be **Momentum-SAM / MSAM** (Becker, Altrock, Risse, NeurIPS 2025), which perturbs parameters in the direction of the accumulated momentum vector instead of computing an extra SAM gradient. There is also a different "mSAM" line, micro-batch-averaged SAM, from OPT 2022 / arXiv:2302.09693; that is not the same no-overhead claim.
- I found no verified published application specifically to cross-sectional stock return prediction or GNN stock ranking. Treat any finance IC delta as an engineering hypothesis, not literature-backed.

SAM pseudocode integration, simplified:

```python
def grad_norm(model):
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.norm(p=2))
    return torch.norm(torch.stack(norms), p=2).clamp(min=1e-12)

def sam_step_one_day(model, opt, loss_fn, rho=0.05):
    # First backward on normal weights.
    loss = loss_fn()
    loss.backward()
    scale = rho / grad_norm(model)

    eps = []
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                eps.append(None)
                continue
            e = p.grad * scale
            p.add_(e)
            eps.append(e)

    # Second backward on perturbed weights.
    opt.zero_grad()
    loss_perturbed = loss_fn()
    loss_perturbed.backward()

    with torch.no_grad():
        for p, e in zip(model.parameters(), eps):
            if e is not None:
                p.sub_(e)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

Momentum-SAM style pseudocode:

```python
def msam_perturb_from_adam_momentum(model, opt, rho=0.05):
    vecs = []
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state[p]
            m = state.get("exp_avg")
            if m is not None:
                vecs.append(m.norm(p=2))
    denom = torch.norm(torch.stack(vecs), p=2).clamp(min=1e-12) if vecs else None
    if denom is None:
        return []

    eps = []
    with torch.no_grad():
        for group in opt.param_groups:
            for p in group["params"]:
                m = opt.state[p].get("exp_avg")
                if m is None:
                    eps.append((p, None))
                    continue
                e = rho * m / denom
                p.add_(e)
                eps.append((p, e))
    return eps

def restore_eps(eps):
    with torch.no_grad():
        for p, e in eps:
            if e is not None:
                p.sub_(e)
```

Expected IC delta for our setup: I would budget SAM/MSAM as **+0.000 to +0.004 mean IC**, with a larger possible effect on fold variance than on average IC. This is first-principles: SAM targets the 30% overfitting residual, not the 70% label-noise ceiling. It will not fix a true sign-reversal regime. Full SAM roughly doubles training time and is poor ROI under a 30h M4 budget unless a smaller pilot shows validation/test fold stability. MSAM is worth a cheap optimizer wrapper pilot because its compute overhead is low, but it is too new to make a strong paper claim in finance.

Failure mode analysis:

- SAM can improve flatness while preserving the wrong objective. SAM + ListMLE can still learn a flatter version of the fold-4-inverting ranking map.
- SAM doubles cost and complicates the current grad accumulation loop.
- MSAM depends on optimizer momentum quality. With Adam and per-day losses, the momentum vector may reflect noisy daily rank shocks.

Recommendation: test MSAM with Huber/MSE, not ListMLE. Do not run full SAM before the robust-loss and hparam sweeps.

## Q5: Data Augmentation for Cross-Sectional Ranking

Evidence basis:

- C-Mixup (Yao et al., NeurIPS 2022, arXiv:2210.05775) samples regression mixup pairs by label similarity to avoid arbitrary incorrect interpolated labels.
- RC-Mixup (Hwang, Kim, Whang, KDD 2024, arXiv:2405.17938) combines C-Mixup with robust training/clean-sample identification for noisy regression.
- FrAug (Chen et al., ICLR 2023, arXiv:2302.09292) performs frequency-domain augmentation for time-series forecasting while preserving temporal relationships.

The ranking-specific constraint is that the label is not just a scalar value; it is a position in a same-day cross-section. Mixing labels across different days can destroy the rank semantics because a `+1.0` z-label on a low-dispersion day and `+1.0` on a crisis-dispersion day do not necessarily imply the same portfolio opportunity.

Safe principle 1: **Mix only within the same training day and preferably within sector/volatility buckets.**

Formula:

$$
P(j|i,t)\propto \exp\left(-\frac{(y_{it}-y_{jt})^2}{2\sigma_y^2}\right)
\mathbf{1}[\text{same sector or same vol bucket}].
$$

$$
\tilde x=\lambda x_i+(1-\lambda)x_j,\quad
\tilde y=\lambda y_i+(1-\lambda)y_j,\quad
\lambda\sim Beta(a,a).
$$

Safe principle 2: **Use mixed samples in the pointwise anchor, not in the pairwise rank set at first.** Pairwise order among synthetic samples is artificial; Huber on mixed labels is defensible, but ranking synthetic points against real stocks can create false preferences.

Safe principle 3: **FrAug only applies if the model ingests raw return sequences.** The current `run_loss_horserace.py` uses precomputed S6/S8 factor tensors, not raw lookback sequences. Applying frequency perturbations to already aggregated factors is not the method in FrAug and is hard to defend.

Pseudocode for same-day C-Mixup-style augmentation before the loss:

```python
def same_day_cmixup_features(x, y, mask, alpha=0.2, sigma_y=0.5, mix_prob=0.25):
    # x: (n_stocks, n_features), y/mask: same day.
    valid_idx = torch.where(mask)[0]
    if valid_idx.numel() < 20 or torch.rand(()) > mix_prob:
        return x, y, mask

    xv = x[valid_idx]
    yv = y[valid_idx]
    n = valid_idx.numel()

    dy2 = (yv.unsqueeze(1) - yv.unsqueeze(0)).pow(2)
    probs = torch.exp(-dy2 / (2.0 * sigma_y * sigma_y))
    probs.fill_diagonal_(0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
    partner = torch.multinomial(probs, num_samples=1).squeeze(1)

    lam = torch.distributions.Beta(alpha, alpha).sample((n,)).to(x.device)
    lam_x = lam.view(-1, 1)
    x_mix = lam_x * xv + (1.0 - lam_x) * xv[partner]
    y_mix = lam * yv + (1.0 - lam) * yv[partner]

    # Replace a random subset of valid stocks; do not change graph topology.
    x_aug = x.clone()
    y_aug = y.clone()
    x_aug[valid_idx] = x_mix
    y_aug[valid_idx] = y_mix
    return x_aug, y_aug, mask
```

Failure mode analysis:

- Mixup can blur the extremes. That is bad for top/bottom portfolios if overused.
- Cross-day mixup can leak regime information or generate labels inconsistent with the daily cross-section.
- RC-Mixup requires a clean/noisy sample detector. In our setting, "small loss" can mean "easy market beta name" rather than clean alpha signal. It should be a Tier 3 experiment, not Tier 1.
- FrAug is not appropriate for S6/S8 factors unless the pipeline is moved upstream to raw time-series windows.

Expected value: augmentation is lower ROI than robust losses and hparam regularization. Use it only if we move to raw sequence inputs or if Huber/MSE overfitting remains visible after tuning.

## Q6: Hparam Tuning

First correction: the current code does not have `weight_decay=0`; `default_hparams()` sets `weight_decay=1e-4`, and `torch.optim.Adam(..., weight_decay=...)` uses coupled L2-style decay. The Stage 0 ranking winners used `lr=2e-3`, `dropout=0.3`; MSE uses `lr=1e-3`, `dropout=0.3`; epochs=50, patience=10, grad accumulation=4.

The tuning target should be fold stability, not just mean validation IC. The fold-2 ListMLE pilot proves that val IC can be badly misleading.

Useful objective for tuning:

$$
Score = \overline{IC} - \gamma\,sd_{fold}(IC) - \kappa\,\mathbf{1}[\min_f IC_f < -0.10],
$$

with `gamma` around 0.25 to 0.50 for selection only. This is not a new primary endpoint; it is a supplementary model-selection rule to avoid fold-4-like losses.

PyTorch/config integration:

```python
def candidate_hparams(base):
    grid = []
    for lr in [1e-3, 5e-4, 2e-4]:
        for wd in [1e-4, 3e-4, 1e-3]:
            for dropout in [0.2, 0.3, 0.5]:
                hp = base.copy()
                hp.update(lr=lr, weight_decay=wd, dropout=dropout)
                grid.append(hp)
    return grid

def fold_stability_score(mean_ic, fold_sd, min_fold_ic, gamma=0.35, floor=-0.10):
    penalty = 0.05 if min_fold_ic < floor else 0.0
    return mean_ic - gamma * fold_sd - penalty
```

Per-hparam expected IC delta, specific to this setup:

| Hparam change | Expected mean IC delta | Why |
|---|---:|---|
| Adam -> AdamW with `weight_decay in {3e-4, 1e-3}` | +0.002 to +0.006 | Sagawa et al. show regularization matters for worst-group generalization; our fold variance says overfitting residual exists. |
| Lower ranking-loss LR from `2e-3` to `{5e-4, 2e-4}` | +0.000 to +0.006 | ListMLE collapse may be amplified by large softmax gradients; lower LR may reduce but not remove regime inversion. |
| Dropout `{0.2,0.5}` around current 0.3 | -0.002 to +0.004 | S6 is only 3 features, so 0.5 can underfit; S8 may benefit more. |
| Early stopping patience 5 instead of 10 | +0.001 to +0.005 | Helps the 30% overfitting residual; but val/test sign flip means it is not enough alone. |
| Gradient clip 0.3 or 0.5 for listwise/rank losses | +0.000 to +0.003 | Controls softmax/listwise spikes; unlikely to solve fold-4 sign reversal. |
| Pairwise margin/temperature with scale guard | +0.001 to +0.006 | Current margin=0.01 plus hinge saturates and scale-collapses; temperature and anchor are higher leverage than margin alone. |
| Temporal validation split rather than random/in-fold validation, if not already temporal | +0.003 to +0.010 | The pilot val/test sign flip is exactly a validation design failure. This is the cheapest high-leverage change if current val is not a strictly later block. |

Failure mode analysis:

- More tuning can silently turn supplementary experiments into selection overfit. Keep a small fixed grid, record all cells, and report the grid.
- Tuning on mean IC alone will rediscover ListMLE-like high-val/high-collapse settings.
- Weight decay can reduce prediction scale; pairwise/rank losses need `pred_cs_std` monitoring.

## Q7: Compound Loss Design

Yes, a **GCE-Margin/BT + Huber anchor + Group-DRO** compound is principled, but only if each term maps to a specific observed failure:

- Huber anchor addresses noisy 21d values and preserves scale.
- GCE-BT addresses noisy pairwise ordering by bounding gradients.
- Group-DRO addresses regime/fold instability.
- A variance floor addresses the observed pairwise prediction-scale collapse.

Compound formula:

$$
L_{day}= \lambda_h L_{Huber}
       + \lambda_r L_{GCE-BT}
       + \lambda_\sigma[\max(0,\sigma_{min}-std(s))]^2.
$$

Then apply Group-DRO over regime groups:

$$
L_{train}=\sum_g q_g \frac{1}{|D_g|}\sum_{t\in D_g} L_{day,t}.
$$

Concrete starting weights:

- `lambda_h=0.50`
- `lambda_r=0.50`
- `lambda_sigma=0.05`
- `gce_q=0.7`
- `tau=0.25`
- `top_frac=0.20`
- `y_gap=0.10`
- `dro_eta=0.03 to 0.05`

Pseudocode:

```python
def compound_gce_rank_dro_day_loss(pred, target, mask, **kwargs):
    base = gce_bt_loss(
        pred, target, mask,
        tau=kwargs.get("tau", 0.25),
        gce_q=kwargs.get("gce_q", 0.7),
        alpha=kwargs.get("rank_alpha", 0.50),
        top_frac=kwargs.get("top_frac", 0.20),
        y_gap=kwargs.get("y_gap", 0.10),
    )
    p = pred[mask]
    scale_guard = F.relu(kwargs.get("sigma_min", 0.05) - p.std()).pow(2)
    return base + kwargs.get("sigma_penalty", 0.05) * scale_guard

LOSS_REGISTRY["compound_gce_bt"] = compound_gce_rank_dro_day_loss
```

Training schedule:

```python
# Epoch 0-4: Huber only.
# Epoch 5+: compound loss.
# Optional: enable Group-DRO only after epoch 5, once group losses are meaningful.
if ep < 5:
    loss_type = "huber"
else:
    loss_type = "compound_gce_bt"
```

Why not just add everything immediately: GCE downweights hard pairs; DRO upweights hard groups. Those forces can conflict early in training when all groups are hard. A warmup makes the "hard group" signal less random.

Failure mode analysis:

- If fold 4 is a true unseen sign reversal, no compound loss can infer it from past data. DRO can only reduce sensitivity to analogous past regimes.
- If GCE is too bounded, the model may ignore the exact top/bottom errors that matter for portfolio performance.
- If `lambda_sigma` is too large, it can force noisy dispersion in predictions and hurt IC.
- This should not be the first experiment; it is an integration experiment after anchored RankNet and Huber establish baselines.

## Q8: Practical Recommendations

The practical goal is not to overturn Stage 1. It is to identify a supplementary, mechanistically defensible loss that either improves IC by a small amount or explains why no loss can clear the 21d noise ceiling. Every ranking experiment should carry the fold-4 viability gate:

$$
\min_{\text{arch,feature}} IC_{fold4} > -0.15,\quad
sd_{fold}(IC) \le 2\,sd_{fold}(MSE),\quad
median(pred\_cs\_std)\ge 0.05.
$$

### Tier 1 (Highest ROI, Quick Wins)

1. **Robust pointwise sweep: Huber vs Tukey vs MSE.**
   - Cells: `3 losses x 2 models x 2 features x 5 folds x 5 seeds = 300`.
   - Cost: about 6-9h on M4, using Stage 1 speed as reference.
   - Expected IC delta: Huber +0.002 to +0.006; Tukey -0.002 to +0.004.
   - Complexity: low, pure `LOSS_REGISTRY` additions.
   - Why first: it tests the 70% noise hypothesis without inviting ListMLE-style listwise collapse.

2. **Anchored RankNet/BPR with scale guard.**
   - Cells: `1 loss x 2 models x 2 features x 5 folds x 5 seeds = 100` for pilot; expand to 10 seeds only if fold-4 gate passes.
   - Cost: about 3-5h pilot, 6-9h full.
   - Expected IC delta: -0.002 to +0.006; portfolio Sharpe may improve slightly if top/bottom pairs stabilize.
   - Complexity: medium, all-pairs per day but `n~500` is manageable.
   - Why: closest to CIKM 2025 RankNet/BPR evidence while explicitly fixing our pairwise scale-collapse mechanism.

3. **Regularization/optimizer pilot on MSE and Huber.**
   - Cells: `4 configs x 2 losses x 2 models x 2 features x 5 folds x 3 seeds = 480` if full grid; cheaper staged pilot should start with MLP/S6 and SAGE/S8 only.
   - Cost: 6-10h staged, depending on breadth.
   - Expected IC delta: +0.003 to +0.008 if overfitting residual is real.
   - Complexity: low.
   - Recommended configs: AdamW, `weight_decay={3e-4,1e-3}`, `lr={5e-4,1e-3}`, `patience=5`.

### Tier 2 (Medium Effort)

4. **Group-DRO wrapper on Huber and anchored RankNet.**
   - Cells: `2 base losses x 2 models x 2 features x 5 folds x 3 seeds = 120`.
   - Cost: 4-7h plus implementation time.
   - Expected IC delta: average +0.000 to +0.004; worst-fold improvement could be larger, around +0.01 to +0.03, if historical groups resemble fold 4.
   - Complexity: medium-high because `train_one_v2()` must aggregate day losses by group.
   - Why: it is the most direct response to "fold-4 is anomalous regime."

5. **Anchored soft Spearman IC.**
   - Cells: `1 loss x 2 models x 2 features x 5 folds x 3 seeds = 60`.
   - Cost: 2-4h plus `torchsort` install risk.
   - Expected IC delta: -0.003 to +0.004.
   - Complexity: medium due dependency and smoothing hyperparameter.
   - Why: useful diagnostic for surrogate mismatch, but I would not expect it to beat robust MSE consistently under 21d noise.

### Tier 3 (Speculative High-Payoff)

6. **Tempered ListNet.**
   - Cells: `1 loss x 2 models x 2 features x 5 folds x 3 seeds = 60`.
   - Cost: 2-4h.
   - Expected IC delta: -0.010 to +0.005.
   - Complexity: low.
   - Why speculative: it is the cleanest "listwise but not ListMLE" test, but it shares the softmax mechanism most likely to overlap with fold-4 collapse.

7. **MSAM with Huber/MSE.**
   - Cells: `1 optimizer variant x 2 losses x 2 models x 2 features x 5 folds x 3 seeds = 120`.
   - Cost: 4-7h if MSAM is implemented cleanly; full SAM would be roughly 2x and is not worth it yet.
   - Expected IC delta: +0.000 to +0.004; larger expected effect on fold variance than mean IC.
   - Complexity: medium.
   - Why speculative: NeurIPS 2025 MSAM is new and I found no verified finance cross-sectional application.

8. **RC-Mixup / same-day C-Mixup.**
   - Cells: start with `Huber + MLP/S6 + SAGE/S8 x 5 folds x 3 seeds = 30`.
   - Cost: 1-2h plus data-path complexity.
   - Expected IC delta: -0.004 to +0.003.
   - Complexity: medium-high.
   - Why low priority: current features are factor tensors, not raw sequences; mixup can blur portfolio-relevant extremes.

The experimental score for every tier should be reported as:

$$
Report = \{\overline{IC},\ \Delta IC\text{ vs MSE},\ Sharpe,\ \Delta pred\_cs\_std,\ IC_{fold4},\ sd_{fold}(IC)\}.
$$

This keeps the supplementary work tied to the Stage 1 evidence and to the published methods it borrows from: robust regression from Huber/Tukey, pairwise ranking from RankNet/BPR, Group-DRO from Sagawa et al. ICLR 2020, soft rank from Blondel et al. ICML 2020, SAM/MSAM from Foret et al. ICLR 2021 and Becker et al. NeurIPS 2025, and mixup from C-Mixup/RC-Mixup.

Practical runner pseudocode:

```python
SUPPLEMENTAL_EXPERIMENTS = [
    ("huber", {"delta": 1.0}, {"lr": 1e-3, "dropout": 0.3, "weight_decay": 3e-4}),
    ("tukey", {"c": 2.0}, {"lr": 1e-3, "dropout": 0.3, "weight_decay": 3e-4}),
    ("anchored_ranknet", {"tau": 0.25, "alpha": 0.5, "top_frac": 0.2}, {"lr": 5e-4}),
    ("anchored_soft_ic", {"alpha": 0.5, "regularization_strength": 1.0}, {"lr": 5e-4}),
    ("listnet_temp", {"tau_y": 1.5, "tau_s": 1.0, "y_clip": 2.0}, {"lr": 5e-4}),
]

for loss_type, loss_kwargs, hp_over in SUPPLEMENTAL_EXPERIMENTS:
    hp_run = default_hparams() | hp_over
    # Reuse _single_run(...) and existing result schema, but write to a new
    # supplementary CSV so the preregistered Stage 1 result remains untouched.
    result = _single_run(
        state, feat_np, feat_name, fold, model_type, seed,
        loss_type, loss_kwargs, hp_run,
        preds_dir=PROJECT_ROOT / "experiments/loss_horserace/preds_supplemental",
    )
```

## Summary Action List

1. Implement `huber`, `tukey`, `trunc_mse`, and `anchored_ranknet` in `run_loss_horserace.py` using the existing `LOSS_REGISTRY` pattern.
2. Run a 5-seed robust pointwise sweep and a 5-seed anchored RankNet pilot. Do not use mean IC alone; apply the fold-4 viability gate.
3. If anchored RankNet passes scale and fold-4 gates, expand it to 10 seeds. If it scale-collapses, stop pairwise ranking and report that the hinge-collapse mechanism generalizes.
4. Add Group-DRO only after a base loss is stable. Use calendar/regime groups computed from training data only.
5. Treat soft Spearman and tempered ListNet as diagnostics. They are useful for mechanistic discussion, but lower expected ROI for OOS improvement than robust pointwise + regularization.
6. Keep all new results supplementary. The Stage 1 preregistered 0/8 verdict remains unchanged.

Key cited sources:

- Burges et al., "Learning to Rank using Gradient Descent", ICML 2005: https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/
- Cao et al., "Learning to Rank: From Pairwise Approach to Listwise Approach", ICML 2007: https://www.microsoft.com/en-us/research/?p=153086
- Xia et al., "Listwise Approach to Learning to Rank: Theory and Algorithm", ICML 2008.
- Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009.
- Wang et al., "The LambdaLoss Framework for Ranking Metric Optimization", CIKM 2018, DOI:10.1145/3269206.3271784: https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/
- Zhang and Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels", NeurIPS 2018, arXiv:1805.07836: https://papers.nips.cc/paper/8094-generalized-cross-entropy-loss-for-training-deep-neural-networks-with-noisy-labels
- Wang et al., "Symmetric Cross Entropy for Robust Learning With Noisy Labels", ICCV 2019, arXiv:1908.06112: https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.html
- Ye et al., "Active Negative Loss Functions for Learning with Noisy Labels", NeurIPS 2023: https://papers.nips.cc/paper_files/paper/2023/hash/15f4cefb0e143c7ad9d40e879b0a9d0c-Abstract-Conference.html
- Blondel et al., "Fast Differentiable Sorting and Ranking", ICML 2020, arXiv:2002.08871: https://icml.cc/virtual/2020/poster/53612
- Engilberge et al., "SoDeep: A Sorting Deep Net to Learn Ranking Loss Surrogates", CVPR 2019, arXiv:1904.04272: https://openaccess.thecvf.com/content_CVPR_2019/html/Engilberge_SoDeep_A_Sorting_Deep_Net_to_Learn_Ranking_Loss_Surrogates_CVPR_2019_paper.html
- Sagawa et al., "Distributionally Robust Neural Networks for Group Shifts", ICLR 2020: https://openreview.net/forum?id=ryxGuJrFvS
- Duchi and Namkoong, "Learning Models with Uniform Performance via Distributionally Robust Optimization", Annals of Statistics 2021, arXiv:1810.08750: https://arxiv.org/abs/1810.08750
- Sinha, Namkoong, Duchi, "Certifying Some Distributional Robustness with Principled Adversarial Training", ICLR 2018, arXiv:1710.10571: https://openreview.net/forum?id=Hk6kPgZA-
- Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021, arXiv:2010.01412: https://arxiv.org/abs/2010.01412
- Becker et al., "Momentum-SAM: Sharpness Aware Minimization without Computational Overhead", NeurIPS 2025: https://openreview.net/forum?id=XyCDB1Uiqa
- Yao et al., "C-Mixup: Improving Generalization in Regression", NeurIPS 2022, arXiv:2210.05775: https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html
- Hwang, Kim, Whang, "RC-Mixup: A Data Augmentation Strategy against Noisy Data for Regression Tasks", KDD 2024, arXiv:2405.17938: https://openreview.net/forum?id=bVDTckZMxw
- Chen et al., "FrAug: Frequency Domain Augmentation for Time Series Forecasting", ICLR 2023, arXiv:2302.09292: https://arxiv.org/abs/2302.09292
- Kwiatkowski and Chudziak, "On Evaluating Loss Functions for Stock Ranking: An Empirical Analysis with Transformer Model", CIKM 2025, DOI:10.1145/3746252.3760812, arXiv:2510.14156: https://arxiv.org/abs/2510.14156
