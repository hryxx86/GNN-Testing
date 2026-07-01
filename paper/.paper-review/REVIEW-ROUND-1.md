# PaperJury — Review Round 1 (merged + adjudicated)

Manuscript: `paper/main.tex` · Engine: review-engine-v3 (one round) · Reviewers: 3 isolated ICAIF/AAAI-tier domain panels (R1 statistics, R2 GNN, R3 quant-finance) via `finance-gnn-reviewer`.
Raw weaknesses: 35 (R1×14, R2×10, R3×11). Merged below by theme; `raised_by` = corroboration count.

Disposition legend:
- **FIX-TEXT** = valid-fixable by editing existing text (hedge / clarify / disclose), no new data. Apply with author sign-off.
- **VERIFY-THEN-FIX** = needs the actual project convention confirmed from protocol/code (not fabricated), then a one-sentence clarification.
- **QUEUE (author-required)** = needs a new experiment / author decision; not fixable in this pass.
- **DROP** = not valid on inspection.

---

## MAJOR — substantive (consensus first)

### M1. Family-2 "0/6 survive" is a power artifact of an n≈12 bootstrap, risks being over-read as "no edge effect"
- raised_by: 4 (R1-05, R1-10, R2-07, R3-06) — strongest consensus.
- evidence: "block bootstrap over folds, and BH-FDR over the six contrasts ... $n\approx12$ fold blocks"; "6/6 are underpowered".
- disposition: **FIX-TEXT** (the paper already hedges "fail-to-reject ≠ no-effect"; strengthen so the causal-family conclusion is stated as *uninformative-by-power*, not as "edges are harmless"; soften the Discussion line "even the one edge that looked harmful is not harmful at fixed capacity") + **VERIFY-THEN-FIX** (report bootstrap variant / block length over folds / n_replications, which §4.5 omits).

### M2. Sharpe annualization √(252/21) vs the 21-day overlapping return series is unreconciled
- raised_by: 3 (R1-09, R3-03, R3-10).
- evidence: "The long-short Sharpe is annualised by $\sqrt{252/21}$."
- disposition: **VERIFY-THEN-FIX** — state precisely whether the L/S return series is daily-overlapping or non-overlapping 21-day, justify √12 (or correct the SE/CI for overlap), and reconcile with the block=21d IC bootstrap. Feeds the load-bearing net-Sharpe numbers, so must be pinned down even though net Sharpe is "descriptive".

### M3. C/L5s (27.5% undefined IC) carried as 1 of M=9 SPA candidates; headline imputation + M not stated
- raised_by: 3 (R1-08, R2-06, R2-10).
- evidence: "enters only the SPA candidate set"; table "L5s ... 0.0018" vs "mathematically undefined ... treated as missing".
- disposition: **FIX-TEXT** — state which imputation the headline M=9 SPA uses, whether M would drop to 8 under exclude, and that Table `tab:ic` IC=0.0018 is conditional-on-defined cells (label it). Robustness numbers already exist (§4.6).

### M4. Universe-C feature basis chosen by a leak-affected procedure (Plan-AAA, 5/15 survive T−1) — selection-stage look-ahead
- raised_by: 2 (R1-11, R3-08).
- evidence: "only 5 of the top 15 stay in the top 15"; Universe-C carries most load-bearing rejections.
- disposition: **FIX-TEXT** (sharpen L1: state the selection-stage leak explicitly, and that pre-registering the protocol does not neutralize a basis chosen under leakage; soften Universe-C confirmatory weight) + **QUEUE** (re-run the IC ladder on a leak-free top-15 basis to show invariance — author-required).

### M5. L6 / L7 representativeness overstated
- raised_by: 2 (R2-01, R2-02).
- evidence: "L6 ... attention vs structure (dense)" as sole stand-in for AD-GAT/MASTER; "HATS-3R-adapt ... restricts the relations to ... 3" vs original 75.
- disposition: **FIX-TEXT** — reframe L6 as "a controlled dense-attention rung (GAT without the correlation mask), not a reimplementation of MASTER/AD-GAT" and L7 as "a 3-relation HATS-style rung, not full 75-relation HATS"; note the relation-starvation caveat for the L7−L2 reading. (L6 limitation already partly says this — extend.)

### M6. Family-2 "fixed capacity" = fixed L2 hyperparameters ≠ fixed capacity; HP mismatch handicaps the denser edge sets
- raised_by: 1 (R2-03).
- evidence: "freezes the full hyperparameter vector of the correlation-graph operating point and varies only the edge set, so the matched ΔIC isolates the pure edge effect at fixed capacity".
- disposition: **FIX-TEXT** — rename to "fixed *operating point*" (not "fixed capacity"), and add the caveat that L2-tuned HPs may underfit denser edge sets, so the matched-ΔIC isolates "edge effect at the L2 operating point", which the paper already says in §3.3 but contradicts in the abstract/intro wording.

### M7. Seed-averaged daily ΔIC for DM understates uncertainty vs the intro's cross-seed-dispersion thesis
- raised_by: 1 (R1-06).
- disposition: **FIX-TEXT** — disclose the seed-averaging choice for DM/SPA and acknowledge it removes cross-seed variance from the inference (internal tension with §1); or justify why day-level is the inference unit.

### M8. BH-FDR over the pooled 20-test family ignores cross-universe dependence
- raised_by: 1 (R1-02).
- disposition: **FIX-TEXT** — state the dependence assumption (PRDS) or note BH-Yekutieli robustness, and that the CSV also supports per-universe BH (report both); the per-universe rejection set is already computed.

### M9. MDE=2.8×SE constant + HLN-on-HAC stated without assumptions
- raised_by: 1 (R1-03, R1-04 grouped).
- disposition: **FIX-TEXT** — state MDE 2.8 = z_{.975}+z_{.80} (two-sided 80% power z-test) and its assumptions; clarify the HLN(t_{T−1}) vs NW-HAC roles so they are not read as double-counting.

### M10. Survivorship / point-in-time membership + delisting returns not stated
- raised_by: 1 (R3-01) — but a first-order finance acceptance gate.
- evidence: "the S&P 500 constituents as of each trading day".
- disposition: **VERIFY-THEN-FIX** — confirm from the project data pipeline whether membership is genuinely PIT and how delisting/dropped names enter the 21-day label and L/S; add one explicit sentence. If not PIT, this is **QUEUE** (a real threat).

### M11. Label "next-day 21-day forward" + execution lag ambiguous
- raised_by: 1 (R3-02).
- disposition: **VERIFY-THEN-FIX** — pin down the target window (T→T+21 vs T+1→T+22) and the 1-day execution lag from the locked label definition.

### M12. Transaction-cost arithmetic (turnover_L1 definition, one-way vs round-trip) under-specified
- raised_by: 1 (R3-04).
- disposition: **VERIFY-THEN-FIX** — define turnover_L1 (Σ|Δw| vs half) and confirm one-way vs round-trip; the net-Sharpe sign margins (C L2−L1 −0.72, C L3−L2 +0.08) are within a 2× cost factor.

### M13. Mechanism transfer: L5s (SAGE + dense + dropout) collapse used to explain L2 (GAT + sparse) underperformance
- raised_by: 1 (R2-04).
- disposition: **FIX-TEXT** — soften "the C/L5s collapse naming the candidate mechanism" → "suggests a candidate mechanism"; note it is demonstrated on a different model/graph, not proven for L2.

### M14. Tuning parity is dimensional (6-D, 30 trials), not power parity
- raised_by: 1 (R2-05).
- disposition: **FIX-TEXT** (add caveat that equal-budget = equal trial/dimension count, not equal search adequacy) + **QUEUE** (a trials-sensitivity check is a new run).

### M15. Local DM rungs (BH-reject) elevated to abstract while SPA fails to reject — over-claim risk
- raised_by: 1 (R1-07).
- disposition: **FIX-TEXT** — already heavily hedged ("local ladder rungs, not global SPA wins"); optionally soften abstract verbs ("beats" → "outranks locally"). Low urgency.

---

## MINOR / mechanical (polish track)

- m16 (R1-12) **FIX-TEXT**: report SPA p_lower/p_consistent/p_upper bracket for the C=0.077 boundary case.
- m17 (R1-13) **FIX-TEXT**: one line reconciling the three effective sample sizes (T=749 days / n_eff≈36 / Family-2 n≈12) and which governs which test.
- m18 (R1-14) **FIX-TEXT**: clarify whether SPA(vs L0) and the DM ladder(vs L2) are one controlled family or two, and the joint-error stance.
- m19 (R2-08) **QUEUE**: |ρ|>0.6 / 126-d graph threshold has no sensitivity ablation.
- m20 (R2-09) **FIX-TEXT**: related-work gaps (StockMixer AAAI'24, THGNN/MDGNN); add 1–2 citations + soften "first to combine".
- m21 (R3-05) **VERIFY-THEN-FIX**: state the news source + publication-timestamp provenance.
- m22 (R3-07) **FIX-TEXT**: the level net-Sharpe (−0.22 / +0.95) has no CI; either add one or restate as a paired-only claim.
- m23 (R3-09) **FIX-TEXT**: caveat on equal-weight decile liquidity + short-borrow cost.
- m24 (R3-11) **FIX-TEXT**: note Family-2 is evaluated in IC only, not carried to net economics (consistency with §5.3).

---

## Summary

- 15 MAJOR (after dedup) + 9 minor. raised-by≥2: M1(4), M2(3), M3(3), M4(2), M5(2).
- **FIX-TEXT now (hedge/clarify, no new data):** M1(part), M3, M4(part), M5, M6, M7, M8, M9, M13, M14(part), M15, m16, m17, m18, m20, m22, m23, m24.
- **VERIFY-THEN-FIX (confirm convention from project, then 1 sentence):** M1(boot params), M2, M10, M11, M12, m21.
- **QUEUE (new experiment / author decision):** M4(leak-free re-run), M14(more trials), m19(threshold ablation); and M10 escalates here if membership is not PIT.
- No DROP — all 35 are at least partially valid; none was found factually wrong on inspection.

Author sign-off required before any manuscript edit (hard rule 1).

---

## DISPOSITION (applied 2026-06-26, author-signed)

H博士 signed off on the full fix scope (M10 → measure-and-disclose via Wikipedia audit; M4 → disclose-and-downgrade; M2/M12/conf-net → verified-correct, text-clarify only). All edits applied to `paper/main.tex` + `paper/references.bib`. Verification: brace/cite/ref/env/dollar all balanced; every introduced number cross-checked vs source CSV; 4 pre-existing figure-caption brace bugs found and fixed (see note at end).

| ID | Severity | Layer | Disposition | Resolution + location |
|----|----------|-------|-------------|------------------------|
| M1 | MAJOR | 2 | FIX-TEXT | §causal: "not harmful" → uninformative-by-power (fail-to-reject bounds effect, not harmless); + Family-2 bootstrap detail (stationary block bootstrap, 5000 reps, fold-block unit). `main.tex` §3.2 + §causal. src: compute_fc_edge_causal.py:54-55 |
| M2 | MAJOR | 1 | VERIFIED→FIX-TEXT | L/S Sharpe is non-overlapping 21d → √(252/21) matched; distinct from daily-IC block bootstrap. `main.tex` §3.2. src: run_storya_e1_anchor.py:809,853 (annualization correct, not a bug) |
| M3 | MAJOR | 1 | FIX-TEXT | M=9 exclude-pointwise imputation for C/L5s undefined days (M not dropped to 8); tab:ic caption labels IC=0.0018 conditional-on-defined. §headline + tab:ic caption + §stability. src: compute_family1_ladder.py:68,95-119 |
| M4 | MAJOR | 0 | DISCLOSE (a) + QUEUE | Universe-C positive (MLP>LGB, FC edges) → suggestive; null/negative unaffected (selection leak only inflates signal). Abstract/§headline/§exploratory/Limitation L1. QUEUE: leak-free re-run |
| M5 | MAJOR | 2 | FIX-TEXT | L6 = controlled dense-attention proxy (not MASTER/AD-GAT reimpl); L7 = 3-relation HATS-style (not full 75) + relation-starvation caveat. §related work |
| M6 | MAJOR | 2 | FIX-TEXT | "fixed capacity" → "operating point (L2 HPs)" in abstract/intro; + underfit caveat (0/6 may be partly FC underfit). abstract/§1/§3.2 |
| M7 | MAJOR | 1 | FIX-TEXT (freeze) | Seed-averaging disclosed (inference-unit note): day is the pre-registered inference unit; cross-seed dispersion not in family tests. §Power. Method change would break freeze md5 59ddd0a2. src: compute_e6_dm_spa.py:156-172 |
| M8 | MAJOR | 1 | FIX-TEXT | Pooled 20-test BH = PRDS; per-universe BH agrees on EVERY contrast (verified family1_dm_hln.csv). §Family-1. src: compute_family1_ladder.py:239-246 |
| M9 | MAJOR | 1 | FIX-TEXT | MDE 2.8 = z₀.₉₇₅+z₀.₈₀ derivation + assumptions; HLN vs NW-HAC adjust different quantities (not double-counted). §Power + §Family-1 |
| M10 | MAJOR | 0 | DISCLOSE + MEASURED | Wikipedia audit: names gap 14.8%, survivorship 8.2%, look-ahead 8.1%, composition mismatch 16.3% (<20% → no escalation). Methods §3.1 conditional-estimand + Limitation L8. src: analyze_m10_universe_gap.py; Rule9 TP2(2 FIXED)+TP3(2 FIXED+3 specs). QUEUE: PIT rebuild (not triggered) |
| M11 | MAJOR | 1 | VERIFY→FIX | Label window T→T+21 close-to-close + 1-day exec lag (features at T-1); fixed "log return" → simple market-excess return. §3.1. src: run_storya_e1_anchor.py:425 |
| M12 | MAJOR | 1 | VERIFIED→FIX-TEXT | turnover_L1 = Σ\|Δw\| (=4 full rotation), one-way (L1_one_way). §3.2. src: run_storya_e1_anchor.py:838,863,886 (definition clear, not a bug) |
| M13 | MAJOR | 2 | FIX-TEXT | C/L5s collapse "names" → "suggests" candidate mechanism; demonstrated on different arm (SAGE+dense), not proven for L2. §Discussion + §stability. QUEUE: L2 over-smoothing test |
| M14 | MAJOR | 2 | FIX-TEXT + QUEUE | equal-budget = equal trial/dim count, not search adequacy; caveat on "MLP beats GAT". §3.1. QUEUE: trials sweep |
| M15 | MAJOR | 2 | FIX-TEXT | abstract "beats" → "outranks locally". abstract |
| m16 | minor | 3 | FIX-TEXT | SPA C bracket p_lower=0.055/p_cons=p_upper=0.077 (even lower bound ≥5%); no near-miss; underpowered. §spa. src: family1_spa.csv |
| m17 | minor | 3 | FIX-TEXT | Three eff sample sizes (T=749 / n_eff≈36 / Family-2 n≈12) reconciled. §Power |
| m18 | minor | 3 | FIX-TEXT | SPA(vs L0) and DM ladder(vs L2) = two separate inferential layers, no joint family-wise rate claimed. §spa |
| m19 | minor | QUEUE | QUEUE | \|ρ\|>0.6 / 126d sensitivity → Limitation L9 future work |
| m20 | minor | 3 | FIX-TEXT | Cited StockMixer (AAAI'24, MLP-only) + MDGNN (AAAI'24); "first to combine" → "not aware of prior work". §related work + references.bib (real BibTeX, WebSearch/WebFetch-verified) |
| m21 | minor | 3 | VERIFY→FIX | News source = EODHD news feed; publication-timestamp gate before NYSE close T-1. §3.1. src: plan.md:461 |
| m22 | minor | 3 | FIX-TEXT | Level net-Sharpe CIs added (C/L0 −0.22 [−0.82,+0.36]; C/L1 +0.95 [+0.18,+1.77]); paired ΔSharpe is load-bearing. §cost. src: cost_ladder_by_arm.csv |
| m23 | minor | 3 | FIX-TEXT | Equal-weight decile liquidity + short-borrow caveat → Limitation L5 |
| m24 | minor | 3 | FIX-TEXT | Family-2 evaluated in IC only, not carried to net economics. §cost |

**QUEUE (non-blocking, future work):** M4 T-1-clean re-run; M13 L2 over-smoothing direct test; M14 trials-sensitivity sweep; m19 graph-threshold/window ablation; M10 PIT universe rebuild (NOT triggered — gap 14.8% < 20%).

**Pre-existing bug found+fixed during verification (NOT a PaperJury finding):** 4 figure captions (`fig:headline`, `fig:spa`, `fig:regime`, `fig:cost`) were missing the `\caption{...}` closing brace (only `\emph{...}` closed) — identical brace deficit Δ4 in the committed baseline `eac6063`, so not introduced this session. Would have broken the first Overleaf compile (runaway `\caption` argument). Fixed; brace balance now Δ0.

All 15 MAJOR + 9 minor: applied (FIX-TEXT/DISCLOSE/VERIFY-FIX) or QUEUE'd. None dropped.

---

## ROUND-1 CLOSED (2026-06-27)

Three author-specified closeout tasks, all passed:

1. **M10 composition-mismatch character (追加 1).** The 16.3% two-sided mismatch was characterised, not waved through under the <20% line: survivorship side = 54/87 cap-change (index small edge) + 28 M&A + 5 other (0 micro-cap); look-ahead side = 76 mid-window additions, smallest market cap \$6.6B, **none below \$5B** (source `artifacts/audits/m10_universe_gap.{csv,md}` via `analyze_m10_universe_gap.py`). Both sides are S&P 500 liquidity-scale boundary names → no small-cap/illiquid contamination. Limitation L8 rewritten into **two separate parts** — (i) survivorship gap (contrast argument) and (ii) composition mismatch (boundary biased-sampling, NOT contrast-cancelled) — with the liquidity characterisation quantified; the gap口径 and the mismatch are described separately, and no "all <20% so disclose" sentence is used.

2. **Wording-consistency grep sweep (追加 2).** Swept `M=9|0.077|0.0018|beats|MLP>LGB|FC edge|suggestive|near-miss|M=8` across `paper/ docs/ artifacts/`. Submission + current-state docs: **zero red-line violations** (all "near-significant" hits are the discipline-enforcing "NEVER near-significant" rule or unrelated exploratory Plan-AAA factor text). SPA口径 uniform: M=9 ×4, M=8 ×0, C/L5s IC=0.0018 labelled conditional-on-defined everywhere. M4 downgrade: "suggestive" ×6 across abstract/headline/§spa/§causal/L1. Two fixes from the sweep: (a) added the missing M4 suggestive pointer to §5.2 (the one place re-asserting "MLP beats LightGBM" without it); (b) tightened `docs/analysis.md` "C marginal" → "fail-to-reject, not a near-miss". Figure gallery already bakes the red line ("C's 0.077 … never near-significant").

3. **End-to-end compile (流程 3).** Installed `tectonic` 0.16.9 (Overleaf-equivalent engine, auto-fetches acmart). Full `paper/main.tex` + real `figures/*.pdf` + `references.bib` compiles to an **11-page PDF, 0 errors**. The 4 figure-caption brace fixes are confirmed (body + all 8 figures + appendix typeset). **A 5th pre-existing latent bug was found and fixed that brace-balance could not catch:** `sheppard_arch` had an empty `year` field → ACM-Reference-Format's `[n.\,d.]` no-date path broke math mode in the bibliography (`main.bbl:288 Missing $ inserted`), halting before any PDF. Added `year = {2024}` (software access year) → clean compile. This is exactly the "balance-passes ≠ compiles" case the closeout guarded against.

**Status: CLOSED.** No open Round-1 items. (Page budget: 11pp vs 8–10pp target — a trimming task for the author, not a Round-1 finding.)

### Round-2 confirmation (2026-06-27)

Re-ran the 3-panel protocol on the edited paper (`artifacts/reviews/2026-06-27_paperjury_round2.md`). All 3 panels **PASS-WITH-CONCERNS, 0 CRITICAL, 1 MAJOR**. Round-1's 24 findings all confirmed closed (M7 PARTIAL→CLOSED). All red lines held; every load-bearing number independently re-verified vs source CSV with **zero errors introduced by the Round-1 edits**. The 1 MAJOR (qf-01, L8 liquidity-vs-distress benignity) + 4 CONCERN (gnn-02 capacity-matched prose, qf-02 source caveat, stat-03 BH p-column, stat-01 §1 forward-pointer) were **fixed this round**; 5 optional clarity CONCERNs deferred (add length to 11pp paper, no correctness impact). Post-fix: brace Δ=0, end-to-end compile clean. **Round-1 + Round-2 both CLOSED.**
