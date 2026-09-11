# 任务：C5 leak-free re-selection sensitivity（给实验机 Claude Code 的说明）

> 来源：H博士 2026-09-10 交付的任务简报（原文保留；§9 为实验机 Claude 的实施注记，供 Rule 9 Touchpoint 1 评审）。仓库 = GNN-Testing（Colab 上在 `/content/GNN-Testing`，`experiments/`、`data/` 软链到 Drive `GNN测试`；本地 Mac 副本在 `~/Desktop/GNN-Testing`）。下面的行号来自 2026-09-10 的本地副本，可能略有漂移，以 grep 为准。

## 0. 背景与目标（先读）

ICLR 2027 稿的审稿意见 2：Limitation L1 说宇宙 C 的正结果（MLP − LightGBM，即 C 的 L1−L0 = +0.0148，HLN p = 0.011，BH 拒绝）因为特征基选择泄漏只算 "suggestive"，但论文没有给出泄漏把这个对比吹大了多少。论文现在的处理是用无泄漏的宇宙 B 做锚点（B 的 L1−L0 = +0.0143，p = 0.052），并在 L1 里承诺："A re-run of L0 and L1 on the five surviving factor groups is the definitive check, and we have not run it."

**本任务就是把这个 re-run 跑出来**：在宇宙 C 里只保留 T−1 re-rank 存活的 5 个因子组（记作 **C5**），用与 confirmatory run 完全一致的协议重新调参并评估 L0（LightGBM）和 L1（MLP），报告 C5 上的 L1−L0。这是 **post-hoc sensitivity**：不进入两个 confirmatory family，不新开 BH 族，报告原始 HLN p 值。

## 1. C5 的定义（先核对再跑）

- 存活组来源：`artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv`，`proxy_rank_t1 <= 15` 的 5 组：**ROC30、KMID、KUP、CNTP20、CORR60**（`summary.md` 结论 "LOW STABILITY … 5/15"）。
- 组 → 列映射：`artifacts/plan_aaa/ranking.csv` 的 `group_members` 列（`scripts/analyze_plan_aaa_t1_diagnostic.py` 第 56、155 行附近加载）。宇宙 C 的 51 列硬编码在 `run_storya_e1_anchor.py:160-176`（`UNIVERSE_C_ALPHA158_NAMES` 48 列 + `UNIVERSE_C_EXTRA_NAMES` 3 列 hc 列）。
- 预期 C5 = **20 列**：ROC30, MA60, MAX60, MIN60, QTLU60, QTLD60 | KMID, KMID2, KSFT, KSFT2, OPEN0, HIGH0, VWAP0 | KUP, KUP2 | CNTP20, CNTD20, CNTP30, CNTD30 | CORR60。
- **不含**那 3 个 hc 列（`hc_mom12m`、`hc_ret_std_5d` 等）。它们在 T−1 audit 里被 `na_option="bottom"` 钉在 rank 58，`summary.md` 说它们不受 alpha158 泄漏影响，但论文 L1 的原话是 "five surviving factor groups"，主结果必须严格对应这句话。含 hc 的 23 列变体（C5h）只作为可选的第二步，主结果出来以后再说。
- 第一步先用脚本打印 C5 的列名，与 `ranking.csv` 逐组核对，把核对结果（20 列清单）写进 `progress.md` 新条目，再往下跑。

## 2. 协议冻结（必须与 confirmatory 一致，一项都不能改）

- 同 12 折 expanding walk-forward 日历、同 21 天 purge、同 label（21 日前向收益，横截面去均值 + z-score）、同 10 个 canonical seeds。
- 同调参设置：`run_storya_v21_tune.py --n-trials 30 --top-k 5`，固定 `TUNE_SEEDS=[11,22,33]`，单一调参窗 train_end 2022-06-30 / val_end 2022-12-31（脚本第 87-88 行）。这个窗全在测试期（2023Q1–2025Q4）之前，不要动。
- **L0 和 L1 都必须在 C5 上重新调参**，不能复用 `experiments/storya_v21_tune/C_L0.json`、`C_L1.json`（特征维度变了，等预算规则要求各臂独立调 30 trials）。
- 调完冻结：生成 `frozen_hparams_c5.json`，记录 md5，走 `run_storya_v21_main12.py` 的 `_frozen_hp_provenance.json` 校验（第 642-653 行会拒绝 mode/md5 不匹配）。
- 注意 `experiments/storya_v21_tune/B_L2.json` 现在是 90-trial 版本，30-trial 在 `B_L2.json.30trial.bak`；本任务不碰 B，但别误用。

## 3. 代码改动清单（新增一个 universe 要动的地方）

1. `run_storya_e1_anchor.py:77` `ALL_UNIVERSES` 加 `"C5"`；在 `build_universe_C`（:357）旁边加 `build_universe_C5`：直接调用 `build_universe_C` 然后按第 1 节的 20 列做列选择（不要重新造特征，保证数值与 C 逐列一致）。
2. `run_storya_v21_main12.py:126` `ALL_UNIVERSES`；`:625` `universe_idx_map`；`:667-672` universe 分发的 if/else；`:544` argparse choices。
3. `run_storya_v21_tune.py:177-181` 分发；`:301` argparse choices。
4. `run_storya_v21_main12.py:180-187` 的 `cell_id = universe*1200 + …` 假设只有 2 个 universe，给 C5 分配 index 2 并断言 cell_id 与已有 2160 行无碰撞。
5. frozen key 用 `f'{universe}_{arm}'`（main12:157）；`run_v21_tune_launcher.py:149-152` merge 默认要求 20/20 studies，对 C5 只有 2 个 study，用 `--allow-partial` 或改 expected。
6. `compute_family1_ladder.py:56` `UNIVERSES` 加 `"C5"`。`LADDER_PAIRS`（:63）是冻结的，文件头写着 "NO adding pairs"，**不要加 pair**；L1−L0 本来就在列表里。
7. `run_storya_v21_l7_hats.py` 不用改（本任务不跑 L7）。

## 4. 运行顺序

1. `--smoke`：1 折 1 seed 端到端跑通 tune → main12 → family1，确认输出目录和列数（20）。
2. 调参：`run_storya_v21_tune.py --arm L0 --universe C5 --n-trials 30 --top-k 5`，再 `--arm L1`。产出 `experiments/storya_v21_tune/C5_L0.json`、`C5_L1.json`。冻结、算 md5、写 provenance。
3. 主评估：`run_storya_v21_main12.py --universe C5 --arms L0 L1 --seeds <全部10个> --folds <全部12个> --frozen-hparams <frozen_hparams_c5.json> --out-dir experiments/storya_v21_main12_c5`。240 个 cell。检查 `results.csv` 无失败、无重复 cell_id，`per_day_ic/` 完整。
4. 统计：`compute_family1_ladder.py --main-dir experiments/storya_v21_main12_c5 --output-dir artifacts/storya_v21_family1_c5`，取 `family1_dm_hln.csv`（L1−L0 的 seed-averaged daily ΔIC 与 HLN p）、`family1_ic_ci.csv`（L0、L1 的 pooled IC + 95% block-bootstrap CI，block 21 天，5000 次）、`family1_mde.csv`（MDE = 2.8 × SE）。再跑出 per-seed 符号一致数 k/10 和 leave-one-seed-out 翻转数 m/10（复用生成论文附录 C.2 seed 表的脚本，`grep -rn "leave-one-seed\|LOSO" scripts/` 找）。**不做 BH**，报告原始 p。
5. 可选第二层（主结果出来后再决定）：L2（correlation-GAT）在 C5 上重调 + 评估（+120 cell），看 L2−L1 惩罚在无泄漏的 C 子集里是否还在。

## 5. 预期用时（来自 progress.md 记录的 wall time）

- L0 约 1 s/cell；L1 约 60–110 s/cell → 240 cell 约 2–4 小时；L0、L1 调参各不到 0.5 小时。
- 可选 L2：调参约 5 小时 + 120 cell 约 5–6 小时。
- Colab 的 tune sqlite 有回收风险，先 `scripts/colab_v21_tune_db_sync.py --backup`。结果回传走之前的 tar+scp 到 `experiments/_rerun_colab_staging/`，代码推送用 `scripts/sync_to_drive.sh`。

## 6. 交付物（原样报回，论文要直接引用这些数）

1. C5 的 20 列清单及与 `ranking.csv` 的核对结果。
2. `frozen_hparams_c5.json` 的 md5 和两臂调出的超参。
3. C5 上 L0、L1 的 pooled IC（seed-averaged）及 95% CI。
4. C5 上 L1−L0：seed-averaged daily ΔIC、HLN p、MDE@80%、per-seed 符号 k/10、LOSO 翻转 m/10。
5. 一行对比：C5 vs C（+0.0148, p=0.011）vs B（+0.0143, p=0.052）。
6. （若跑了）C5 上 L2−L1 同上一套指标。
7. `docs/analysis.md` 新条目（带日期编号，标题 "C5 leak-free re-selection sensitivity (post-hoc)"）+ `progress.md` 条目；按 `.gitignore` 白名单提交 `results.csv`、`per_day_ic/`、frozen json。
8. 不改动任何 confirmatory 表；不把结果称为 confirmatory。如果 C5 上的 L1−L0 变小或不显著，照实报告，这本身就是论文要的答案。

---

## 9. 实施注记（实验机 Claude，2026-09-10；偏离简报处逐条列出，供 TP1 评审 + H博士 裁决）

**9.1 C5 列清单核对（`artifacts/plan_aaa/ranking.csv` `group_members` 逐组）**

| 组（Plan AAA label） | Plan AAA rank | proxy_rank_t1 | 成员列 |
|---|---|---|---|
| ROC30+5 | 2 | 8 | ROC30, MA60, MAX60, MIN60, QTLU60, QTLD60 |
| KMID+6 | 4 | 13 | KMID, KMID2, KSFT, KSFT2, OPEN0, HIGH0, VWAP0 |
| KUP+1 | 10 | 6 | KUP, KUP2 |
| CNTP20+3 | 12 | 15 | CNTP20, CNTD20, CNTP30, CNTD30 |
| CORR60 | 15 | 9 | CORR60 |

合计 6+7+2+4+1 = **20 列**，与简报 §1 预期逐列一致；全部属于 `UNIVERSE_C_ALPHA158_NAMES`（48 列）子集，不含 hc 列。（source: `artifacts/plan_aaa_t1_diagnostic/group_ranking_comparison.csv` 列 `proxy_rank_t1`；`artifacts/plan_aaa/ranking.csv` 列 `group_members`）

**9.2 偏离 §3 第 1/2 条：不把 "C5" 加进 `ALL_UNIVERSES`，改为独立的 `SENSITIVITY_UNIVERSES = ['C5']`**

理由：`ALL_UNIVERSES` 在 anchor 与 main12 中都驱动 `--universe both` 的展开（anchor:1028、main12:614）和 `_meta.json` 的 `universes` 字段（anchor:910）。加进去会让 confirmatory 默认调用（`both`）静默多跑一个 universe，违反简报 §8 "不改动任何 confirmatory 表"。做法：`KNOWN_UNIVERSES = ALL_UNIVERSES + SENSITIVITY_UNIVERSES`，argparse choices 用 KNOWN，`both` 仍严格 = B,C。`--universe C5` 必须显式给出。

**9.3 偏离 §3 第 6 条：`compute_family1_ladder.py` 不改模块常量 `UNIVERSES`，改为 CLI `--universes`（默认 `B,C`）+ `--arms`（默认全 10 臂）+ `--sensitivity` 开关**

理由：脚本内 `cl5s_robustness` 硬编码 `('C','L5s')`、`degeneracy_report` 以 L2 为参考臂、SPA 需要全部候选臂——只有 L0/L1 的 C5 目录会 KeyError/空序列。`--sensitivity` 模式：跳过 SPA 与 C/L5s 稳健性；DM/HLN 行照常算但 **不写 BH 判定**（`BH_FDR_*` 列置空，`bh_fdr_q` 置 NaN），ledger `role` 标 "POST-HOC SENSITIVITY（非 confirmatory，不入 BH 族）"。默认参数下脚本行为与 2026-06-21 confirmatory 运行字节一致（会用 confirmatory 目录复跑 L1−L0 行核对 p=0.010852 复现）。`LADDER_PAIRS` 不动。

**9.4 `run_v21_tune_launcher.py --merge` 加 `--merge-universes/--merge-arms/--merge-out`**

默认 merge（B,C × 10 臂 = 20，写 `frozen_hparams.json`）行为不变，并新增按 universe 过滤 rows——否则 `C5_L0.json`/`C5_L1.json` 落到同一 OUT_DIR 后，默认 merge 会数出 22/20 而失败。C5：`--merge --merge-universes C5 --merge-arms L0,L1 --merge-out frozen_hparams_c5.json` → `expected=2, complete=true`，满足 main12 `load_frozen_hparams` 的完整性门。

**9.5 cell_id**：`universe_idx_map['C5']=2` → C5 cell_id ∈ [2400, 3599]，与 confirmatory [0, 2399] 无交集；`assert_cell_id_injective` 扩展到 3 universe（[0, 3599]，3600 个）并显式断言 C5 区间下界 2400 > 2399。

**9.6 运行设备（偏离 §5 "Colab"，待 H博士 裁决）**：confirmatory 的 C/L0、C/L1 共 240 cell 全部在 **Mac M4（MPS + CPU LightGBM）** 上跑（source: `experiments/storya_v21_main12_tuned_macC/results.csv`，cell_id 1200–1439，L1 平均 61 s/cell），C/L2–L6 才在 T4 上跑。本次 C5 vs C 是同臂对比，为消除设备差异这一混杂，C5 的 L0/L1 调参 + 240 cell **同样在 Mac 上跑**（预计 ≈2.7 h，与 T4 相当）；Colab T4 已就绪（依赖已装），留给可选的 L2 层（confirmatory C/L2 即在 T4）——L2 层按简报 §4.5 等主结果出来后由 H博士 决定是否启动。

**9.7 per-seed 符号 / LOSO**：新增 `analyze_c5_sensitivity.py`，复用 `analyze_paper_eval_robustness.seed_pooled`（n_test_days 加权的 per-seed pooled IC，与论文附录 C.2 同一估计量），对 (C5, L1, L0) 输出 k/10、m/10，并同法在 confirmatory C 上复算作为交叉核对（应复现 `artifacts/audits/paper_eval_robustness.csv` 行 0：10/10、0 flips）。一行对比表的 C/B 数字从 `artifacts/storya_v21_family1/family1_dm_hln.csv` 读取，不手抄。

**9.8 不做的事**：不跑 L7；不加 pair；不做 BH；不动 B；不动任何 confirmatory 目录/表；结果措辞一律 "post-hoc sensitivity"。

**9.9 TP1 Round A（Codex gpt-6-astra xhigh，verdict BLOCK-EXECUTION）处置 → 本次运行重新定性为 "test-informed feature-subset sensitivity"**

- **CODEX-A-01 CRITICAL — ACCEPTED（亲自核实）**。`analyze_plan_aaa_t1_diagnostic.py:86-99` 的 proxy 排名用面板"最后 313 个有效标签日"打分，本机重建 = **2024-09-27 → 2025-12-26**（脚本注释写的 "Q2-2024→Q2-2025" 与实际不符，`summary.md` caveat 3 已承认日历漂移），全部落在 12 折 confirmatory 测试期 2023Q1–2025Q4 内。而 Plan AAA 原排名本身在 5 折测试季 **2024-04-01 → 2025-06-30**（`data/reference/fold_manifest_expanding.json`，313 天 = 12 折 fold 5–9 的测试季）上算 permutation Δ-IC，同样在 confirmatory 测试窗内。**结论：C 与 C5 的列选择都使用了测试期标签。C5 不是 leak-free re-selection；本任务不能兑现论文 L1 "A re-run of L0 and L1 on the five surviving factor groups is the definitive check" 的承诺。** 处置：(i) 本次 C5 运行照跑（H博士 明确要求的数字仍有意义：**对重要性度量稳健的 20 列子集上 L1−L0 是否仍在**），但一律称 **"post-hoc, test-informed feature-subset sensitivity"**，结果措辞按 Codex 表格（四种结局各有可辩护表述），明确"不解决 feature-selection bias"；(ii) 论文 L1 / 附录 "definitive check" 句需撤回或改写——**paper-side 动作，报 H博士 裁决**；(iii) 真正 leak-free 的检查见 §9.10 提案。
- **CODEX-A-02 MAJOR — ACCEPTED（亲自核实）**。`group_ranking_comparison.csv`：`proxy_rank_raw<=15` 与 `proxy_rank_t1<=15` 的 15 组**集合完全相同**；orig∩raw = orig∩t1 = 同一 5 组（ROC30+5 / KMID+6 / KUP+1 / CNTP20+3 / CORR60）。即 T−1 shift 没有移除任何一组，5/15 是 **"Plan AAA permutation top-15 ∩ 单特征 |IC| proxy top-15"**——重要性**度量方法**之差，不是泄漏移除之果。论文 main.tex:290（L1）、:998、:1012（图 caption）的 "only 5 of the 15 groups survive strict T−1 re-ranking" 是误表述，**需改口径（报 H博士）**。C5 在本文档中一律描述为 "the intersection of Plan-AAA's top-15 with the single-feature-IC proxy top-15 (the proxy top-15 is identical with and without the T−1 shift)"。
- **CODEX-A-03 MAJOR — FIXED（代码）**。`analyze_c5_sensitivity.py::paired_contrast`：10 seed 先平均 → g_t = ΔIC_C,t − ΔIC_C5,t（同 749 日、同折序，全量时严格等长断言）→ HLN p（自动 lag + lag 21）+ 21d stationary bootstrap 95% CI（5000 次，直接重采样配对序列 g_t）。符号约定：**正 = C 的 L1−L0 优势大于 C5**。解读固定为"特征限制 + 重调后的变化"，不是泄漏膨胀的识别量。不进入冻结的 confirmatory pair 表。
- **CODEX-A-04 MAJOR — FIXED（代码）**。并列表（`c5_comparison.md`）已含 L1−L0 的 21d bootstrap CI、HLN p 自动 lag 与 lag-21 两档（C5 / C / B 及 paired 对比均报）；MDE 标注为 "approximate nominal（2.8 × bootstrap SE）"，不称精确功效。
- **CODEX-A-05 CONCERN — ACCEPTED-AS-CONCERN**。报告两臂冠军超参 + MLP 参数量（20 输入 vs C 的 51 输入，`analyze_c5_sensitivity.py` 用 `make_nn_model` 实算）；hc 排除 = 论文 "five surviving factor groups" 的字面定义，**不是** hc 组未通过泄漏审计的证据（它们在 proxy 里是 unscored，被 `na_option="bottom"` 钉底）。**C5h 在此预先指定**：C5h = C5 + `hc_mom12m, hc_ret_std_5d, hc_ret_std_10d`（23 列），流程与 C5 完全相同；**只在 H博士 要求时跑，不依据 C5 结果决定**。
- **CODEX-A-06 CONCERN — FIXED（代码）**。main12 的 C5 分支额外写 `_run_provenance.json`：git rev、平台、Python/torch/lightgbm/numpy/pandas 版本、device、有序特征表、selector 输入 md5（`group_ranking_comparison.csv`、`ranking.csv`、alpha158 meta/npy）、重建的选择日期窗、调参窗与 12 折日历、frozen md5。`experiments/storya_v21_tune/C5_{L0,L1}.json` 保留完整 top-5 决赛表；tune 的 sqlite study 备份到 `artifacts/storya_v21_tune/`（白名单 json 外另存 md5）。

**9.10 提案（option 2，Codex 建议）："C-pre" = 用 pre-test 信息重新选列 — 待 H博士 批准，本任务不启动**

- 选择器：对 168 个候选（158 Alpha158，T−1 shifted + 10 hc）在 **2021-07-01 → 2022-06-30**（= 调参窗 train 段；标签终点 ≤ 2022-06-30，purge 21 天，完全早于调参 val 2022H2 与测试期 2023Q1–2025Q4）算单特征日度 rank-IC 均值；按 Plan AAA 的 61 组定义（`artifacts/plan_aaa/ranking.csv` `group_members`；**需先核实组定义（相关性聚类）的形成窗口是否也在 pre-test**，否则组定义改在同一 pre-test 窗重聚类）取 mean|IC| 排名 → top-15 组 → 列集 C-pre（列数不定）。
- 之后与 C5 流程完全相同：L0/L1 各 30-trial 重调 → 240 cell → sensitivity 统计 + paired 对比。成本：选择器 < 10 min；其余同 C5（Mac ≈ 3 h）。
- 性质：仍是论文历史上的事后分析（post-hoc），但选择信息边界干净，可以兑现 "leak-free re-selection" 的字面承诺。需单独 TP1 评审；实施前不动任何代码。
