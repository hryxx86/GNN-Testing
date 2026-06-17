# artifacts/storya_v21_tune — v2.2 §4 等预算调参产物（FROZEN）

> 调参后超参的**权威冻结副本**（git 管理）。原始运行产物在 `experiments/storya_v21_tune/`（gitignored 数据目录）；此处是必须长期保全 + Colab 可拉取的子集。

## 当前内容

| 文件 | 说明 |
|---|---|
| `frozen_hparams.json` | **核心交付物**：20 个 `{universe}_{arm}` 的调参后冠军超参（含 L7 HATS）。`complete: true`, `n_studies: 20`。tuned 2160-cell 重跑（D-RERUN-12F）的 HP 来源。 |
| `{B,C}_{arm}.json` ×20 | 每 study 的完整输出：`winner_params` + `top_table`（top-5 决赛 × 3 调参 seed 的 val Rank IC，§4 regime-mismatch ammunition）+ `tune_window` + `n_trials`。 |

## 调参协议（docs/protocol_v2_freeze.md v2.2 §4）

- 每 study = 一个 arm × universe。Optuna TPE，N=30 单 seed 搜（seed 11）→ top-5 决赛 × 3 调参 seed [11,22,33] → 冠军 = max 3-seed 均 val Rank IC。
- 调参窗：train `TRAIN_START..2022-06-30` / val `2022H2`（既 early-stop 又打分）。搜索空间中心 = pilot 默认。
- 搜索空间维度：LGB 5 维（num_leaves/lr/min_data/lambda_l1/lambda_l2）；MLP/SAGE 5 维（lr/wd/dropout/hidden/num_layers）；GAT/HATS 6 维（+gat_heads）。
- 双机：Mac 8 study（L0/L1/L2s/L5s × {B,C}）+ T4 12 study（L2/L3/L4/L5/L6/L7 × {B,C}）。
- Rule 9 Touchpoint 2: `artifacts/reviews/2026-06-15_codex_code_A.md`（PROCEED-WITH-FIXES，2 MAJOR 修）。

## ⚠️ 解读红线

`winner_mean_val_ic_3seed` 是**调参选模目标**（2022H2，对 early-stop 集乐观偏置）——**不是测试结果，不得当 finding**。confirmatory 结果来自调参后用这些 HP 重跑 2160-cell（2023Q1–2025Q4），再过 §2a（DM-HLN+BH-FDR+SPA）。

## 全 study sqlite（all-trials）位置

完整 Optuna study db（30 全 trial）不在 git（二进制）：Mac 8 个在 `experiments/storya_v21_tune/studies/`（本地）；T4 12 个在 Drive `.../studies_backup/`（dbsync 备份，runtime recycle 后仍存）。

## 变更日志

- 2026-06-16: 调参完成（20/20），frozen_hparams.json 冻结（→ progress: 2026-06-16-b）
