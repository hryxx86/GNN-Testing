# scripts/ — 数据准备脚本

> 数据处理/图构建的辅助脚本。**不是实验主逻辑**（实验主入口在根目录 `run_*.py` 和 `.ipynb`）。

---

## 当前内容 (as of 2026-05-26)

| 文件 | 用途 |
|------|------|
| `build_dynamic_graphs.py` | 构建动态相关性图（Phase B 遗留） |
| `prepare_events.py` | 事件聚合（新闻 → 每日 event） |
| `process_news.py` | 新闻文本清洗 |
| `verify_docs_provenance.py` | 检查 advisor 文档中每个数字 claim 是否有 source 引用（`.claude/rules/docs.md` §4） |
| `colab_launch.sh` | 一行启动 Colab 训练 (tmux + nohup + log)，用于 SSH 远程跑长任务 |

---

## 关键文件速查

| 文件 | 用途 | 产出于 | 状态 |
|------|------|-------|------|
| `build_dynamic_graphs.py` | Dynamic graph snapshots | 2026-02 | archived in spirit |
| `prepare_events.py` | Event preparation | 2026-04-08 | active |
| `process_news.py` | News cleaning | 2026-04-08 | active |
| `verify_docs_provenance.py` | Doc provenance linter | 2026-04-23 | active; invoked via `/verify-docs-provenance` |
| `colab_launch.sh` | Colab training launcher (tmux + nohup) | 2026-05-26 | active; one-liner: `bash scripts/colab_launch.sh <run_script.py> [args...]` |

---

## 相关上下游

- 产出 → `data/dynamic_graphs/`, `data/pilot/`, `data/fullscale/sp500_news_events.parquet`

---

## 变更日志

- **2026-04-20**: 新增 README（→ progress: 2026-04-20-d）
- **2026-04-23**: 新增 `verify_docs_provenance.py` — doc provenance linter（→ progress: 2026-04-23-a）
- **2026-05-26**: 新增 `colab_launch.sh` — Colab SSH 远程一行启动训练（tmux + nohup + log）
