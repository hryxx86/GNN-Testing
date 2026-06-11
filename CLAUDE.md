# CLAUDE.md — GNN-Testing Project Rules

> Loaded every session. Rules 1-4, 7, 8-invariants, 9, 10 are universal and inline here.
> Rules 5 (quad-doc), 6 (archived), 8-notebook-specifics are split into path-scoped files under `.claude/rules/`:
> - `.claude/rules/docs.md` — loads when editing progress/plan/analysis/README/handoff/advisor docs
> - `.claude/rules/notebooks.md` — loads when editing `.ipynb`
> - `.claude/rules/experiments.md` — loads when editing `run_*.py`, `analyze_*.py`, `build_*.py`, experiment scripts
> - `.claude/rules/archived.md` — loads when reading/editing `archived/**`

---

## Rule 1: Addressing

Every response MUST start with **"H博士，"** then the actual answer.

---

## Rule 2: No Guessing, No Substitutes

- **Unknown things or new plans**: MUST ask H博士 for details first. Never make assumptions or decisions autonomously.
- **Code implementation**: MUST NOT use substitute libraries, workarounds, or placeholder code to "get it working." If the intended approach doesn't work, stop and ask.
- **Architecture changes**: Propose and explain impact BEFORE making changes. H博士 must approve.

---

## Rule 3: Communication

- **Language**: Chinese for conversation, English for code and technical reports.
- **Transparency**: Explain the impact of any change BEFORE executing it.
- **Goal-oriented**: Experimental results must be good — outcomes over process.

---

## Rule 4: Session Start Protocol

Every session, BEFORE doing anything else:

1. Read `progress.md`, `plan.md`, `docs/analysis.md`
2. If a `docs/session_handoff_*.md` exists and is recent, read its YAML frontmatter first (see `.claude/rules/docs.md` §5 for the manifest schema)
3. Orient and summarize current state to H博士
4. Quick sanity check: scan recent code changes for new bugs, missing fixes, or inconsistencies
5. **浏览本次任务相关文件夹的 `README.md`**（任务所在文件夹 + 即将产出/读取的文件夹）。根目录 `README.md` 是总入口。

---

## Rule 5: Quad-Doc System (summary — full details in `.claude/rules/docs.md`)

Four document classes maintain project state. Each has its own update trigger:

| File | Purpose | Update When |
|------|---------|-------------|
| `progress.md` | 做了什么 (past) | After every completed task |
| `plan.md` | 接下来做什么 (future) | Start + end of each session |
| `docs/analysis.md` | 分析发现 (findings) | After every analysis/experiment |
| `<folder>/README.md` | 文件夹索引 (what's there) | Structural changes only |

Entry IDs are date-based: `## YYYY-MM-DD-x: Title`. Tri-doc cross-reference is mandatory: `→ progress: 2026-02-27-b | plan: 2026-02-27-b | analysis: N/A`.

`plan.md` keeps a Decision Log at the bottom. README update triggers are narrow (structural only, not every file creation).

**See `.claude/rules/docs.md`** for: README scope details, numeric provenance rule (H4), session handoff manifest schema (H5), reviewer output schema (H3).

---

## Rule 6: Unimplemented Plans (summary — details in `.claude/rules/archived.md`)

- `archived/plans/phase_f_design.md` — planned but unimplemented. Requires H博士 discussion before implementation.
- When a planning doc is superseded, merge useful parts into `plan.md`, move original to `archived/`.
- `archived/**` is read-only by default.

---

## Rule 7: Key Paths

- Google Drive folder: `GNN测试` (NOT `GNN-Testing`)
- Local project: `/Users/heruixi/Desktop/GNN-Testing`
- Conda env: `gnn` — Python: `/opt/homebrew/Caskroom/miniforge/base/envs/gnn/bin/python`
- GitHub repo: `https://github.com/hryxx86/GNN-Testing`
- **Colab 架构**: **Code = GitHub（clone 到 Colab 本地盘），Data = Google Drive（软链）**。Drive 挂载**不是** git 仓库，`cd <drive> && git pull` 一直无效 → 改用 `scripts/colab_bootstrap.sh`（git clone 代码到 `/content/GNN-Testing` + 软链 `data/experiments/plots/wandb` 自 Drive）。`artifacts/` 不软链（git 管理）。
- **Colab 每次 runtime 的 3 个 cell**（path B, locked 2026-06-10）:
  ```python
  # Cell 1 — 挂 Drive
  from google.colab import drive; drive.mount('/content/drive')
  # Cell 2 — 代码自 GitHub + 数据软链 Drive
  !curl -sSL https://raw.githubusercontent.com/hryxx86/GNN-Testing/main/scripts/colab_bootstrap.sh | bash
  %cd /content/GNN-Testing
  # Cell 3 — SSH 隧道（自建 sshd + http2 cloudflared，打印 hostname）
  !bash scripts/colab_ssh_tunnel.sh
  ```
- **SSH 不变式（铁律）**: sshd 监听端口 **必须 == 隧道转发端口**。`colab_ssh_tunnel.sh` 强制 sshd 绑 22 + 隧道指 22。**不要再用 `colab_ssh` / `launch_ssh_cloudflared`**（PyPI 停在 0.3.27 / 2021-10；其 sshd 配 2222 而隧道指别处 → 永久 origin 502 + 本地 `websocket: bad handshake`）。
- 隧道强制 `--protocol http2`（cloudflared 默认 QUIC/UDP 易被掐 → 本地连接超时）。
- SSH 命令模板: `sshpass -p "GNNTEST" ssh <HOSTNAME>.trycloudflare.com "命令"`。依赖: `brew install cloudflared sshpass`（本地已安装）。
- 本地 `~/.ssh/config` 已代理所有 `*.trycloudflare.com`（含 keepalive；脚本输出兼容，本地无需改动）。
- **注意**: hostname 每次重启 runtime 会变，需要 H博士 提供新地址。

---

## Rule 8: Technical Conventions

### Universal invariants (apply everywhere, not just to specific file types)

- **Data leakage**: All market context features use strictly T-1 close values. No exceptions.
- **Return definition**: Next-day close-to-close, locked and cannot be changed.

### Format preferences

- `.ipynb` for analysis/experiments; `.py` scripts are acceptable for local batch runs and utilities.
- `.ipynb`-specific conventions (NotebookEdit semantics, cell structure) are in `.claude/rules/notebooks.md`.
- Experiment/analysis script conventions (seeds, output layout, walk-forward) are in `.claude/rules/experiments.md`.

### Runtime

- Local Mac Mini M4 16GB (CPU/MPS) or Google Colab Pro via SSH (Tesla T4/A100, 视分配).

---

## Rule 9: Codex 协作 + Code Review (MANDATORY)

Codex 是项目的批判性研究顾问。**每个关键节点必须触发讨论，不可跳过。**

### 三个强制触发点

#### 触发点 1：Plan 写完后

当 `plan.md` 新增或修改了实验方案/技术路线时：
1. 发给 Codex 批判性评估 → 双方用证据讨论（通常 2-3 轮）
2. 达成共识或整理分歧摘要 → 提交 H博士 审批

**Slash command**: `/codex-plan-review <plan-file>` — encodes the protocol, handles fallback to finance-gnn-reviewer if Codex >15min unresponsive, requires structured output per `.claude/rules/docs.md` §6.

#### 触发点 2：Code 写完后

当新写或大幅修改了代码文件时：
1. 发给 Codex 做 correctness review（聚焦：逻辑正确性、数据泄露、统计方法）
2. 逐条回应：接受并修复，或用代码证据反驳
3. Claude 必须亲自读代码/跑测试验证 Codex 的发现
4. 达成一致后才进入实验运行

**Slash command**: `/codex-code-review <file...>`.

#### 触发点 3：实验结果出来后

当实验产出 IC/Sharpe/统计检验等结果时：
1. 发给 Codex 分析评估（结果可信度、统计方法、结论是否过度解读）
2. 用数据和逻辑讨论 → 达成共识的结论写入 `docs/analysis.md`

**Slash command**: `/codex-results-review <results-dir>`.

### Session 结束审查

工作结束前，启动 3 个 Explore agent **并行审查**本次修改的代码：
1. **Data Leakage Audit** — feature/label/graph/train-test 前瞻性泄露
2. **Statistical Methodology** — IC/Sharpe 公式、假设检验、多重比较
3. **Code Correctness** — 训练循环、边界情况、可复现性

报告 PASS/FAIL/CONCERN + 严重性。Critical 当场修复，Major 记录到 progress.md。

**Slash command**: `/session-closeout` — spawns the 3 Explore agents in parallel, aggregates their structured outputs.

### 讨论规则

1. **不盲信** — Codex 可能误判或过度防御，每条建议都要审视
2. **不盲拒** — 真正的问题必须认真修复
3. **用证据说话** — 代码片段、测试输出、数学推导、文献引用。不接受"我觉得"
4. **Claude 必须亲自验证** — 不准口头说"已验证"而没有实际去读代码/跑测试
5. **最终决策权在 Claude + H博士** — Codex 是顾问

### 分歧解决（防死锁）

3 轮沟通仍无法一致 → Claude 整理分歧摘要 → H博士 裁决。**Codex 讨论不得阻塞项目推进。**

### Fallback reviewer

**触发条件**：Codex CLI 在 Rule 9 任一触发点响应超过 15 分钟（含：完全不响应 / 错误中断 / 输出空内容）。

**替代方案**：启动 `finance-gnn-reviewer` subagent（定义在 `.claude/agents/finance-gnn-reviewer.md`）代替 Codex 执行当轮 review。

- 身份定位：senior ML research scientist, GNN + quantitative finance specialization, NeurIPS/ICML/ICAIF reviewer-level
- 职责：Plan review / Code review / Results review，和 Codex 相同的三触发点覆盖
- 输出格式：CRITICAL / MAJOR / CONCERN 三档，同 Codex，per `.claude/rules/docs.md` §6 frontmatter schema
- 诚信同样要求（Rule 9 诚信条目全部适用）

**使用方法**：
```
Agent(subagent_type="finance-gnn-reviewer", prompt="...")
```

The slash commands (`/codex-*-review`) handle fallback automatically.

**重要**：启用 fallback 必须在 progress.md 记录：`## YYYY-MM-DD-x: Fallback Reviewer — Codex timeout, finance-gnn-reviewer took touchpoint N`。不准掩盖 fallback 使用。fallback subagent 的发现享受和 Codex 相同的审议权重。

### 主线聚焦

- **聚焦 correctness** — 数据泄露、统计正确性、逻辑 bug
- **拒绝防御性代码** — "以防万一"的 edge case、过度 error handling → 明确拒绝
- **拒绝偏离主线** — 与当前实验目标无关的建议 → 拒绝
- **不在风格上纠缠** — 变量命名、格式等不影响正确性的问题 → 跳过

### 诚信要求（CRITICAL）

1. **不准捏造** — 不准编造 Codex 回复、伪造讨论过程、虚报"Codex 已同意"
2. **不准跳过** — 三个触发点是强制的，不准以任何理由跳过
3. **不准敷衍** — 不准走过场式讨论，不准说"已修复"但实际没改
4. **不准隐瞒** — Codex 发现的真正问题必须如实报告 H博士
5. **不准偷懒验证** — "我已验证"必须是真的读了代码/跑了测试

### 讨论记录

每次 Codex 讨论记录到 `progress.md`：`## YYYY-MM-DD-x: Codex Review — [Plan/Code/Results]`. Full review body saved to `artifacts/reviews/<YYYY-MM-DD>_<reviewer>_<touchpoint>_<round>.md`. Required frontmatter schema in `.claude/rules/docs.md` §6.

---

## Rule 10: Current Project State (as of 2026-04-22)

- Phases 1-2, v3 Pipeline, Week 1-3: ALL DONE
- **v4 5-fold walk-forward DONE**: 6 models × 3 seeds × 5 folds = 90 runs → `experiments/wf5_results.csv`
- **Step 0 Reruns DONE**: horizon ablation (360) + arch comparison (150) + permutation v2 (16K shuffles)
- **SEC Gate 1 STOP**: Layer 1 Lazy Prices 对 NN 有害; Layer 2/3 CANCELLED
- **Phase 5 进行中**: Step 3 Plan Z 已完成（Hansen SPA + FDR + Fold 4 泄露诊断）
- **Loss horse race (2026-04-22)**: smoke v3 passed; awaiting Colab A100 Stage 0 launch
- **README 体系建立 (2026-04-20)**: 24 个文件夹 README；Quad-Doc（Rule 5）仅在**结构性变更**时触发 README 更新
- **Infrastructure overhaul (2026-04-22)**: `.claude/rules/` path-scoped rule files; `.claude/commands/` Rule 9 slash commands; structured reviewer YAML schema; doc provenance verifier
- **Session handoff**: `docs/session_handoff_2026-04-20.md` + future handoffs must use manifest schema (`.claude/rules/docs.md` §5)
- **Technical decisions**: See `plan.md` Decision Log — all decisions final unless H博士 reopens
