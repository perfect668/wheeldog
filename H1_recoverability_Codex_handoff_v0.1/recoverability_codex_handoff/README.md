# Recoverability-Aware Physical Assistance — Codex Handoff v0.1

> 2026-11-11 | Research prototype, not a clinically validated medical or fall-prevention system.

## 一句话
验证：**对当前人体支撑任务同样可行、当前施力能力相近的 H1 全身姿态，是否会由于运动与响应约束，在有限时间的人体状态恢复能力方面存在显著差异？** 只有实验证实这个问题真实存在，才开始设计恢复性感知的预测控制算法。

## 研究定位
- 应用 Demo：通用轮式双臂 H1 辅助床旁起身—站立—步行伴随；初期研究用仿真人体或独立承重测试机构，绝不直接试验人体失衡。
- 候选核心：Recoverability-Aware Whole-Body Assistance for Mobile Dual-Arm Robots（暂定研究方向，**不是已确认的创新成果**）。
- 方法可能涉及人体简化平衡模型、机器人运动学/动力学约束、有限时间可恢复性、预测式控制。
- 当前阶段：**证明/证伪研究问题（Experiment 0）**，不要先开发完整 VLA、RL、Flow Matching、Isaac Sim 场景或人形动画。

## 团队职责
- **用户（PI）**：提供 H1 URDF/mesh、硬件资料、研究选择，审核安全和实验优先级。
- **Codex（实现者）**：实现、测试、运行实验、提交可复现数据与图表，列明假设与局限。
- **ChatGPT（方法/审阅者）**：审查模型、公式、对照公平性、代码差异、图表解释和论文贡献。

## 使用说明
1. 先阅读根目录 `AGENTS.md`、`docs/RESEARCH_BRIEF.md`、`docs/EXPERIMENT_PROTOCOL.md`、`docs/MODEL_SPEC.md`、`TASKS.md`。
2. 把 `PROMPT_TO_CODEX.md` 作为第一条 Codex 工作指令。
3. 第一轮只在 Linux 笔记本的 CPU 上跑；RTX 4060、远程 4×4090 暂时不需要。
4. 第一轮产出必须能经 `pytest` 复核，并生成**真实计算**得到的图（不得绘制假想曲线冒充结果）。
5. 在用户提供实际 URDF 后，才做 `docs/H1_ASSET_CHECKLIST.md` 所述的真实机器人建模。

## 目录中的文件
- `AGENTS.md`：给 Codex 的长期项目规则。
- `docs/RESEARCH_BRIEF.md`：背景、真实研究问题、竞争工作、论文叙事。
- `docs/MODEL_SPEC.md`：第一阶段的物理模型和单位约束。
- `docs/EXPERIMENT_PROTOCOL.md`：实验与对照组、指标、统计方法、继续/停止标准。
- `docs/H1_ASSET_CHECKLIST.md`：URDF 上线前的数据审计与模型验证。
- `docs/REVIEW_PROTOCOL.md`：每轮开发完 Codex 给用户/ChatGPT 交付什么。
- `TASKS.md`：里程碑与完成判据。
- `PROMPT_TO_CODEX.md`：第一轮可直接复制给 Codex 的任务单。
