# 分阶段任务单

## 已完成：M0 / EXP0（CPU，原始数据冻结）

### M0-A：建立可复现 Python 项目
- [x] `pyproject.toml` + README 使用说明；本地干净环境安装成功。
- [x] 结构 `src/recoverability/`, `tests/`, `configs/`, `scripts/`, `results/`（results 通常 gitignore，仅保留小型示例）。
- [x] `python -m pytest` 无报错，`python -m ...` 可无界面执行。

### M0-B：建立人体简化动力学和受限执行器
- [x] LIPM 侧向 x/v 模型，固定足底 CoP，SI 单位。
- [x] 期望力→实际力，包含 F_max、变化率/滞后、可选延迟。
- [x] 不同扰动：Δv 速度脉冲和短时外力脉冲。
- [x] 集成数值验证、符号与单位测试。

### M0-C：可恢复性 oracle 求解器
- [x] 给定扰动后求解有限时域控制是否可行，恢复条件可配置。
- [x] `recoverable / infeasible / solver_error` 三种状态，记录 solver 原因。
- [x] 同时保存轨迹和求解耗时。
- [x] 动力学、限制、终端条件严格通过后验数值检查。

### M0-D：实验对照与图表
- [x] 两种人工设置的辅助执行器能力，相同 F_max、不同 R_f/tau/delay；强调只是 toy model。
- [x] 扫描 Δv × horizon × 动态约束，保存 CSV/NPZ。
- [x] 画恢复热图、恢复阈值曲线、典型轨迹和求解成功/失败比例。
- [x] 人工一致性检查：对称扰动结果、能力单调性、无扰动稳定性。

### M0-E：Codex 交付
- [x] `docs/EXP0_RESULTS.md`：真实运行数据、发现、失败、限制、是否继续。
- [x] commit + 测试命令 + 环境版本 + 数据/图表路径。
- [x] 不创建“新算法”也可以完成第一轮；第一轮目标是模型验证。

## 当前审阅点：M1-A / M1-B（用户已授权，资产已提供）
- [x] M1-A：审计 URDF + meshes / links / joints / limits / inertial parameters，并对照产品手册。
- [x] M1-B：URDF内正运动学、雅可比差分、双臂IK、冗余静态配置搜索；真实H1语义冲突待厂家核实。

本轮结果：`docs/M1_ASSET_AUDIT.md`、`docs/M1_KINEMATICS_RESULTS.md`；仅几何条件性通过，停在审阅点。M0限定审阅G0/G1通过，G2/G3仍未验证。以下后续任务尚未授权：

- [ ] 匹配扶手目标与当前辅助余量的配置对。
- [ ] 计算与真实硬件约束相关的未来可达性和辅助能力。
- [ ] 初步使用 MuJoCo 验证动态接触，但不训练 RL。

## 后续：M2 / 研究方法（必须通过 Go/No-Go）
- [ ] 实现并公平比较名义 MPC、FORTE-inspired 当前力裕度、场景鲁棒 MPC、人体预测 NMPC。
- [ ] 根据差距提出计算高效的恢复性感知方法，并单独消融其核心模块。

## 后续：M3 / 真机与论文 Demo
- [ ] 先使用独立保护的人体模拟负载与支撑测试装置，完成硬件与控制安全审查。
- [ ] 分别验证支撑配置、接触过渡和随行运动，再组合成完整 Demo。
