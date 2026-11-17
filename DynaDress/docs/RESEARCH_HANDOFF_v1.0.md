# DynaDress：IROS 研究方案与工程移交（v1.0）

**日期：** 2026-11-17  
**状态：** 方案冻结；M0 待执行（尚无服务器实测结果）  
**论文暂定题目：** *DynaDress: Active Identification of Coupled Human–Garment Dynamics for Adaptive Robotic Dressing*

> 本文用于在 ChatGPT Work、Codex 和研究成员之间传递已经确定的研究方案和任务边界。以下为计划和设计，不代表代码已经完成或实验已经通过。

## 0. 一页结论

| 决策项 | 固定方案 |
| --- | --- |
| 论文目标 | IROS 方法论文；养老照护作为辅助穿衣场景背景 |
| 主任务 | 坐姿单侧穿袖；H1 双臂抓持同一袖口两侧 |
| 人体 | 躯干固定 + 单侧肩 3 DoF / 肘 1 DoF 被动动力学上肢，不做主动人体策略 |
| 衣物 | PhyRC 的 T 恤 FEM 布料，研究接触、滑动、卡住与物理参数变化 |
| 仿真 | PhyRC 2027 Challenge + Isaac Sim 6.0.1（Docker，Headless） |
| 服务器 | 单张 RTX 4090 24GB；其余 CPU/内存/磁盘/驱动待检测 |
| 方法 | 耦合交互世界模型 → 双隐变量在线辨识 → 信息感知受约束 MPPI |
| 当前阶段 | M0：原始 PhyRC 基线部署，M1 才改 H1 和被动人体 |

## 1. 科学问题与技术定位

辅助穿衣时，相似的接触负载变化可能来自衣物滑动/卡住，也可能来自人体上肢的被动阻抗与位移。若机器人不能辨别原因，就难以选择恰当的推进、旋转或退让动作。

**核心洞察：** 动作既是完成任务的执行输入，也能是获取物理信息的主动探测输入。论文研究如何在线学习未知人—衣物耦合响应，并将辨识结果用于真实受约束的后续决策。

本研究**不主张首次**进行机器人穿衣、视觉力反馈、人体运动预测、单纯衣物世界模型或触觉 MPC。核心差异在于：**人体/衣物两类隐含动力学属性 + 任务内主动可辨识动作 + 信息感知控制**的统一闭环。

## 2. 三项创新点

**C1｜交互条件化耦合世界模型。** 以机器人候选动作为条件预测人体被动响应、衣物进度和接触模式、交互负载；采用简化物理动力学加轻量残差网络。

**C2｜解耦在线交互辨识。** 分别估计人体有效阻抗 `z_h` 和衣物有效接触特性 `z_c`，利用衣物尚未接触人体与接触之后两个阶段的信息差异；估计有决策价值的不确定性，不声称直接测得所有生理参数和材料常数。

**C3｜信息感知受约束 MPPI。** 在穿袖进度、负载与风险约束下加入预期信息增益，使机器人选择有助于区分阻力来源的小幅动作，并持续更新规划。

## 3. 固定任务与非目标

- **机器人：** ROKAE 逐星 H1 轮式双臂，底盘固定；双臂分别抓袖口两侧；已有 H1 URDF 待导入。
- **人体：** 坐姿固定躯干，仅单侧肩 3 DoF + 肘 1 DoF，被动动力学、可变有效刚度和阻尼，目标参考姿态在单次任务中不变。
- **衣物：** 现成 T 恤 FEM 资产；只完成单侧穿袖，不做整件上衣穿戴。
- **动作：** 抓持/初始化 → 对齐手腕 → 穿入 → 推进到预定位置 → 结束/退让。
- **信号：** 双臂运动/关节信息、人体上肢状态估计、袖口状态、接触/负载反馈。区分真机可测观测与仿真特权真值。
- **PICO4：** 暂不参与核心仿真或训练；后续可用于机械臂遥操作示教。

不做 AMP 主动人体、意图识别、VLA 预训练、人体起身/助行、移动底盘控制以及完整穿衣。

## 4. 方法与数学接口

人体被动动力学：

```text
M_h(q) q_ddot + C_h(q,q_dot) q_dot + g_h(q) = tau_hold + J_h^T F_cloth
tau_hold = g_h(q) + K_h (q_ref - q) - D_h q_dot
```

世界模型与在线辨识：

```text
s_next_pred = f_physics(s, a; z_h, z_c) + residual_theta(s, a, z_h, z_c)
(z_h_hat, z_c_hat, uncertainty) = Encoder(obs_history, action_history)
```

MPPI 目标：

```text
J = J_task + lambda_F * J_force + lambda_R * J_risk - lambda_I * InformationGain(action)
```

**动作接口：** 左、右末端短时位姿增量。  
**预测量：** 人体上肢下一状态、袖口进度、接触模式（free/sliding/sticking/snagging）、负载及其不确定性。  
**物理要求：** 衣物接触须通过物理求解器使人体关节发生被动响应，不得通过直接脚本位姿修改模拟反作用。

## 5. 仿真平台与复用

**主框架（不再变更）：** [PhyRC 2027 Challenge](https://github.com/rirolab/PhyRC_2027_Challenge)，使用 Isaac Sim 6.0.1 Docker 环境、T 恤/FEM 布料、夹爪抓取、相机与录制管线。

**参考项目：** [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym) 提供人体关节/控制和穿衣任务逻辑；[DexGarmentLab](https://github.com/wayrise/DexGarmentLab) 提供衣物资产和组件参考；MuJoCo/RehabBot 仅供被动上肢建模思路参考，不作为新的主仿真引擎。

**工程注意：** 原始假人的改造侧手臂碰撞结构应停用，避免与动态手臂重复碰撞。PhyRC `ContinuousClothControl.py` 的牵引/应变保护量不是经标定的真实力传感数据。应分别标记物理求解器真值、模型估计量、模拟真机传感器量。

## 6. 开发路线与完成定义

| 里程碑 | 实施范围 | 完成定义 |
| --- | --- | --- |
| **M0（当前）** | 单卡 4090 环境、Docker/NVIDIA Toolkit、PhyRC 原始场景 Headless 部署 | 原生 `PHYRC-SMOKE-PASS`、日志、环境版本与截图/视频 |
| **M1** | 替换 H1 双臂；搭建 4 DoF 被动上肢；FEM 衣物接触；数据记录 | 连续穿袖，衣物通过接触力引发人体真实被动运动，批量数据落盘 |
| **M2** | 物理先验+残差世界模型；在线人体/衣物隐变量辨识 | 能预测候选动作引起的响应与负载，并在线适应新组合 |
| **M3** | 信息增益项 + 短时域 MPPI | 对比固定策略、被动在线辨识、主动在线辨识的完整闭环 |
| **M4** | H1 真机 + 真实衣物 + 可调阻抗假人手臂 | 接触负载、穿袖成功率与恢复能力的真实物理结果 |

## 7. M0 立即执行任务

首先测量 OS/驱动、CPU、内存、磁盘、Docker、NVIDIA Container Toolkit。PhyRC 官方建议至少 32GB 内存、100GB 空闲 SSD；必须以服务器实测为准。采用官方锁定的 Isaac Sim 6.0.1，不额外安装宿主机 Isaac Sim，不以远程 X11 GUI 为验收依赖。

```bash
git clone https://github.com/rirolab/PhyRC_2027_Challenge.git
cd PhyRC_2027_Challenge
./run.sh doctor
./run.sh build
./run.sh prepare
./run.sh smoke
```

记录实际 commit、镜像 ID、耗时、GPU/内存占用、日志、可视化结果；**原生 `PHYRC-SMOKE-PASS` 是首要成功标志**。输出 `docs/M0_REPORT.md`。如需离屏截图/视频，单独实现 Headless 数据导出脚本，不能伪造运行成功。本阶段不修改 H1、人体或世界模型。

## 8. 论文验证方案：仅三组主实验

1. **世界模型：** 测量人体运动、衣物进度、接触模式和接触力的多步预测误差。
2. **主动辨识：** 固定模型 / 被动在线更新 / 主动信息感知的对比；指标为新参数组合下的辨识效率、接触负载和穿袖进度。
3. **H1 真机：** 可调阻抗假人手臂和不同衣物条件；比较完成率、峰值/累计接触负载、完成时间和卡住后的恢复。

**原则：** 真实部署时不能读取仿真特权变量；验证的物理反馈必须与实际可用的传感链路对应。

## 9. 决策记录：冻结内容

- 被动动力学人体是论文正式模型；静态人体只用于加载和碰撞调试；主动人体不做。
- 坚持 PhyRC / Isaac Sim 6.0.1；不因 MuJoCo 更轻量而更换已确定平台。
- 保留 FEM 衣物，不自行开发新的布料物理求解器。
- 核心创新为主动解耦辨识与信息感知决策；接触安全约束是必要组成而非独立主创新。
- PICO 暂不参与，底盘不移动，不做整件上衣穿戴。
- 后续仅根据实现中真实出现的问题调整实现细节；方案变更应有明确理由和记录。

## 10. 关键资源

- [PhyRC 主仓库](https://github.com/rirolab/PhyRC_2027_Challenge)
- [PhyRC 安装指南](https://github.com/rirolab/PhyRC_2027_Challenge/blob/main/docs/competition/SETUP.md)
- [Isaac Sim 6.0.1 容器说明](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/install_container.html)
- [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym)
- [DexGarmentLab](https://github.com/wayrise/DexGarmentLab)
- [Force-Modulated Visual Policy（CoRL 2025）](https://proceedings.mlr.press/v305/hao25b.html)
- [Learning to Collaborate from Simulation（RA-L 2020）](https://publications.ri.cmu.edu/learning-to-collaborate-from-simulation-for-robot-assisted-dressing)
- [Do You Need a Hand?（T-RO 2024）](https://research.tudelft.nl/en/publications/do-you-need-a-hand-a-bimanual-robotic-dressing-assistance-scheme/)
- [2026 软性衣物世界模型相关工作](https://arxiv.org/abs/2610.30988)

## 11. 项目状态与接续建议

**已确定：** 论文问题、人体模型、任务边界、仿真框架、核心算法与 M0–M4 开发次序。  
**未完成/未经验证：** 目标服务器部署、PhyRC smoke 实测、H1 导入、被动上肢与 FEM 的真实碰撞、模型训练和论文实验。  
**下一步：** Codex 执行 M0 并提交 `M0_REPORT.md`，由研究成员审阅真实日志后进入 M1。这个文档可作为新 Work 项目的固定初始上下文。
