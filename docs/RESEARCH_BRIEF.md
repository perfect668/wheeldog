# 研究上下文与论文主线（供 Codex 完整阅读）

## 1. 设备与算力
- 机器人：ROKAE 珞石逐星 H1，轮式移动底盘、双机械臂、当前为电动夹爪。用户已有 **H1 URDF**，尚未放入本项目。实验室还具有可编程照护床和洗浴椅。
- 用户确认机器人具有足够的拟研究静态支撑能力；但**动态抗倾覆、接触力监测、控制时延、故障保护仍待逐项验证**。
- 开发：Linux 笔记本 + RTX 4060；服务器：4×RTX 4090。第一阶段算法无需 GPU，多 GPU 留给以后高保真仿真/学习。
- 目标：IROS 2027 研究投稿；时间有限，应采用 Go/No-Go 关口。

## 2. 应用 Demo 与问题区分
Demo：H1 到床边定位双臂作为扶手 → 帮助从坐位起身 → 转入随行助行 → 到指定卫生间区域。Demo 是演示语境，不是创新本身；不要直接模仿专用承重机器人 E-BAR 的机械设计，也不要宣称 H1 能接住跌倒者。

## 3. 科学链条（必须坚持）
1. **现象（待实验证实）**：当前扶持同样合格的两种机器人全身配置，面对相同侧向人体扰动，可能在有限时间内表现出不同的恢复可行性。
2. **问题**：只看当前轨迹跟踪/静态施力余量，是否会选到未来缺少恢复动作空间的配置？
3. **原因假设**：有限的关节速度/力矩/行程、底盘移动和稳定性、接触力方向、测量/动作延迟、人体状态演化共同限制可恢复性。
4. **尖锐矛盾**：目前能安全完成任务 ≠ 接下来仍来得及、够得着、力有效地处理变化。
5. **洞察假设**：辅助性能应显式考虑“在指定反应时间与约束下恢复到可控人体状态集合”的可行性，而不只是最大力。
6. **候选设计（以后再做）**：可恢复性指标 + 未来状态/扰动场景 + H1 全身配置优化 / MPC。
7. **证据**：同一初始人体与扶手目标、匹配当前力能力的配置对，交叉测试恢复边界，并与强基线比较。

## 4. 论文尚未成立的风险
- “伸直机械臂比弯曲灵活”本身是常识，不是科研创新。
- FORTE 已经优化机器人任务相关的剩余力多面体及轨迹力裕度。不能宣称首次考虑未来施力能力。
- 2026 NSR tactile pHRI 已经利用人体质心模型、预测接触规划与全身 QP 实现助行。不能宣称首次预测人体并调整接触位置。
- 鲁棒 MPC、viability/reachability 也已经成熟。若只把两者相加，审稿人会质疑没有方法贡献。
- 如果普通鲁棒 MPC 的性能/计算效率不弱于我们的候选方法，则需要修正主线而非掩饰结果。

## 5. 相关文献（核对并阅读原文，不凭题目猜实现细节）
- FORTE (2026) — https://arxiv.org/abs/2609.21497 ; code https://github.com/yeying256/FORTE
- Tactile Physical Human–Robot Interaction (NSR 2026) — https://doi.org/10.1093/nsr/nwag402
- Elderly Bodily Assistance Robot / E-BAR (ICRA 2025) — https://news.mit.edu/2025/eldercare-robot-helps-people-sit-stand-catches-them-fall-0513
- Asymmetric Admittance Control for Compliance to Time-Varying Interaction Force in Dual-Manipulator Rehabilitation System (RA-L 2025) — https://doi.org/10.1109/LRA.2025.3600137
- Human biomechanics references: https://jmontp.github.io/LocoHub/datasets/gt24/ ; https://www.addbiomechanics.org/download_data.html
- MuJoCo docs: https://mujoco.readthedocs.io/en/stable/programming/modeledit.html
- Pinocchio docs: https://stack-of-tasks.github.io/pinocchio/index.html

## 6. 阶段范围
- **本轮：EXP0** 低维人体状态 + 受限辅助执行器；模型一致性/可恢复域与简单当前力裕度比较。全部 CPU 运行。
- **下一轮：EXP1** H1 URDF 模型审计、FK/Jacobian/IK、真实约束下的配置对和力能力指标；不急着做大场景。
- **后续：EXP2** 控制优化与强基线；**EXP3** 高保真接触仿真和带独立保护的机器人实测。
