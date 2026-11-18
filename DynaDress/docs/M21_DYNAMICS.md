# M2.1：已知被动上肢动力学核对

本文件只报告实际运行的无衣物测试与固定惯性近似，不把外加测试脉冲当成衣物力矩标签。世界模型训练/测试结果见 `M21_REPORT.md`。

## 仿真方程与接口

复用 `dynadress/human/passive_arm.py` 的关节顺序：`shoulder_x, shoulder_y, shoulder_z, elbow_y`。PhysX 实际返回顺序相同，索引为 `[0,1,2,3]`。参考姿态为 M2 的 `[0,1,0,0.35]` rad。

- 上臂长 0.30 m、质量 1.8 kg、COM 位于局部 z=-0.15 m。
- 前臂/手长 0.26 m、质量 1.0 kg、COM 位于局部 z=-0.13 m。手碰撞体与前臂同刚体；其质量未另行重复加入。
- 两个肩中间载体各 0.001 kg，惯性各轴 1e-5 kg·m²。形状刚体对角惯性直接复用原实现。
- 躯干固定。关节 USD 限位依次为 `[-100,100],[-150,70],[-100,100],[0,145]` 度。

真实采集器每物理步发送 `clip(g(q)+K(q_ref-q), ±[30,30,20,20])` Nm。位置驱动刚度为零；`D` 由零目标速度的隐式 velocity drive 实施，最大驱动力矩 100 Nm。显式 effort 不再减去 `D*dq`，否则会重复阻尼。重力补偿在 hold 未限幅时抵消重力；限幅后仍有残余重力项。

`dynadress/models/impedance.py` 提供：

```python
mechanics = FixedImpedanceDynamics(
    substeps=12, use_hold_clip=True, joint_limit=True)
q_next, dq_next = mechanics(q, dq, tau_cloth, K, D, q_ref, dt=0.05)
```

`q/dq/tau_cloth` 为 `(...,4)`，单位分别为 rad、rad/s、Nm。K、D 直接进入方程，广播为四关节参数。`configuration()` 返回可序列化配置。`tau_cloth` 是无直接标签的等效广义作用，不等同于机器人腕力，也不受人体 hold 力矩上限裁剪。

## 实际测得的参考惯性

2026-11-18 在现有 Isaac Sim 原生环境中，用真实 PhysX `get_mass_matrices()` 读取参考姿态惯性。完整精度、重力、关节顺序与三个测试结果保存在 `artifacts/m21/dynamics/physx_reference.json`。

```text
M [kg m²] ≈
[ 0.05452233   0.00000000  -0.00684564   0.00000001 ]
[ 0.00000000   0.24103186   0.00000000   0.05947511 ]
[-0.00684564   0.00000000   0.00504841   0.00000000 ]
[ 0.00000001   0.05947511   0.00000000   0.02283958 ]
```

特征值为 `[0.00411866,0.00768092,0.05545208,0.25619051]`，严格正定。两次相同参考姿态读取的矩阵一致。保留非对角项，尤其肩 Y—肘 Y 耦合。参考重力补偿约 `[0,5.949592,0,1.244340]` Nm。根据相同质量/COM/关节轴写出的解析 `gravity_compensation(q)`，在三个实测响应全程与 PhysX 的最大差异不超过 6.1e-6 Nm。

## 离散更新与近似范围

20 Hz 世界模型内部使用 12 个 1/240 s 子步，固定 M、忽略 C 和惯性随姿态变化。每个子步 h 使用：

```text
c_sat = clip(g(q)+K*(q_ref-q), ±tau_limit) - g(q)-K*(q_ref-q)
A = M + h*diag(D) + h²*diag(K)
dq_next = solve(A, M*dq + h*(K*(q_ref-q)+tau_cloth+c_sat))
q_next = q + h*dq_next
```

K、D 采用隐式更新，hold 限幅残差在子步输入姿态计算；hold 未饱和时残差严格为零，饱和时为一阶近似。真实 PhysX 的 K 由逐步显式 effort 实施，因此本模型的隐式 K 与其有小的离散误差，已用响应数据核对。关节越界时将 q 投影到真实 USD 限位，并清除向外的速度，这是完全非弹性 hardstop 近似，并非完整 PhysX 碰撞冲击求解。

为降低训练开销，关闭 `use_hold_clip` 和 `joint_limit` 时启用 12×12 增广仿射矩阵的幂，精确组合相同 12 个隐式子步。该选项只有在报告清楚未发生 hold 饱和/关节撞限的使用范围时才适用。默认开启两项约束；测试验证了正常范围内快路径与逐子步路径一致。

该近似没有声称覆盖大幅摆动、关节撞限冲量、耦合碰撞的全部刚体动力学，也没有把网络的等效力矩解释为唯一可辨识的真实接触力矩。

正式训练为了降低自动微分开销，选用 `use_hold_clip=False, joint_limit=False, substeps=12` 的仿射路径。原始 100 条 episode 的 26465 个有效 20 Hz 状态经 `scripts/m21_dynamics_audit.py` 核对：训练/验证/测试分别 17474/4206/4785 帧，均未发现 hold 饱和或撞限。训练集四轴 hold 最大绝对值为 `[0.586,7.850,0.309,1.684]` Nm，距 `[30,30,20,20]` 上限充分远；距机械限位的最小余量依次为 `[1.6624,0.1885,1.7058,0.3215]` rad。全数据最小余量同样为 0.1885 rad。详见 `data_regime_audit.json`。该核对只覆盖原始 20 Hz 记录，不能证明未记录的 240 Hz 帧绝无瞬时撞限，也不保证模型自由预测始终位于此区域；预测范围需另行统计。

## 实测无衣物脉冲/回稳核对

新运行三个各 6 s、1440 物理步的无衣物案例：M2 参考姿态的 K=4,D=0.5 与 K=20,D=2，以及 M1 原参考 `[0,-1.3,0,0.3]` 的 K=4,D=1。所有 case 都只在 reset 设置 q/dq；此后完全由 PhysX 积分。肩 Y 在 1.0–1.5 s 加 1 Nm 测试脉冲，肩 X 在 2.5–2.6 s 加 0.15 Nm。每步保存 q/dq、hold、solver gravity 与独立 `test_pulse_Nm`。

20 Hz 取样后，从每个可用历史起点输入真实候选脉冲，比较未来 q。下表为四关节和所有窗口共同 RMSE（rad），不是衣物任务上的模型分数。

| 案例 | 0.25 s 结构化 | 0.5 s 结构化 | 1 s 结构化 | 1 s 保持 | 1 s 速度外推 |
|---|---:|---:|---:|---:|---:|
| M2 K=4,D=0.5 | 0.004074 | 0.004357 | 0.004632 | 0.059622 | 0.160628 |
| M2 K=20,D=2 | 0.000146 | 0.000161 | 0.000183 | 0.011484 | 0.038384 |
| M1 K=4,D=1 | 0.000331 | 0.000397 | 0.000414 | 0.052850 | 0.114135 |

K=4,D=0.5 的强脉冲碰到肩 Y 的 70°上限，共 18 个物理步位于上限附近；该案例误差包含 hardstop 冲击近似。其他两个案例未碰该限位。三个 case 都没有 hold 限幅。6 s 完全开环 q RMSE 分别为 0.004337、0.000167、0.000398 rad；在撤去测试脉冲后均恢复。曲线 `artifacts/m21/dynamics/impedance_validation.png` 同时显示角度与速度，数值见 `validation.json`。这些结果支持参考惯性在局部运动中可用，但不替代 FEM 接触任务评估。

## 复现

在服务器 `/root/gpufree-data/dynadress`：

```bash
./scripts/run_native.sh python /root/gpufree-data/dynadress/scripts/m21_dynamics_probe.py
./scripts/m2_model_python.sh scripts/m21_dynamics_validate.py
./scripts/m2_model_python.sh scripts/m21_dynamics_audit.py
./scripts/m2_model_python.sh -m unittest tests.test_m21_dynamics
```

实际日志为 `setup_logs/m21/dynamics_probe.log`、`dynamics_validate.log`。7 项数值测试真实通过，包括正定性/配置、静止平衡、被动能量递减、耦合和梯度、批次一致性、仿射组合一致性、机械限位。矩阵和原始响应的哈希记录于 `artifacts/m21/dynamics/evidence_manifest.json`。
