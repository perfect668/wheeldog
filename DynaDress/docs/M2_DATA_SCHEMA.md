# M2 数据与预测接口

原始 episode 在服务器 `/root/gpufree-data/dynadress/artifacts/m2/episodes/<id>/`。
复用 M1.1 的预抓持、预张开初态、真实 H1 与原始 FEM。一次 episode 重建完整 USD 场景；在 episode 内不重置人体关节或衣物节点。

## 原始记录

| 文件 | 内容 |
|---|---|
| `config.json` | 实际 K/D 四维数组、摩擦、三维初始偏差、动作类型、固定参考姿态、采样率 |
| `metadata.json` | 完整性、结果、环境版本、源码哈希、负载坐标系/单位/子树质量 |
| `samples.jsonl` | 20 Hz 状态与动作；物理步 240 Hz；样本在新动作发送前取状态，新动作作用于后续区间 |
| `result.json` | 真实完成或失败、原因、状态转移、最终几何判据 |
| `material_readback.json` | 从 USD 材料实际读回的静摩擦与动摩擦；不是仅保存请求配置 |
| `cloth_initial.npz` / `cloth_final.npz` | 一次性初始节点与真实结束节点；初始化含三角形及袖口边界 |
| `frame_index.jsonl` / JPEG | 仅选定 episode 启用 2 Hz RGB，保留真实物理时刻；不是所有样本均有图像 |

每行包含 `physics_step`、`control_step`、`simulation_time_s`、`sample_wall_time_ns`。机器人转动关节单位 rad/rad/s，四个手指滑动关节为 m/m/s；原始 `robot_joint_names` 给出顺序。

`action` 包括左右末端目标矩阵、世界平移/旋转增量、夹持命令、状态机阶段和探索动作。`solver_state` 包括机器人与人体 q/dq、实际末端 FK、80 个有序袖口节点、手/腕/肘位置、开口几何、附件误差和原始腕部关节反力。

## 负载通道

| 分组/字段 | 含义与来源 |
|---|---|
| `solver_state.wrist_reaction.raw_child_frame_N_Nm` | PhysX `get_link_incoming_joint_force()` 的原始腕部传递反力，顺序 Fx/Fy/Fz/Tx/Ty/Tz；前三项 N，后三项 Nm；incoming joint 的 child frame，原点为 localPos1 |
| `estimated_load.external_world_N_Nm` | 世界坐标轴、腕关节原点处的净外部载荷估计：子树所需惯性/重力支撑减去求解器关节反力 |
| `estimated_load.inertial_support_world_N_Nm` | Newton–Euler 补偿项，单独保留以便复算 |
| `solver_contact` | `available=false,value=null`：尚无经过验证的衣物—人体独立接触对力接口 |
| `commanded_wrench` | 人体保持与重力补偿的实际显式命令；不进入负载目标 |
| `measured_wrench` | `available=false,value=null`：没有物理 F/T 传感器测量；求解器关节反力不能冒充真机测量 |

补偿子树包含合并后的 wrist7/法兰/工具刚体和两根手指。每根手指 0.04 kg，不能省略。对各刚体使用 PhysX 实际质量、质心、惯性、世界速度/加速度：

`F_support = sum m (a_com - gravity)`

`T_support = sum (I_world alpha + omega × I_world omega + (com - joint_origin) × m(a_com - gravity))`

原始六维反力经 child joint 的实际旋转变换到世界坐标，再与补偿项相减。符号表示作用于腕部子树的外部载荷。此估计可以包含衣物附件、其他碰撞、数值/模型误差及关节摩擦，**不等于单独的人体接触力，也未用于真实力阈值保护**。没有使用人体保持力矩或 PhyRC traction。

## 属性条件与观测分离

`privileged` 保存 K、D、固定 q_ref、衣物摩擦；配置还保存固定几何和材料参数。它们作为 M2 已知条件单独输入模型，不混入状态历史。未来属性估计器可以替换相同条件向量；本轮不实现估计器。

`dynadress/models/features.py` 定义 `m2.features.v1`，完整字段名/切片/单位保存在导出的 `features.json`。

- 状态：人体 q/dq、机器人 q/dq、双末端位置/旋转前两列、袖口中心/法向/跨度/面积/平面残差/腕上进度、均匀选取的八个边界节点、左右净腕部载荷估计。
- 动作：双末端目标位姿、平移增量、旋转增量。夹持在这一批内恒为原生附件保持，不作为变化的数值特征。
- 条件：K[4]、D[4]、摩擦[1]、q_ref[4]。
- 历史：8 个 20 Hz 状态样本及其各自前一动作；候选当前动作单独输入。历史首尾间隔 0.35 s。
- 目标：下一完整紧凑状态。人体、袖口、进度及载荷是主要输出；同时预测机器人状态以支持无未来机器人状态输入的多步 rollout。

运动先验对人体/机器人关节采用当前速度积分，对袖口关键点采用已知双腕平均平移，对进度采用该平移沿袖口法向的投影；载荷和形状指标使用保持先验。GRU 残差修正变形、人体移动和负载变化。这是紧凑运动先验，不是完整 FEM 或刚体逆动力学的可微复制。

模型不预测完整 15878 节点布料、RGB-D、成功标签或真实接触力。缺失袖口/负载的初始化行不作为训练样本；不填零。六条旧 M1.1 episode 缺少负载，因此只审计，不用于本轮负载监督。

## 划分与评估

采集计划 `dynadress/configs/m2_collection/plan.json` 包含 50 个独立 K×D×摩擦组合，每组合两个不同动作/对齐条件的 episode。固定随机种子 1702 将组合分为 35 个训练、7 个验证、8 个测试组合；同组合的两个 episode 必须在同一分区。相邻帧仅在所属 episode 内构建窗口。归一化仅拟合训练 episode。

预测时间步 0.05 s。训练同时使用单步和二十步（1 秒）递推损失；最终评估 0.05、0.25、0.5、1 s，后续步只输入模型自身预测状态与记录的未来动作。未来动作来自原始反馈控制器，作为已知动作序列供预测；不读取真实未来状态，不等同于自主闭环控制评估。

报告同时给出全阶段和“袖口距腕部 > -50 mm”的几何子集；后者用于检查接近腕部的交互区域，不能作为有真实接触力的标签。保留失败数据，不把可预测性或任务成功率当作属性已可辨识的证明。
