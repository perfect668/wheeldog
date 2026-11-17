# M1.1 任务数据

当前任务数据位于远端 `/root/gpufree-data/dynadress/artifacts/m11/`；标称成功目录 `20261117T144712_trajectory02/`。所有采集是实际原生 PhysX 运行。完整逐帧视频/JSONL/末态留在远端，审阅包只选取必要文件。

每个正常结束的目录包含：

| 文件 | 语义 |
|---|---|
| `config.json` / `metadata.json` | 人体几何初态、固定参考姿态、K/D、IK 初值种子、独立腕朝向、速度、阈值、资产/环境版本和实际执行源码 SHA256 |
| `samples.jsonl` | 20 Hz 同步状态和动作；`metadata.complete` 表示日志正常关闭，任务是否成功另看 `result.success` |
| `result.json` | 成功/失败、真实状态转移时刻、结束原因、末态开口判据、重试次数 |
| `frame_index.jsonl` | 每张 RGB 的物理步、模拟时刻、主机采集时间戳和状态；10 Hz，800×600 |
| `cloth_initial.npz` | 一次性解析预张开状态、原始细化三角面及袖口索引；不是物理展开实验的快照 |
| `cloth_final.npz` | 最终 FEM 节点世界坐标及同一袖口索引；不逐帧写节点 |
| `trajectory.mp4` / `video_probe.json` | 实际连续 RGB 编码、解码帧数/分辨率/哈希；不是补帧合成视频 |
| `initialization_checks.json` | 附件解析后的一次性节点写入回读误差 |
| `validation.json` | 对记录时钟、固定参考、末态网格和最后十个成功条件的复核；并非独立重新运行仿真 |

物理 240 Hz，控制/记录 20 Hz。`physics_step` 来自 World，`simulation_time_s` 来自真实模拟时钟，`sample_wall_time_ns` 是主机时间。视频记录零时间渲染；播放 1 秒对应 1 秒模拟时间，不代表墙钟实时性能。末帧可能早于结束时刻不足 0.1 秒，使用 frame_index 对齐，不能用视频时长代替物理时钟。

`action` 包含双末端 4×4 世界目标矩阵、世界平移增量、旋转增量（本轮固定朝向为零）、袖口伺服目标和原生附件保持指令。双臂使用同一世界平移保持相对构型，关节命令另经速度/限位限制。夹爪为仿真假设工具，两个手指位置 drive 的初始/保持目标为 0 m；实际四个手指状态在关节数组内。稳定夹持来自 vertex attachment，不能解释为摩擦夹持强度测量。

五个信号来源组：

- `solver_state.available=true`：人体 q/dq、机器人全部 q/dq、手部/腕部/肘部、FEM 袖口边界节点。`end_effector_actual` 是**由实际求解器关节角做 FK**得到的位姿，固定底盘；不是外部跟踪器测量。`geometry`、附件误差、跟踪误差是从求解器状态计算的几何量，不能当作原始接触测量。
- `solver_contact.available=false,value=null`：没有提取 FEM 接触冲量/力。空 rigid contact report 不能证明无接触。
- `estimated_load.available=false,value=null`：未使用 traction，也没有将几何/应变保护命名为力。
- `commanded_wrench.available=true`：人体显式命令与广义重力补偿，单位 Nm；阻尼另由零位置刚度的隐式速度 drive 执行。不是人体接触力/实测力矩。
- `measured_wrench.available=false,value=null`：没有真实传感器。

`privileged` 单列人体 K、D、固定 q_ref，供训练/评价使用；不是未来控制器已能获取的在线识别结果。人体顺序 `shoulder_x,shoulder_y,shoulder_z,elbow_y`，rad/rad/s；机器人顺序每行 `robot_joint_names`，14 个转轴为 rad/rad/s，4 个手指滑轴为 m/m/s。

几何完成：袖口边界环拟合平面；人体轴与平面相交后，采用点在多边形内及到边界的最短距离，保守包住斜切截面（35 mm 半径/法向对齐程度 + 3 mm 裕量）。要求法向对齐 ≥0.9、平面残差 <12 mm，手尖/腕部已穿过，最终袖口位于腕部以上 `0.26 * target_forearm_fraction - 4 mm`，附件误差 <10 mm、末端跟踪 <8 mm，连续十个控制采样成立。十个 20 Hz 采样点跨度实际是 **0.45 秒**；实测日志转移 reason 的“0.5 s”是按十个周期书写，报告以记录时间为准。

`hand_section` 在穿过之前表示手臂轴投影处的开口可容纳性，必须结合有符号 `plane_distance_m` 与 `hand_entered/wrist_entered`，不能单独当作已穿入。`axial_from_wrist_m` 只是进度，不能单独判成功。成功标签仅适用于预抓持、原生附件、当前胶囊手/前臂和当前目标截面的仿真任务。

小批量每次新进程重新创建场景，等价于全场景复位；未提供进程内 reset。`scripts/m11_batch.py` 必须先提供 `success:true` 且 `geometric_completion` 的标称目录，才会运行 K4±5 mm、K12 的 0/+5 mm 条件。GPU 非位级确定性可能导致细小轨迹差异。失败数据保留实际原因；旧版控制器失败不能与新版数据混合当作相同控制策略的实验。
