# M1 episode 字段与复现

> 本文保留 a7d836e 交付时的历史状态和命令；M1.1 已更新集成脚本并得到成功穿袖记录。当前状态/命令见 [M11_REPORT.md](M11_REPORT.md)，新任务字段见 [M11_EPISODES.md](M11_EPISODES.md)。不要将下述旧集成命令当作新版入口。
当前数据为独立诊断及失败集成实验，**不是成功穿袖数据集**。远端根目录 `/root/gpufree-data/dynadress/artifacts/m1/`；本地仅同步轻量证据，完整逐帧布料 NPZ 留在远端。

每个 episode 包含 `metadata.json`、`samples.jsonl`、`result.json`、相机 PNG 和可选高维状态。`metadata.complete` 只表示记录正常关闭；任务成功必须另看结果，不能把 `diagnostic_complete` 当作穿袖成功。进程异常时未关闭的 metadata 保留 `complete:false`。

| 字段 | 语义 |
|---|---|
| physics_step | Isaac 物理步计数，必须严格递增 |
| control_step | 控制更新计数；独立人体测试 240 Hz，集成尝试 20 Hz |
| simulation_time_s | 从 World 读取的模拟时间，非循环次数推算 |
| sample_wall_time_ns | 读取记录时的主机 Unix 时间戳，不是实时运行承诺 |
| action | 已发送的双末端矩阵目标/推进量/夹爪保持指令，或外力矩脉冲；独立人体实验机器人字段为空 |
| solver_state | 求解器积分后的关节角、角速度、衣物/关键点位置等；集成末端跟踪误差来自实际关节 FK |
| solver_contact | 当前未导出 FEM 接触冲量/力，`available:false,value:null` |
| estimated_load | 当前未接入 traction，`available:false,value:null` |
| commanded_wrench | 显式人体关节力矩及重力补偿，均是命令；阻尼另由隐式速度 drive 执行 |
| measured_wrench | 没有真实传感器，`available:false,value:null` |
| privileged | 固定参考姿态、K/D 等仿真真值，与观测分开 |

五个来源组统一为 `{available: bool, value: object|null}`。某组可用不代表组内所有未来字段都有测量。机器人广义坐标包含 14 个 revolute 和 4 个 prismatic：前者 rad/rad·s⁻¹，后者 m/m·s⁻¹；顺序查看 H1 result 的 `joint_names`。人体顺序固定 `shoulder_x, shoulder_y, shoulder_z, elbow_y`，单位 rad/rad·s⁻¹；位置单位 m、力矩 Nm。

`response.npz:rows` 每行是 `step,simulation_time,q[4],dq[4],explicit_hold[4],gravity[4],external_pulse[4]`，240 Hz；JSONL/RGB/布料快照 10 Hz。`motion.npz` 是 H1 独立测试专用：`rows=step,time,actual_q,command_q`，另有 `joint_velocity`。H1 视频 20 Hz，人体/集成视频 10 Hz，播放速率对应模拟时间。

`cloth_*.npz:positions` 是原始 FEM 顶点世界坐标。集成 JSONL 的 `cuff_keypoints_m` 是整个选定边界环；抓持索引在 metadata。`cuff_progress.axial_progress_m` 仅为轴向几何代理，`completion:false`，尚无手部包含/穿袖成功判据。原生 vertex attachment 是人为夹持约束，不是基于摩擦验证的真实夹爪抓持；误差小不能证明实际夹爪夹持力合理。

所有采集脚本执行固定预算（人体 1440 物理步、H1 1440 步、集成最多 2880 步），避免无限制录制。当前降采样常数在各脚本中；尚未提供完整通用 RGB-D/高维采样配置接口，RGB-D 中只归档 RGB，不能宣称深度记录已交付。

批量入口：`bash scripts/m1_batch.sh`，依次运行 K=4/12 下的脉冲、无外力、FEM 接触和关闭人体碰撞对照。同一配置/命令的新进程相当于全场景复位；尚未实现集成任务的进程内 reset 接口。运行失败可保存未完成目录和 traceback；分析只选择正常关闭且具有修正物理时钟契约的实际数据。随机种子仅为确定性诊断标签，当前脚本无随机采样，不保证跨 GPU 位级确定性。
