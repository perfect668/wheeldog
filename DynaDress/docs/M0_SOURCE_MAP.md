# DynaDress M0：上游源码阅读与 M1 入口

阅读基线：PhyRC `48ffff25c9ff86cda3a2a21c7fc33d9022afa2fc`。本文件是静态源码分析，尚未经过本服务器仿真验证。以下路径相对 `upstream/`。

## 场景、机器人和控制

- `src/PhyRC_Sim/Env_StandAlone/Teleop_TShirt_Stretch4_Env.py:2225`：`TeleopTShirtStretch4_Env` 创建地板、桌子、单件蓝色 T 恤、人体、椅子和两台 Stretch4。代码存在旧的“四件衣服”注释，但当前循环只创建蓝色衣物；不要仅凭注释确定场景数量。
- 同文件 `spawn_stretch4:3140` / `configure_stretch4:3155`：引用 `assets/custom/robots/stretch_4/stretch_4.usd`，以 `base_footprint` 创建 `SingleArticulation`，配置关节/驱动、夹爪和连杆索引。`stretch4_tuning` 保存并重施控制参数。
- `drive_robot:4626`：9 维控制到关节目标和移动底盘伺服。`ContinuousClothControl.apply_base_drive` 的 Warp kernel 产生底盘驱动 wrench；它是**主动控制输入**，不是衣物接触力测量。
- M1：在独立 `dynadress/robots/` 实现 H1 适配，明确固定基座、双臂关节顺序、夹爪/腕相机坐标、末端与抓取点映射。当前 Stretch4 的 `(2,9)` 动作与 `(2,13)` 状态接口不能直接当作 H1 接口。本阶段没有实现这些内容。

## 抓取、FEM 和接触

- 主场景 `make_gripper_toggle:3809` 筛选附近衣物节点并维护开合意图；开合命令不等于真实抓取成功。
- `Env_Config/Garment/NativeGrasp.py:36` 的 `update_native_grasp`：`OmniPhysicsVtxXformAttachment` 将选定节点与 `grasp_center_link` 连接，默认 `anchor_count=12`，在抓取改变时重建。原生路径由 PhysX 联合 FEM/contact 求解，并非每步把所有布料直接跟随到目标位置。旧 checkpoint 和无 robot 的测试适配仍存在其他路径。
- 主场景 `follow_grabbed:4267`、`cloth_pre_step:4498`、`cloth_post_step:4516`、`initialize_manipulation:5018`：物理步前后回调。`SurfaceContactGuard.py`、`shield_cloth`、`guard_cloth_motion`、`prevent_cloth_blowup` 包含额外的碰撞/状态保护，需与原生求解行为区分，后续记录它们是否触发。不能将全部节点变化都解释为纯 FEM 求解输出。
- `Env_Config/Garment/Particle_Garment.py`：虽然保留旧名字，实际是 Isaac 6 **surface deformable / FEM**，`SurfaceClothPrim:41` 使用 `DeformablePrim(..., deformable_type='surface')`。`get_world_positions:68` / `get_velocities:83` 读取物理张量；位置/速度 setter 会改写状态，不能用于伪造交互数据。
- 同文件约 200–330 行创建表面 deformable、细分、CCD、自碰撞、求解迭代和材料。默认厚度 0.01 m；`reference_ratio=0.005/thickness`，Young 模量 `50000*reference_ratio`，泊松比 0.30，密度和弯曲参数还依赖资产 resize metadata。材料阻尼与速度阻尼不同。主场景设定迭代数，上游 smoke 预期为 64；实际材料以运行后 `output/verification/smoke.json` 为准，本机尚无该证据。
- `scripts/prepare_assets.py` 调用 `make_wearable_shirt.py` 生成衣服并规范人体/材质 USD metadata；`config/geometry.json` 与 `config/assets.lock.json` 必须共同保留。不要改变哈希或绕过校验。
- `HumanClothFriction.py`、`GripperClothFriction.py`、`TableClothFriction.py`、`ZeroSceneFriction.py`、`Human/HumanContactMaterial.py` 定义不同接触对的组合设置；M1 审核时不能只看一个摩擦系数。

## 人体

- `Env_Config/Human/Human.py:10`：加载静态人体 USD，设置 XForm 和 geometry collision，并处理单位/坐标轴。
- 主场景 `pose_arms_at_attention:1636`、`pose_legs_seated:3641`、`build_mitten_collider:3527` 和 `/World/Human/CollisionBody`：坐姿/手部姿态及碰撞网格处理；`HandSphereColliders.py`、`RoundHead.py` 是辅助碰撞/几何入口。
- 当前没有目标研究所需的单侧 4 DoF 被动动力学人体。M1 应在独立 `dynadress/human/` 明确肩部三自由度、肘一自由度、躯干固定、惯性、关节限制、阻抗参数和碰撞体更新，再接入衣物碰撞。先保留基线对照，不在原始源码里直接替换。

## 时钟、状态和图像

- `Env_StandAlone/BaseEnv.py:35`：默认物理 240 Hz，rendering dt=1/60 s；环境变量可覆盖，运行后需保存实际值。过时的“360 Hz”注释不能当作配置。
- `Policy/environment.py:18`：`DressingEnv` 为单场景 Gymnasium 适配；一进程一个 Isaac 实例。策略 20 Hz，控制 60 Hz，每动作三个控制步，默认十二物理步。`step:268`、`render:302`、`close:307` 是后续非 GUI 调用入口。
- `Policy/state.py:39`：`snapshot` 读取 live PhysX 关节、根姿态、连杆、衣物节点位置/速度/拓扑；人体/椅子静态姿态来自 USD。它没有现成的、校准过的衣物接触力真值流。
- `Policy/contract.py` 与 `config/policy_interface.json`：键名、shape、单位、wxyz 四元数、动作与观测白名单。
- `Policy/teleop_recording.py`：物理 tick 的状态档案与事件，分块 NPZ；回放是恢复记录状态，不是重新执行 FEM，不能拿回放视频充当新的交互验证。
- `Policy/training_teleop.py`、`training_deferred.py`、`training_dataset.py`：20 Hz obs/action/next_obs、同步相机与 HDF5；`policy.hdf5` 与 `audit.hdf5` 分开。必须核对 complete 标志和导出完成状态。研究数据可另立 schema，但不能无意混用竞赛允许观测和审计真值。
- `Policy/sensors.py:26`：5 个相机（overview、双腕、双头），Replicator render product + RGB / `distance_to_image_plane` annotator；`capture:98` 固定同一模拟时刻完成离屏渲染，输出 RGB `(5,256,256,3)`、深度/有效 mask `(5,256,256,1)`。`rep.orchestrator.step(delta_time=0.0)` 后检查时钟未前进。
- `scripts/export_policy_dataset.py`：离线导出入口；主场景 `start_recording:4836` 等为 viewport 录制路径，不能假设无 DISPLAY 就能使用该 GUI 录制接口。后续 headless 证据采集优先在独立脚本调用 `DressingEnv` + `RGBDSensors`，保存实际 PNG 和编码视频，同时保留模拟时间。

## 力数据语义：必须保留来源

`Env_Config/Garment/ContinuousClothControl.py` 明确说明这是控制器 operating limit，而非布料材料失效准则或校准力传感器。

- `ClothLoad.measure:56`：根据初始三角面与当前节点计算最大主拉伸应变，做面积加权 p95/p99。它是几何估计量。
- `ClothLoad.traction:75`：用跨抓取边界的拉伸边长度和面积权重估计阻力方向，累加类似 force/torque 的量，随后**除以线性部分范数**；松弛时还可退回“远离衣服中心”的方向。缺少物理材料标定、没有求解器接触冲量读取，因此返回值不是 N / N·m 的实测 wrench。
- `constrain_drive:117`：默认 p95 起止 0.315/0.45、p99 起止 0.495/0.72，带应变增长预测、方向性限速及 anti-windup；不是人体舒适力阈值。

后续数据字段至少区分：`solver_state`（物理节点/关节等）、`solver_contact`（只有验证过真实 API 的力/冲量才可进入，记录 dt/坐标/单位/接触对）、`estimated_load`（应变、归一化 traction 方向、compliance scale）、`commanded_wrench`（执行器控制输入）、`measured_wrench`（未来真机 F/T 传感器，记录标定、重力补偿和时间戳）。未知或不可获得的量标记 unavailable，不填零冒充真值。

进入 M1 前先完成 GPU 基线 smoke、实际图像/视频和资源测量，再按机器人适配 → 动态人体 → 接触核验 → 同步记录顺序修改。M0 没有实现任何世界模型、辨识或规划算法。
