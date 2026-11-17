# DynaDress M1 实测报告

> 本文保留 a7d836e 交付时的历史状态和命令；M1.1 已更新集成脚本并得到成功穿袖记录。当前状态/命令见 [M11_REPORT.md](M11_REPORT.md)，新任务字段见 [M11_EPISODES.md](M11_EPISODES.md)。不要将下述旧集成命令当作新版入口。
日期：2026-11-17。**结论：M1 部分完成，整体未通过。** 已实测用户提供的珞石双 AR5 整机模型、被动四自由度上肢以及原始 FEM 对动态 articulation 的接触响应。尚无可靠的机器人牵拉耦合证明、完整坐姿人体集成或成功穿袖 episode。下面的诊断通过不能替代整体验收。

## 1. 基线、资产与环境

- 远端工程：`/root/gpufree-data/dynadress/`；本地备份：`/home/user/桌面/DynaDress/`。
- PhyRC upstream commit：`48ffff25c9ff86cda3a2a21c7fc33d9022afa2fc`。保留 M0 prepare/smoke/GUI 路线；未修改 upstream tracked source，M1 结束再次检查。
- 原生 Isaac：`/root/isaacsim`，`6.0.1-rc.7+release.42383.32955d8d.gl`。不宣称与 NGC 锁定容器 digest 等价。没有安装其他仿真栈或更改 FEM 求解器。
- GPU：单张 NVIDIA L40，46068 MiB，驱动 580.126.09，UUID `GPU-0f2fe019-9da2-92c7-0bd5-edd867d3df1a`。不是最初计划的 4090。
- Ubuntu 24.04；14 个分配 CPU、120 GiB cgroup 内存上限（主机 `free` 展示约 1 TiB，不能当作实例额度）。数据盘约 49 GiB，仍有约 46 GiB 可用。Docker daemon 权限问题沿用 M0，全部实测使用原生入口。
- 真实模型来自用户 `/home/user/wheeldog/assets/h1/`；整机 URDF 包含左右各七轴 AR5、末端法兰，未发现夹爪。原始 zip SHA256：`5e40a5558cf3ed0dd59d1e9a08fa38944ddbb7c48e1612d0b7010a39cbeeb977`；原始 URDF SHA256：`484feff61bd4addd6aa949286c9f3e56c485c26000ad89ed0526eff3caec4aad`。
- 用户已允许添加**仿真假设双指工具**；不是原厂夹爪。衍生任务 URDF 固定非双臂自由度、固定底盘，增加两侧各两个 prismatic 手指；原始资产不改。掌体 0.025×0.06×0.04 m/0.2 kg，每根手指 0.01×0.008×0.07 m/0.04 kg，开合行程 0–0.035 m。
- 模型来源可追溯，但 URDF 与手册关节 7 方向/限位、力矩、质量等存在待核实项。`assets/h1/manual_evidence.json` 记录手册声明及不确定性，本轮依据用户提供 URDF 运行，不冒称实机批次标定完成。

## 2. 实际实现

| 入口 | 实现及范围 |
|---|---|
| `dynadress/robots/prepare_h1.py` | 包哈希校验、网格唯一解析、惯性/限位审计、衍生固定底盘 URDF 和明确标注的工具 |
| `scripts/m1_import_h1.py` | 安装环境中的 Isaac 6 URDFImporter API，生成 USD；不使用宇树模型 |
| `dynadress/robots/h1.py` | 源 URDF FK、双七轴位姿 DLS IK、关节界限和命令速度限制 |
| `scripts/m1_h1_motion.py` | 实际双臂和工具运动、源限位检查、实际速度、live link pose 与 FK 交叉核对、自碰撞接触对 |
| `dynadress/human/passive_arm.py` | 世界固定 anchor、肩 XYZ 三关节及肘 Y 一关节；胶囊上臂/前臂/手 |
| `dynadress/configs/m1_passive.json` | 几何、质量、初态、固定参考姿态及阻抗配置 |
| `scripts/m1_human_test.py` | 无衣物保持/外力矩脉冲/撤力恢复及原始 FEM 落衣接触与关闭碰撞对照 |
| `dynadress/garments/cuff.py` | 从原始三角网格提取边界环；轴向进度明确仅为代理 |
| `scripts/m1_coupled_pull.py` | 真实 H1＋被动上肢＋原始 FEM 同场景集成尝试、原生 vertex attachment、限幅 IK 和失败终止；尚非完整穿袖状态机 |
| `dynadress/recording/episode.py` | 流式 JSONL、严格递增时钟和明确缺失值/来源，真值另存 privileged |
| `scripts/m1_batch.sh` | 两种 K 下四种诊断条件的有限批次 |
| `scripts/m1_analyze.py` | 只分析实际完成且具有修正时钟契约的实验，输出曲线/数值 |
| `scripts/m1_encode.py` | ffmpeg 编码、ffprobe 帧数、完整解码和哈希 |
| `scripts/m1_resources.py` | 有限时间采集实际进程 RSS/CPU tick 和 GPU 显存/利用率 |

当前没有统一机器人运行时 reset/service 类：IK 可复用，执行仍由独立脚本调度；重新运行进程重建/复位场景。未实现完整的碰撞规避规划、人体限位退让、停滞监测和手部包含判据。不能把这些入口描述为完整 M1 控制框架。

## 3. 被动上肢动力学

物理步 240 Hz。肩固定在世界 anchor 上，肩 XYZ/肘 Y 限位分别为 ±100°、[-150°,70°]、±100°、[0°,145°]。上臂 0.30 m/半径 0.045 m/1.8 kg，前臂 0.26 m/半径 0.035 m，前臂与手合计 1.0 kg，手为前臂固定胶囊。两肩轴载体有 0.001 kg 正质量和正惯性，避免零质量退化。这些是**仿真假设参数**，不是已识别人类生物力学参数；手质量/惯性采用集中近似，尚未与坐姿人体表面拟合。

每步从 PhysX articulation 读取广义重力补偿 `g(q)`，发送裁剪后的 `g(q)+K(q_ref-q)` 显式关节力矩；`-D*dq` 通过**零位置刚度、零目标速度的隐式 force velocity drive**执行，避免再显式叠加阻尼。张量 API 使用 rad 单位；USD 初始 angular damping 按 degree 换算。实际读取 drive gains 检查 stiffness=0、damping=D；不存在额外位置 drive 刚度。显式 effort 上限 [30,30,20,20] Nm，隐式阻尼 drive maxForce=100 Nm；两者是独立限制，**不是合力矩硬限幅**。

每个诊断 `q_ref=[0,-1.3,0,0.3]` 固定，D 四关节均为 1 Nms/rad，K 四关节分别均设 4 或 12 Nm/rad。只在初始化设置关节状态，响应由动力学积分产生；无逐帧姿态覆盖。脉冲测试仅 shoulder Y 从 1.0–1.5 s 施加 1 Nm，显式命令单独记录。

## 4. 实测结果与证据

### H1 独立双臂运动

最终工具尺寸下的 GUI 运行：`artifacts/m1/h1_motion/20261117T124541/`，日志 `setup_logs/m1/22_h1_gui_final_tools.log`，退出码 0，`passed:true`。

- live 末端到目标位置误差：左 3.767 μm、右 3.504 μm；live 与源 FK 差 < 0.83 μm。
- 关节限位违规：无；源 URDF 速度限位违规：无。实测峰值约 0.282 rad/s，命令限幅 0.25 rad/s，动态响应存在超调，不能把命令限幅说成实际严格限速。
- 自碰撞保持启用，32 个网格强制 convex decomposition。记录到了腕 link7 与 link5 的瞬时接触，因此本测试不是全工作空间无碰撞证明。
- 该运动位于独立可达姿态附近，不证明同一袖口双臂工作区可达性或夹持稳定性。

### 固定时钟被动响应

第一组已验证结果来自 `setup_logs/m1/18_clock_contract.log`；所有 1440 步真实模拟时长 6.000000313 s。对应目录：

| 条件 | run id | shoulder Y 最大偏移 | 最终 shoulder Y 误差 |
|---|---|---:|---:|
| 脉冲 K4 | `pulse/k4/20261117T122640` | 0.225741 rad | 0.0000713 rad |
| 脉冲 K12 | `pulse/k12/20261117T122708` | 0.114128 rad | 0.0000273 rad |
| FEM 接触 K4 | `contact/k4/20261117T122737` | 0.089669 rad | -0.0000848 rad |
| 关闭人体碰撞 K4 | `contact/k4/20261117T122839` | 4.77e-8 rad | 4.77e-8 rad |
| FEM 接触 K12 | `contact/k12/20261117T122942` | 0.040471 rad | 0.0000618 rad |

图：`artifacts/m1/analysis/responses.png`、同名 PDF 和 `analysis.json`。两种阻抗的脉冲响应均恢复；匹配接触/关闭碰撞控制的四关节响应最大差范数 0.092824 rad。该对照支持原始衣物接触可向动态上肢传递反作用，且刚度变化影响响应。衣物本身由原始物理求解器运动，没有脚本施加接触推力。

最终完整批次 `setup_logs/m1/batch_k*.log` 八项均已实际执行结束：K4 脉冲/无外力/接触/关闭碰撞 run id 依次 `20261117T124932`、`20261117T125002`、`20261117T125031`、`20261117T125133`；K12 依次 `20261117T125235`、`20261117T125303`、`20261117T125332`、`20261117T125434`。每项均 finite、记录关闭、1440 步/6.000000313 s；实际读回 drive stiffness 全零、damping 全 1。无外力和脉冲项 `passed:true`，接触项仅由匹配对照分析，不独立填通过。此次落衣 shoulder Y 峰值 K4=0.089298、K12=0.040606 rad，略有 GPU 求解差异；最终图选择此最新批次。

最新批次 K4 接触/关闭碰撞最大响应差范数为 0.092427 rad。已检查最终 PNG 曲线、H1 实际渲染截图和失败集成截图；H1 视频 120 帧/20 Hz，人体诊断每项 60 帧/10 Hz，均做完整解码校验。完整证据轻量备份为 `/tmp/m1-final-evidence.tgz`（约 6.1 MiB），已同步本地相同 artifact 目录；包含结果、JSONL、响应 NPZ、视频和首/中/末帧。全部逐帧 PNG 与布料 NPZ 留在远端，未重复复制大型资产。

**证据范围是无机器人牵拉的落衣实验**；还没有机器人移动袖口条件下的匹配接触对照、准确接触力或持续穿透深度统计，不能据此标记 M1-C 全通过。

### 机器人牵拉集成：失败证据

| 日志/run id | 实际结果 |
|---|---|
| `19_coupled_pull.log` / `coupled_pull/20261117T123314` | 9.55 s 时 IK 不可行，未证明手部穿入 |
| `20_open_cuff.log` / `coupled_pull/20261117T123806` | 6.85 s 时 IK 不可行，附件误差约 6–9 mm，但末端跟踪有问题 |
| `21_guarded_pull.log` / `coupled_pull/20261117T124356` | 增加实际跟踪保护后，0.10 s 发现约 17.5 cm 偏移并终止；`success:false` |
| `25_separated_cuff.log` / `coupled_pull/20261117T125601` | 改为不重叠相反侧 patch、增加物理速度限制后，0.10 s 左侧约 34.8 mm 偏移，仍保护终止；原生附件最大误差约 18 μm，不能把附件稳定当成机器人跟踪通过 |
| `27_unattached_only.log` / `coupled_pull/20261117T125829` | 不创建抓持附件的定位对照，仍因实际跟踪误差终止；不能只归因为抓持约束 |
| `28_robot_filter_control.log` / `coupled_pull/20261117T130000` | 仅定位用：不抓持并过滤衣物—机器人碰撞，运行 1 s，末端误差不超过约 27.2 mm；不能用于接触/抓持验收 |
| `29_normal_jaw_slow.log` / `coupled_pull/20261117T130202` | 夹爪开合方向转至衣物法向，开袖降低至每侧 10 mm/s；恢复完整碰撞/抓持，仍在 0.10 s 跟踪保护终止 |

第三次只录到两帧，短视频不能被用作成功牵拉证据。早期按扁平袖口 Z 极值选点导致两点过近，后续改为相反 Y 极值并检查 patch 不重叠、间距至少 6 cm；增加机器人真实接触对和 PhysX 速度限制用于进一步定位。不能通过关闭机器人自碰撞、放宽跟踪容差或脚本拖动人体伪造成功。

`26_unattached_control.log` 保留一次失败诊断：直接关闭 FEM mesh 通用 collision 属性导致 deformable tensor view 无法初始化，退出码 1。已撤销该方法；最终源码不以此作为无接触控制。

因此，衣物—机器人接触/初始布局是具体排查方向，但尚未证明唯一根因，也尚未修复。过滤对照仅创建 `UsdPhysics.FilteredPairsAPI`，只允许和 `--diagnostic-unattached` 同用；默认任务路径保留衣物—机器人碰撞。没有 FEM 接触力输出时，空的 rigid contact report 不能解释为“衣物没有接触”。当前还存在近奇异姿态下的速度限幅跟踪滞后，需联合解决布局、真实工具间隙、IK 可操作度和接触稳定性。

集成场景当前只放置被动上肢，未加载坐姿假人/躯干，也没有移除原假人对应侧碰撞；完整人体外观与无重复碰撞验收未做。grasp 使用原生 `OmniPhysicsVtxXformAttachment` 在实际腕 link 上约束衣物顶点，采用物理邻近检查；这不是经过摩擦/夹持力验证的原厂夹爪抓持。

所有完成诊断的目录有 `diagnostic.mp4`、`video_probe.json` 和 `evidence.sha256`。保留失败视频，无标称成功穿袖视频。视频只呈现实际计算结果，没有补帧合成成功场景。

## 5. 原生复现命令

以下在远端执行；每次只运行一个 Isaac 进程，上游文件锁阻止并发占用。所有 M1 程序新进程重新建场景；GUI 诊断按预算运行后自动关闭。

```bash
cd /root/gpufree-data/dynadress
python3 dynadress/robots/prepare_h1.py --help
python3 dynadress/robots/prepare_h1.py assets/h1
./scripts/run_native.sh python /root/gpufree-data/dynadress/scripts/m1_import_h1.py --decomposed
./scripts/run_native.sh python /root/gpufree-data/dynadress/scripts/m1_h1_motion.py --force-decomposition
DISPLAY=:20 ./scripts/run_native.sh python /root/gpufree-data/dynadress/scripts/m1_h1_motion.py --force-decomposition --gui
./scripts/run_native.sh python /root/gpufree-data/dynadress/scripts/m1_human_test.py --stiffness 4 --no-pulse
./scripts/run_native.sh python /root/gpufree-data/dynadress/scripts/m1_human_test.py --stiffness 12
./scripts/run_native.sh python /root/gpufree-data/dynadress/scripts/m1_human_test.py --stiffness 4 --contact
./scripts/run_native.sh python /root/gpufree-data/dynadress/scripts/m1_human_test.py --stiffness 4 --contact --disable-human-collision
./scripts/run_native.sh python /root/gpufree-data/dynadress/scripts/m1_coupled_pull.py --open-cuff
bash scripts/m1_batch.sh
python3 scripts/m1_analyze.py
python3 scripts/m1_encode.py artifacts/m1/h1_motion/<run-id> --fps 20
```

运行日志应按阶段重定向到 `setup_logs/m1/`。`prepare_h1.py` 参数以脚本实际 `--help` 为准。独立人体验证加 `--gui` 可观察，M0 原场景仍用 `./scripts/run_native.sh gui`。M1 集成脚本的结果预期可能是安全终止，不能以进程退出码 0 判断穿袖成功。

## 6. Torch 修复与故障定位

M0 `torch_library_fix` 是运行时动态库搜索顺序适配，完整实现在 `scripts/run_native.sh`：

1. `LD_LIBRARY_PATH` 先放 Isaac `kit/python/lib`、`kit`、`kit/lib`，避免系统 libpython 3.12.3 与自带 3.12.13 混用。
2. 随后加入 `/root/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/torch/lib`，及同一 pip_prebundle 下每个 `nvidia/*/lib`；使用 Isaac 自带 Torch/CUDA 依赖，不复制随机系统库，不安装 CUDA Toolkit。
3. `PYTHONPATH` 使用上游 `sitecustomize`、独立 `native/python` 和上游 scripts。原生 Python 包依赖由 `scripts/install_native_dependencies.sh` 按上游 requirements 安装；没有依赖桌面终端的隐藏 export。
4. 原始容器绝对路径以保护性软链接适配：`/isaac-sim`、`/workspace`、`/project`、`/scripts`、`/output`；既有异目标会拒绝覆盖。M0 资产校验继续保留。

其他已修复问题：Isaac 6 importer API 变化；实例代理必须先收集路径再 deinstance；碰撞 hull 造成不合理持续碰撞，改为实际强制 decomposition；GPU FEM 必须从 World 构造时统一 torch/CUDA 后端；肩关节直接接世界会丢一个 DOF，改为固定 anchor；显式阻尼在小惯量轴不稳，改成检查过的隐式零刚度 drive。

**时钟纠错：**早期 `world.step(render=True)` 按 render dt 推进，循环索引不能当作物理时间。现有脚本全部改为 `step(render=False)`，相机使用原始 sensors 的零时间渲染；人体验证逐步断言时间。日志 01–17 是保留的排错记录，不能作为最终六秒响应数据。分析器排除无 `clock_contract` 的旧数据。

## 7. 数据与验收缺口

资源实测：`setup_logs/m1/resources_final.jsonl`（120 个约 1 Hz 样本）及 `resources_summary.json`。该窗口覆盖被动/FEM 批量诊断：峰值 GPU 显存 4054 MiB，GPU 利用率 92%；单个 Isaac 进程峰值 RSS 6.362 GiB。按 `/proc` CPU ticks 与真实采样间隔计算，活跃进程样本平均约 361%、峰值约 1425%（100%=一个 CPU 核）。这是有限采样窗口，**不是完整任务或 GUI 全程峰值**；进程 RSS 不是整个 cgroup 总内存。

字段说明：`docs/M1_EPISODES.md`。缺失接触力/测量力/traction 明确标记 unavailable；不把 PhyRC traction 命名为真实力。RGB 及小规模布料状态已保存；RGB-D 通用采样、完整动作增量/实际夹爪状态语义、碰撞反作用力提取仍需补齐。没有基于可靠力测量的真实力阈值保护。

| 验收项 | 状态 |
|---|---|
| 用户提供 H1 固定底盘加载、双臂 IK、独立运动视频/误差/限位 | 独立测试通过；全工作区和手册差异未解决 |
| 四自由度被动力学、固定参考、两种阻抗、外力/恢复曲线 | 独立诊断通过 |
| 原始 FEM 对动态上肢反作用 | 落衣对照检测到；尚无机器人牵拉对照证据 |
| 双臂同袖口稳定且物理合理抓持 | 尝试过，未通过完整验收 |
| 坐姿假人外观、移除对应原手臂碰撞 | 未实现 |
| 确定性穿袖、包含判据、终点/退让状态机 | 未通过；只有有限轨迹尝试与失败终止 |
| 同步 episode、缺失值与来源分离、批量入口 | 已实现诊断版本；完整穿袖字段未齐 |
| 标称与扰动穿袖视频 | 无成功视频，保留失败证据 |

## 8. 后续具体修改

1. `dynadress/robots/h1.py` 和 `scripts/m1_coupled_pull.py`：在真实腕/工具碰撞几何下寻找双臂不同腕向、离散 IK 多种子及碰撞可行姿态；验证静态夹持负载后再推进。不能只追求 IK 残差。
2. `dynadress/garments/cuff.py`：保持真实边界拓扑，按弧长选择不重叠抓持 patch，增加袖口面/手部包含、穿透深度及单调进度检查。
3. `dynadress/human/passive_arm.py` 和新的场景装配模块：把肩 anchor 对齐原坐姿模型，明确关闭被替换手臂的刚体碰撞，核对手部质量惯性和表面连续性，增加人体限位退让。
4. `scripts/m1_coupled_pull.py`：先完成匹配初态/动作的接触开关与 K/D 对照，再加入对齐/进入/推进/完成/退让状态机及停滞、数值异常、相对运动监控。
5. `dynadress/recording/episode.py`：实际接口齐全后增加两末端增量、工具状态和可用接触来源；实现统一可配置 RGB-D/布料采样和进程内 reset。保留特权真值与未来控制器观测边界。

在上述实测完成前，不进入世界模型、在线辨识网络或 MPPI 开发。

源码追溯：`docs/M1_SOURCE_MANIFEST.json` 保存最终源码 SHA256，原始资产单独校验。早期诊断是在未提交开发树运行，日志的 source_commit 是 upstream commit，不是 M1 代码 commit；不能声称所有历史运行与最终源码完全一致。编号日志和失败路径保留修改过程；最终提交用于继续开发与复现。

本地 `python3 -m unittest discover -s tests -v` 的三项测试实际通过，覆盖被动力矩/无效状态、缺失测量/时钟约束、真实资产的可达 IK 与命令速度限制；不是 PhysX 集成测试。Python 编译、批量 shell 语法与 Git 空白检查通过。物理仿真通过/失败以本报告对应真实日志和结果文件为准。
