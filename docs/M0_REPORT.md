# DynaDress M0 部署报告

日期：2026-11-16（Asia/Shanghai）。状态：**无卡环境部分准备完成；原始仿真尚未通过 M0 验收，不能据此进入算法实验。**

远端路径：`/root/gpufree-data/dynadress/`。本地交付路径：`/home/user/桌面/DynaDress/`。本文不包含 SSH 密码；日志也不保存密码。

## 1. 服务器实际配置

| 项目 | 实测结果 |
|---|---|
| 系统 | Ubuntu 24.04.5 LTS / x86-64 |
| 内核 | `5.15.0-198-generic` |
| 实例形式 | 受限容器，PID 1 为 tini，非完整 systemd VM |
| 宿主 CPU | Intel Xeon Platinum 8358 @ 2.60 GHz；可见 128 logical CPUs、2 sockets、每 socket 32 cores / 2 threads |
| **本实例 CPU 配额** | `cpu.max = 50000 100000`，即 **0.5 CPU**；不能当作独占 128 核 |
| **本实例内存上限** | `memory.max = 2147483648`，即 **2 GiB**；`free -h` 的 1.0 TiB 是宿主视图 |
| GPU / 显存 / UUID | 无 `/dev/nvidia*`；无可验证 GPU，4090、显存及 UUID 均 unavailable |
| 驱动 | **无法检测**；`/usr/bin/nvidia-smi` 实为零字节可执行占位文件，运行不输出而返回 0，不能证明驱动正常；部分 libcuda / libnvidia 文件也为空 |
| 系统盘 | overlay 30 GiB 级别，检查时约 28 GiB 可用 |
| 数据盘 | `/root/gpufree-data`，`/dev/nvme11n1`、ext4、约 49 GiB 总量 / 47 GiB 可用 |
| 其他可见挂载 | `/data/tini` 等宿主相关挂载并不等于本项目可独占且持久的数据盘，未擅自用作大镜像存储 |
| 权限 | 无 CAP_SYS_ADMIN / CAP_NET_ADMIN，cgroup 不可写；`unshare -m` 失败；组合 `unshare -Ur -m true` 也因 root filesystem propagation 权限失败（exit 1）；无 `/dev/fuse`，没有宿主 Docker socket |

资源距离推荐至少 32 GiB RAM、100 GiB 可用 SSD 有明显缺口，不能靠 CPU-only 运行 Isaac/FEM 弥补。数据盘路径经平台登录提示确认：关机保留，但**释放实例会丢失，且不随系统镜像保存**。交付文档和日志同步到本地用于备份。

证据：`setup_logs/01_server_inventory.log`、`02_container_constraints.log`、`09_namespace_probe.log`、`23_resource_snapshot.log`。

## 2. 环境版本与上游追溯

| 项目 | 实际状态 |
|---|---|
| Docker Engine / CLI | 官方 Docker apt 源安装 `29.9.3`，build `5457199` |
| Buildx / Compose | `0.39.0` / `5.7.1` |
| containerd | `2.4.2`（包 `2.4.2-4~ubuntu.24.04~noble`） |
| NVIDIA Container Toolkit / libnvidia-container | `1.20.2-1`，`nvidia-ctk` CLI 1.20.2 |
| NVIDIA runtime 配置 | `nvidia-ctk runtime configure --runtime=docker` 已执行，`/etc/docker/daemon.json` 注册 `nvidia-container-runtime` |
| Isaac Sim 目标 | **6.0.1**，完全保留上游 Dockerfile |
| 固定 base | `nvcr.io/nvidia/isaac-sim:6.0.1@sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9` |
| PhyRC | `48ffff25c9ff86cda3a2a21c7fc33d9022afa2fc` |
| PhyRC 目标镜像 | `phyrc-2027:isaac-6.0.1`；**未构建成功，镜像 ID unavailable** |
| 原有预装 Isaac | 发现 `/root/isaacsim/VERSION = 6.0.1-rc.7+release.42383.32955d8d.gl` 和 python alias；未以它替代官方 Docker 构建，也未声称其可运行 |

初始 Docker / Toolkit 是未安装，后续已安装；当前是 **daemon / 容器权限失败**，不能继续称“未安装”。GPU 工具是占位，不能将其与真正的驱动安装相混淆。没有安装额外 CUDA Toolkit、ROS2、Conda，也没有替换系统 NVIDIA 驱动。

服务器原有 `/root/.gitconfig` 把 GitHub URL 重写到 `github.chenc.dev`。clone 实际经过该现有代理，`04_provenance.log` 记录 origin、rewrite 与 commit；未修改全局 Git 配置。后续第三方下载代理亦只用于传输，版本和原始 SHA-256 不变。

已读取 `README.md`、`docs/competition/SETUP.md`、`THIRD_PARTY.md`、`Dockerfile`、`run.sh`；另保存 NVIDIA 官方容器和 Toolkit 文档到远端 `docs/references/`。Docker 文档首次访问连接重置，安装步骤采用上游 SETUP 引用的官方 Docker apt 流程。

## 3. 实际执行与结果

所有下列结果均来自真实执行，而非推测成功。日志路径相对项目根目录。

| 命令/阶段 | 结果与日志 |
|---|---|
| `cat /etc/os-release; uname -a; nvidia-smi; free -h; df -h; docker --version; docker info; lscpu` | 完成初始检查；缺失工具与占位 GPU 工具如前述。`01_server_inventory.log` |
| `git clone https://github.com/rirolab/PhyRC_2027_Challenge.git upstream` | 成功，`03_clone.log`；commit / origin 见 `04_provenance.log` |
| `bash scripts/install_container_tools.sh` | 基础包成功，官方 Docker GPG 首次网络失败；重试和补充官方源安装后成功。`06`、`10`、`11`、`12` 日志保留初次失败及后续成功 |
| `nvidia-ctk runtime configure --runtime=docker` | 成功写配置；`13_runtime_config.log` |
| `systemctl enable --now docker` | systemd 非 PID 1，失败；同日志 |
| `dockerd --data-root=/root/gpufree-data/docker --pidfile=/tmp/dynadress-dockerd.pid` | daemon 因 iptables/NAT 权限失败；`14_dockerd.log` |
| 临时诊断 daemon：`--storage-driver=vfs --iptables=false --ip6tables=false --bridge=none --ip-forward=false --ip-masq=false --feature containerd-snapshotter=false` | 使用独立 `/tmp/dynadress-docker-probe.sock` 与 data-root 启动成功，仅用于根因诊断。`17_dockerd_diagnostic.log` |
| `docker -H unix:///tmp/dynadress-docker-probe.sock import /tmp/dynadress-probe-rootfs.tar dynadress-permission-probe:local` | **失败：`unshare: operation not permitted`**，exit 1；无需网络即可证明限制。`20_docker_namespace_test.log` |
| `docker run --rm --gpus all ubuntu:24.04 nvidia-smi` | 默认 daemon 不可达；`15_gpu_container_check.log`。诊断 daemon 上再次执行，拉取 Docker Hub 被 reset、exit 125；`21_gpu_diagnostic_daemon.log`。未识别到任何 GPU |
| `./run.sh doctor` | 依赖命令可见，但默认 Docker daemon 不可达，exit 1；`16_doctor.log` |
| `./run.sh build` | 默认 daemon 不可达，exit 1；`16_build.log`。诊断 daemon 再试，BuildKit bind mount `permission denied`、exit 1；`22_build_diagnostic_daemon.log` |
| `./run.sh prepare` | 先核对四项仓库内资产成功，Hugging Face SSL reset 导致 exit 1；`08_prepare.log`、`18_prepare_retry.log`。后续离线暂存与重跑见第 4 节 |
| `./run.sh smoke` | 未生成 runtime T 恤，打印 `Run ./run.sh prepare first.`，exit 2；`16_smoke.log` |

临时诊断 daemon 已发送 SIGTERM 关闭；没有把关闭网桥、vfs 等诊断设置写入正式 daemon.json，也未对宿主服务或驱动做替换。

## 4. 锁定资产与 prepare 进度

四项仓库内输入——衣服 USD、人体 USD、人体纹理、Stretch4 USD——已经由**原始 `fetch_assets.py`** 校验通过。Hugging Face 不可直连时，在本地使用 HTTP Range 从 `hf-mirror.com` 读取同一固定 commit 的 Scene.zip 中唯一所需纹理；仍严格核对原始锁文件 SHA-256：

`2K-tiling_30_basecolor.jpg`：`5e82c9839954fd25246ce9a9234b3bb2c05e175203ae80d09cf09693477768de`。

材质也固定到 DexGarmentLab commit `e4e298e696bae5d866ded3b31e0ae27becea5376`，经现有 GitHub 代理传输后必须核对原始哈希。辅助脚本 `scripts/stage_locked_downloads.py` 不修改上游锁文件或源码。

一次重跑与 SCP 传输重叠，原始校验正确拒绝了未传完的地板纹理（`24_prepare_staged_floor.log`）；随后等待传输完成、`sha256sum` 验证一致并重跑（`25_staged_floor_sha256.log`、`26_prepare_staged_verified.log`）。此失败未被删除或伪报为成功。

最后在服务器执行原始 `python3 scripts/fetch_assets.py --check`，**全部 8 项资产通过、exit 0**（`27_all_assets_check.log`）。再次执行原始 `./run.sh prepare`，8 项校验和 runtime 输入复制均完成，随后调用 Docker 生成 USD 时因默认 daemon 不可达而 exit 1（`28_prepare_final.log`）。因此，锁定输入已齐全，但 runtime T 恤 USD 和完整 prepare **尚未成功**。本地下载过程见 `local_asset_staging_final.log`。

## 5. Smoke、图片和视频证据

当前 **没有实际 `PHYRC-SMOKE-PASS`**。没有 `output/verification/smoke.json` 的通过结果，没有实际仿真 PNG / 视频，也没有 GPU 仿真资源曲线。上游 README 的 GIF 和示例视频不属于本次运行证据，未拿它们代替交付。

源码中的 smoke 实际检查初始化、衣服拓扑/64 次迭代、两个机器人空载运动、夹爪开合、保存/恢复和节点有限性；它本身不是高负载衣物抓取保留或完整穿衣成功认证。后续应保存真正 smoke.log 和 smoke.json，再补独立 headless 物理交互/相机采集。

## 6. 资源占用

`23_resource_snapshot.log` 为**安装/诊断阶段**采样，不是仿真数据：

- cgroup memory.current = 1,068,580,864 bytes，约 **0.995 GiB**（包括实例其他进程及缓存）；上限 2 GiB；memory.events 中 OOM / oom_kill 均 0。
- 诊断 dockerd RSS 94,044 KiB（约 91.8 MiB），containerd RSS 42,848 KiB（约 41.8 MiB），Jupyter RSS 105,748 KiB。
- CPU 累积 usage_usec=38,180,298，nr_throttled=676 / nr_periods=1731，存在 CPU 配额节流。单次 ps 的瞬时 %CPU 不能当作稳定仿真负载。
- 数据盘约 47 GiB 可用，上游目录约 180 MiB。Docker 诊断数据目录小于 1 MiB，未拉取巨大 Isaac base。
- 无 GPU，因此驱动、显存占用、仿真 FPS / real-time factor 均 unavailable。

## 7. 代码复用与数据语义

详细定位见 [M0_SOURCE_MAP.md](M0_SOURCE_MAP.md)。已完成静态阅读，未修改人体或衣物交互算法。最终 `git diff --exit-code` 返回 0、`git status --porcelain` 为空，见 `29_final_provenance.log`。

| 模块 | 主要入口（相对 upstream/） |
|---|---|
| 场景、机器人、控制 | `src/PhyRC_Sim/Env_StandAlone/Teleop_TShirt_Stretch4_Env.py` 的 `TeleopTShirtStretch4_Env` / `spawn_stretch4` / `configure_stretch4` / `drive_robot` |
| 抓取 | 同场景 `make_gripper_toggle` / `follow_grabbed`，`Env_Config/Garment/NativeGrasp.py` |
| FEM | `Env_Config/Garment/Particle_Garment.py`，`RefineSurfaceMesh.py`，`SurfaceContactGuard.py` |
| 人体碰撞 | `Env_Config/Human/Human.py`，主场景 `build_mitten_collider` / `pose_legs_seated` / `CollisionBody`，`HandSphereColliders.py` |
| 时钟 | `Env_StandAlone/BaseEnv.py`，`Policy/environment.py`；默认 physics 240 Hz / control 60 Hz / policy 20 Hz |
| 观测、记录 | `Policy/state.py`、`contract.py`、`teleop_recording.py`、`training_dataset.py`、`training_deferred.py` |
| 离屏相机 | `Policy/sensors.py`，`RGBDSensors.capture`，`DressingEnv.render` |

`ContinuousClothControl.py` 的 p95/p99 是面积加权几何应变；traction 是由拉伸边估算且**归一化**的阻力方向，松弛时可能退回几何中心方向，**绝不能写成真实接触力 N 或力矩 N·m**。其应变门限是控制 operating limit，不是布料失效或人体舒适力阈值。

后续需要分离：求解器状态、经验证的求解器接触力/冲量、估计负载、主动施加的控制 wrench、未来真机 F/T 传感器数据。现有 snapshot 没有提供已验证的衣物接触力真值接口；必须做 API/单位/坐标/时间步校验再设计标签。原生 FEM 外还有接触保护与状态修正，也应记录触发情况。

## 8. 已解决与未解决

已解决：实际资源识别、零字节 GPU 工具识别、持久数据盘选址、上游 commit 固定、官方容器软件包安装、NVIDIA runtime 注册、基础资产哈希验证、容器权限根因定位、独立研究目录与代码接口分析。没有修改上游 tracked source，也没有创建算法假实现。

尚未解决：

1. **实例权限**：常规 daemon 缺网络权限，诊断 daemon 仍缺 mount/unshare 权限。需平台提供支持 Docker 的 VM / 裸机、支持 NVIDIA 的特权容器模式，或明确开放宿主 Docker 通道；不能只在本容器安装包解决。无须先更换驱动。
2. **资源**：将实例内存提升至至少 32 GiB，数据盘扩大到至少 100 GiB 可用（建议留 150–200 GiB 余量）；单张 RTX 4090 挂载后实测 UUID / 24 GiB / 驱动，不能依据租用规格填表。
3. **网络**：Docker Hub 连接 reset，Hugging Face / GitHub raw 直连受限。小资产可用固定 commit + 原 SHA-256 的传输镜像解决；NGC base 仍需在目标端检查连通，若要求认证使用用户自己的 NGC 账号，不在脚本写凭据。
4. **仿真验收**：待上述条件满足，继续固定 6.0.1 build → prepare → smoke，随后实际采图/视频和 CPU/RAM/GPU 时序。此任务没有擅自改用 MuJoCo 或其他平台。

## 9. 进入 M1 的具体文件建议

先补齐 M0 的真实 GPU 验收，再进入 M1。独立目录 `dynadress/{simulation,robots,human,garments,data,configs}` 已预留，只有说明文档，无假实现。

1. 在独立场景适配中复用 `Teleop_TShirt_Stretch4_Env.py`、`BaseEnv.py` 的初始化/回调组织；保留原 checkout 为对照。
2. 以 `spawn_stretch4`、`configure_stretch4`、`drive_robot`、`NativeGrasp.py` 为映射参考设计 H1 固定基座双臂适配；**本阶段未导入 H1**。
3. 以 `Human.py` 和主场景人体姿态/碰撞体构建为替换边界设计 4 DoF 被动人体；把阻抗/惯性/限位与表面碰撞分开；**本阶段未修改人体动力学**。
4. 复用 `Particle_Garment.py`、各接触材质和 guard，先核验与动态人体接触，再做研究性参数实验。
5. 从 `Policy/state.py`、`sensors.py`、`training_dataset.py`、`config/policy_interface.json` 扩展有来源标签的同步记录 schema，区别 solver / estimate / real sensor。

`scripts/resume_m0.sh` 提供资源、GPU、Docker 和原始源码检查及逐阶段日志，已做 `bash -n`，**尚未在满足条件的 GPU 服务器端到端验证**。它不负责生成图像或视频，也不会把缺失证据报告为完成。

## 10. 验收清单与交付位置

- [x] 实际服务器资源/权限检查，区分宿主视图与配额。
- [x] Docker / NVIDIA Container Toolkit 官方包安装及 runtime 配置（**运行能力未通过**）。
- [x] 上游 Git commit 固定、tracked source 未修改。
- [x] 8 项锁定资产经原始脚本校验通过。
- [x] doctor / build / prepare / smoke 全部实际尝试并保存退出码。
- [x] 独立开发目录、可追溯配置、源码入口/力数据语义分析。
- [ ] Docker 正常运行容器、RTX 4090 透传：实例权限/硬件不具备。
- [ ] Isaac Sim / PhyRC 镜像构建：mount 权限失败，镜像 ID 不存在。
- [ ] runtime USD 生成、真实场景加载与 headless 物理：依赖镜像及 GPU。
- [ ] `PHYRC-SMOKE-PASS`：未出现，原始 smoke exit 2。
- [ ] 本次仿真图片/视频、仿真 CPU/RAM/GPU 占用：尚无真实运行，未伪造。

开发仓库分支为 `dynadress/m0`，初始证据提交 `30f70a63f616182edf7a032b60d1209ec283ce5a`；上游仍是独立 Git checkout，未提交研究修改。`docs/UPSTREAM_ASSETS.lock.json` 保存原始资产锁文件副本，`docs/DOCKER_RUNTIME_CONFIG.json` 保存实际 NVIDIA runtime 配置。

报告、源码分析、脚本和实际日志同时保存在远端和本地。上游完整仓库/资产位于远端 `upstream/`；本地只保留源码审阅快照与额外下载资产，不把该快照称为完整可运行 checkout。

下一步首先在平台侧解决 Docker 能力、单卡挂载、内存与磁盘，再运行 `scripts/resume_m0.sh`。任何图像/视频采集适配都应在独立脚本中进行，保留上游 smoke 原样。M1 的具体修改边界已列于第 9 节和源码分析文档，不应在补齐 M0 前提前编写辨识/世界模型/MPC。
