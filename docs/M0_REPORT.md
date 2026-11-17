# DynaDress M0 部署报告

日期：2026-11-17（Asia/Shanghai）。本报告记录在远程服务器上**实际执行**的原生（绕过 Docker）验证；上游源码保持不变。

## 结论

原生 Isaac Sim 路线已完成可复现的 PhyRC 基线验证：`prepare`、原始 `smoke`、headless 多相机采集和远程桌面 GUI 均真实退出码 0。smoke 日志出现 `PHYRC-SMOKE-PASS`，并生成 `smoke.json`。本路线使用服务器预装 Isaac Sim，不等同于上游固定 NGC Docker digest；Docker 由于服务器容器权限仍不可用。

## 服务器与版本

|项目|实测|
|---|---|
|系统|Ubuntu 24.04.5 LTS，x86-64，kernel 5.15.0-198-generic|
|CPU|Intel Xeon Platinum 8358；本实例 cgroup 配额 14 CPU|
|内存|cgroup 上限 120 GiB|
|GPU|单张 NVIDIA L40，46068 MiB，UUID `GPU-6ad2ec3e-6032-e3ff-70d5-a4e7375b8a9f`|
|驱动|580.126.09，CUDA compatibility 13.0|
|磁盘|数据盘约 49 GiB，总可用约 47 GiB；小规模验证可用，后续建议至少 100 GiB|
|PhyRC|commit `48ffff25c9ff86cda3a2a21c7fc33d9022afa2fc`|
|Isaac|预装 `/root/isaacsim`，`6.0.1-rc.7+release.42383.32955d8d.gl`；应用配置为 6.0.1|

Docker Engine 29.9.3、NVIDIA Container Toolkit 1.20.2 已安装，但 daemon/build 受限于缺少 `CAP_SYS_ADMIN`/`CAP_NET_ADMIN` 和 mount/unshare 权限，不能作为本次运行路径。未替换驱动，未安装 ROS2、Conda 或额外 CUDA Toolkit。

## 实际执行结果

|阶段|结果与证据|
|---|---|
|资产检查/prepare|原始 8 项锁定资产 SHA-256 通过；原生 prepare exit 0，生成 `t_shirt.usd`/`t_shirt_short.usd`|
|原始 smoke|`setup_logs/native/11_smoke_torch_library_fix.log`；`PHYRC-SMOKE-PASS`，exit 0；`smoke.json` 为 `passed:true`|
|GUI|`setup_logs/native/14_gui.log`；Isaac Sim 6.0.1 窗口加载完整场景，ESC 干净退出 exit 0；截图见 `artifacts/native/isaac_gui.png`|
|headless 采集|81 帧、5 路 RGB-D，4.0 s 模拟时间；PNG、状态端点和 H.264 视频已验证完整解码|
|场景审计采集|额外全景/衣物相机，21 帧；日志含 `NATIVE-SCENE-AUDIT-PASS`，exit 0；未修改物理或原始策略相机|

Smoke 实测摘要：两台 Stretch4、1 件 FEM 衣物、15878 vertices、31344 triangles、64 solver iterations；夹爪切换、空载关节运动和 save/load 通过，布料状态有限。`grasp_attached_in_initial_pose=false`，因此不能把该 smoke 描述为真实抓取或完成穿衣。

## 证据路径

远端根目录：`/root/gpufree-data/dynadress/`。本地备份：`/home/user/桌面/DynaDress/`。

- Smoke：`artifacts/native/verification/smoke.json`、`setup_logs/native/11_smoke_torch_library_fix.log`
- GUI：`artifacts/native/isaac_gui.png`
- 多相机视频：`artifacts/native/native_capture/20261117T094945/phyrc_native_multicamera.mp4`
- 全景与衣物近景：`artifacts/native/native_scene_audit/20261117T095510/phyrc_scene_audit.mp4`、`audit_scene_0000.png`、`audit_garment_0000.png`
- 资源：`setup_logs/native/resource_summary.json`；采样范围含桌面、Jupyter、shader 冷启动和运行阶段，峰值 cgroup 内存 17.30 GiB、Python RSS 16.37 GiB、GPU 显存 10168 MiB、GPU 利用率 91%、功耗 182.29 W，CPU 区间峰值约 14.4 cores（配额内突发采样）。

## 代码阅读与 M1 入口

详见 [`docs/M0_SOURCE_MAP.md`](M0_SOURCE_MAP.md)。已定位 Stretch4 articulation/夹爪、surface FEM、人体碰撞体、240 Hz 物理与 60 Hz 控制、RGB-D 接口、状态快照和记录模块。`ContinuousClothControl.py` 的 traction 是应变几何估计/控制限制，不是真实接触力；后续数据必须区分 solver state、solver contact、estimated load、commanded wrench 和 measured wrench。

## 已完成与未完成

已完成：服务器复查、锁定资产校验、原生依赖隔离、prepare、原始 smoke、headless 图像/视频、GUI、资源采样、源码模块定位；上游 tracked source 未修改。

未完成：固定 NGC Docker 镜像构建（权限阻碍）；无法声明预装 Isaac 与 digest 文件完全等价；数据盘低于长期数据建议；smoke 本身不证明衣物抓取/穿衣成功。L40 取代了原计划 RTX 4090，未改变研究方向。

## M1 建议

保持 `upstream/` 作为只读基线，在 `dynadress/robots/` 接入 H1 固定底盘双臂，在 `dynadress/human/` 建立单侧 4 DoF 被动人体，在 `dynadress/garments/` 复用现有 FEM 并先核验接触 API，在 `dynadress/data/` 设计带来源和时间戳的状态记录。不要直接改写上游 Stretch4、人体或 FEM 实现。

启动原生 GUI：

```bash
cd /root/gpufree-data/dynadress
./scripts/run_native.sh gui
```
