# PhyRC 原生部署入口

适用于本服务器预装的 `/root/isaacsim`（6.0.1）和当前独立开发目录；不需要 Docker。此路线由用户在 2026-11-17 授权，是原始 Docker 路线之外的适配。

## 启动

在远程桌面终端中执行：

```bash
cd /root/gpufree-data/dynadress
./scripts/run_native.sh gui
```

脚本默认桌面 `DISPLAY=:20`。点击 Isaac 视口获得键盘焦点；`ESC` 退出，`P` 重置。机器人 1 控制使用 W/S、A/D 等，机器人 2 使用数字键盘及方向键；具体按键保持上游 README 定义。只有一个 prepare / smoke / GUI / capture 可以运行，脚本复用上游文件锁。

通过 SSH 执行原始 headless smoke：

```bash
cd /root/gpufree-data/dynadress
./scripts/run_native.sh smoke
```

成功必须同时出现 `PHYRC-SMOKE-PASS`、退出码 0，并保存 `/root/gpufree-data/dynadress/upstream/output/verification/smoke.json`。它测试初始化、空载动作、夹爪切换、保存/恢复，不代表成功抓住衣服或完成穿衣。

## 重建准备流程

```bash
cd /root/gpufree-data/dynadress
./scripts/install_native_dependencies.sh
./scripts/run_native.sh prepare
./scripts/run_native.sh smoke
```

原始八项资产已齐全；prepare 使用原始 `fetch_assets.py --check` 和 `prepare_assets.py`。如果缺少输入文件，应先按资产锁重新下载并校验，不能关闭校验。

## 实际路径适配

上游脚本中包含容器绝对路径，因此脚本创建以下软链接（只在不存在或指向同一目标时使用，拒绝覆盖其他路径）：

| 绝对路径 | 当前目标 |
|---|---|
| `/isaac-sim` | `/root/isaacsim` |
| `/workspace` | `/root/gpufree-data/dynadress/upstream/.runtime` |
| `/project` | `/root/gpufree-data/dynadress/upstream` |
| `/scripts` | `/root/gpufree-data/dynadress/upstream/scripts` |
| `/output` | `/root/gpufree-data/dynadress/upstream/output` |

源资产保持原样，派生 runtime 在 `.runtime/`。独立 Python 包位于 `native/python/`，与 Isaac 原有 Python 包分离；版本来自上游 `requirements-runtime.txt`。复用上游 `sitecustomize.py`，不另造 PhysX 替代品。

动态库顺序是本实例的必要适配：优先 Isaac 的 `kit/python/lib`，避免系统 libpython 3.12.3 与自带 3.12.13 扩展混用；并显式加入 Isaac 的 Torch lib 和同包 NVIDIA 库目录，解决 CUDA 线性代数 dlopen 问题。没有替换驱动、系统库或修改 Isaac 源码。

## 真实证据采集

```bash
./scripts/run_native.sh python /root/gpufree-data/dynadress/scripts/native_capture.py
./scripts/encode_native_capture.sh /root/gpufree-data/dynadress/upstream/output/native_capture/<run-id>
```

脚本执行新的物理步，不恢复录制轨迹；用原始 `DressingEnv`、5 路 RGB-D 相机和小幅升降动作验证。保存相机 PNG、同一模拟时刻的多相机拼图、状态端点与逐帧时间戳，再用 ffmpeg 编码视频并用 ffprobe/解码检查。视频播放帧率 20 Hz，对应模拟时间，不能据此推断实时速度。

`native_probe.py` 仅用于运行库和接口诊断；它的成功字段不能代替上游 smoke，日志中首次诊断进程有清理断言，已保留真实失败日志。

## 限制

- 预装版本完整标识为 `6.0.1-rc.7+release.42383.32955d8d.gl`，应用配置为 6.0.1。原生路径无法证明文件内容等同于上游固定 NGC Docker 镜像 digest；Docker 镜像构建验收仍未完成。
- 当前卡为单张 L40，不是最初计划的 RTX 4090。
- 数据盘约 47 GiB 可用；本次复用预装 Isaac 足以执行小规模验证，但录制/训练数据仍应扩容至至少 100 GiB 可用。
- 平台释放实例会丢失数据盘；文档、脚本和运行证据需备份。系统重置后需恢复脚本创建的路径软链接和原生依赖目录。
- 保留原始 PhyRC 源码为对照，不在本阶段实施 H1、动态人体、世界模型或 MPC。

## 场景审计证据

`native_scene_capture.py` 使用额外全景和衣物近景相机，仅用于核验原始场景，不改变物理或策略相机。实际运行生成 21 帧并输出 `NATIVE-SCENE-AUDIT-PASS`；视频位于 `upstream/output/native_scene_audit/<run-id>/phyrc_scene_audit.mp4`。
