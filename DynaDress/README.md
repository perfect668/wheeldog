# DynaDress — 仿真开发基础

当前状态：**原生（绕过 Docker）部署已通过真实 smoke、headless 采集和 GUI 验证**。详见 `docs/M0_REPORT.md`。

M1.1：已在原生 PhysX 中完成第一条真实 H1 双臂夹持 → 手部穿入 → 前臂推进轨迹，标称任务 14.85 秒通过记录复核。使用原始 FEM、被动上肢和用户授权的仿真夹爪；初态为一次性预张开/预抓持，不包含桌面抓取。当前结果、准确命令和限制见 `docs/M11_REPORT.md`，数据字段见 `docs/M11_EPISODES.md`。历史 M1 诊断保留在 `docs/M1_REPORT.md`。没有学习算法。

远端工作目录：`/root/gpufree-data/dynadress/`。数据盘在实例关机后保留，但平台提示释放实例会丢失，且不会包含在系统镜像中；必须另做备份。

- `upstream/`：原始 PhyRC checkout，不修改 tracked source；远端保留 `.git` 与原始资产。当前本地目录为审阅源码快照，未复制大资产和上游 `.git`，不能直接声称本地可运行。
- `dynadress/`：独立机器人 IK、被动上肢、衣物拓扑及 episode 记录模块；没有世界模型或学习算法。
- `scripts/install_container_tools.sh`：按官方软件源安装工具（已在远端执行，网络重试和后续配置详见日志）。
- `scripts/stage_locked_downloads.py`：在可联网机器暂存锁定资产，严格核对原始 SHA-256；可选 HF 镜像仅用于传输。
- `scripts/resume_m0.sh`：供资源/权限/GPU 就绪后续跑；目前仅做 shell 语法检查，尚未完整执行验证。
- `setup_logs/`：实际命令日志（不纳入开发 Git）。
- `docs/M0_REPORT.md`：部署结果、证据和阻碍。
- `docs/M0_SOURCE_MAP.md`：模块位置、数据语义与 M1 修改入口。

上游：<https://github.com/rirolab/PhyRC_2027_Challenge>

固定提交：`48ffff25c9ff86cda3a2a21c7fc33d9022afa2fc`。
Docker 路线因服务器 daemon 权限不可用；本次按用户授权使用预装 Isaac Sim `6.0.1-rc.7+release.42383.32955d8d.gl` 原生运行，应用配置为 6.0.1。该版本未证明与固定 NGC digest 文件等价。FEM、上游源码和研究方向不变。
