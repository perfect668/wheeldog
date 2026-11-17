# GPU 挂载后复查（2026-11-17 09:30，Asia/Shanghai）

已实际 SSH 登录并检查。GPU 已可用，但 Docker 运行条件尚未满足；本次未启动 Isaac 仿真。

| 项目 | 实测 |
|---|---|
| GPU | 单张 NVIDIA L40（不是原计划 RTX 4090） |
| 显存 | 46068 MiB 总量；检查时使用 140 MiB，GPU utilization 0% |
| GPU UUID | `GPU-6ad2ec3e-6032-e3ff-70d5-a4e7375b8a9f` |
| 驱动 | 580.126.09；nvidia-smi 显示 CUDA compatibility 13.0，不代表安装了 CUDA Toolkit |
| CPU 配额 | `cpu.max=1400000 100000`，14 CPU |
| 内存配额 | `memory.max=128849018880`，120 GiB；检查时 memory.current=1375985664 bytes |
| 系统 | Ubuntu 24.04.5 LTS，kernel 5.15.0-198-generic |
| 系统盘 | 约 28 GiB 可用 |
| 数据盘 | `/root/gpufree-data`，约 47 GiB 可用，仍低于 100 GiB 建议值 |
| Docker | 29.9.3 已安装；`docker info` 无法连接 daemon |
| Toolkit | 1.20.2，NVIDIA runtime 配置保留 |
| 权限 | 仍无 CAP_SYS_ADMIN / CAP_NET_ADMIN；`unshare -m true` 与 `unshare -Ur -m true` 均 exit 1 |
| socket | `/var/run/docker.sock` 文件存在，但 daemon 不可连接，不能当作有效的宿主 Docker 通道 |
| 上游 | commit `48ffff25c9ff86cda3a2a21c7fc33d9022afa2fc`，`git status --short` 为空 |

原始完整日志：`setup_logs/32_gpu_recheck_20261117.log`（服务器端）。

结论：GPU、CPU 和内存资源已就绪；L40 与原计划 4090 的差异已如实记录，不擅自更改研究方案。仍需平台提供可运行 Docker 的权限/实例或有效宿主 Docker 通道，并将持久数据盘扩大到至少 100 GiB 可用，再继续原始 PhyRC Docker 构建与 smoke。没有更换驱动、技术栈或使用预装 Isaac 绕过上游流程。当前仍无本次真实 smoke pass、仿真图像和视频。
