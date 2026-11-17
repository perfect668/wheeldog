# GitHub 轻量审阅交付

本分支只提交当前 DynaDress 源码、报告、汇总曲线、代表性截图及重画曲线必需的小型数值数据。**M1 整体未通过**；真实结果及缺口见 `M1_REPORT.md`。

## 保留内容

- `DynaDress/` 保存独立工程，原仓库内容保留；本次提交第二父提交保留 DynaDress 本地开发历史。
- `docs/`：M0/M1 报告、来源、原生配置、数据字段及研究方案。
- `dynadress/`、`scripts/`、`tests/`：当前实现、配置、运行/分析入口和测试。
- `assets/h1/`：用户提供的原始 URDF、衍生任务 URDF、审计 JSON 和手册证据；不提交 mesh、zip、USD。
- `artifacts/m1/analysis/`：两种阻抗的响应 PNG/PDF 和分析数值。所选八项最终诊断仅保留 `response.npz`、metadata/result JSON，使曲线可以重新生成。
- 代表性截图：H1 `h1_motion/20261117T124541/frame_0119.png`、失败集成 `coupled_pull/20261117T130202/frame_0001.png`、落衣初态及 M0 全景。H1 motion NPZ 和失败 episode 小型 JSONL 支持审阅跟踪误差和数据来源。
- 只保留 M0 smoke JSON、最终版本核对及资源统计摘要；不提交完整运行日志、视频、逐帧录制或布料状态快照。
- `docs/DELIVERY_MANIFEST.json`：本次实际提交文件的大小及 SHA256（不含清单本身和 submodule 内文件）。

## 基线与运行

`upstream/` 使用 PhyRC submodule，固定 `48ffff25c9ff86cda3a2a21c7fc33d9022afa2fc`。

```bash
git clone --branch codex/dynadress-m0-m1-delivery --recurse-submodules git@github.com:perfect668/wheeldog.git
cd wheeldog/DynaDress
python3 scripts/m1_analyze.py
```

完整证据仍保存在服务器 `/root/gpufree-data/dynadress/` 和本地备份；原报告中的绝对路径描述实际服务器现场，部分目录不在本次轻量上传范围。没有删除原始证据。

本地源码提交 `56e441b`，服务器源码提交 `15db07bf15cef5e7b060ac3b194aeb6dd359f6bd`，M1 源码 SHA256 一致。服务器原生 Isaac 路径为 `/root/isaacsim`，版本 `6.0.1-rc.7+release.42383.32955d8d.gl`。详细原生启动命令见 `NATIVE_SETUP.md` 和 `M1_REPORT.md`。

在新机器运行物理仿真，需要恢复用户提供的 H1 mesh/zip、重新生成绝对路径 URDF/USD，并按原上游锁下载资产及配置 Isaac。当前分支定位为代码与结果审阅，不能宣称 clone 后已具备完整运行环境。
