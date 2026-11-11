# 可审计诊断记录

`M0_IMPLEMENTATION.patch` 是原始 upstream 678c5b4 到绘图修正版 d77d9c9 的完整代码/模型文档差异。正式主数据在 `results/exp0`，收敛数据在 `results/validation`。

`solver_preflight_errors.csv` 原样保留三个探索阶段的 17 条求解异常记录：系统 SciPy 1.8.0、未显式缩放、自动 HiGHS 选择时 14 条；缓存 SciPy 1.18.1 同设置时 2 条；显式缩放但仍自动选择时 1 条。这些返回 `Not Set`/unscaled model 诊断，全部曾分类为 `SOLVER_ERROR`，未计为不可恢复。每阶段 metadata 包含当时的参数、版本及未提交源码的 SHA256。这些是开发诊断快照，不是正式里程碑实验。原始完整探索输出仍留在本机被 gitignore 的同名目录中。

最终使用显式变量/行缩放 + HiGHS IPM，未放宽物理限制。预跑的成功和失败个数不是科学统计样本，报告只使用记录 ae0d139 源码哈希的正式数据。

`pytest.log`：最终源码提交前的 `env -u PYTHONPATH .venv/bin/python -m pytest -q`，41 passed。系统已有 ROS Python 路径会注入 pytest 插件，需要在独立虚拟环境运行时去掉 PYTHONPATH。`offline_install.log` 是从本机已有缓存重新封装的 wheelhouse 安装到干净 venv 的日志，没有继续下载依赖；`requirements-exp0.txt` 为实际版本。

项目获取最初遇到 github.com TLS 故障；使用 GitHub API 和 codeload 读取公开仓库，并按 GitHub 已签名 commit payload 恢复本地 Git 对象。blob/tree/commit 哈希与原始 upstream 一致，`git fsck` 验证连接完整。没有更改远端，也没有使用 H1 资产。
