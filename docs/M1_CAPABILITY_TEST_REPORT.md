# M1-C 测试与数值复核

实际pytest输出：`60 passed in 15.06s`。完整日志：`results/m1c/pytest.log`。精确依赖见`requirements-m1.txt`；使用Python 3.11.15，CPU。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-m1/bin/python -m pytest -q
env -u PYTHONPATH .venv-m1/bin/python scripts/validate_m1c_numerics.py results/m1c/run
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-m1/bin/python -m h1_capability.verification results/m1c/run --workers 4
env -u PYTHONPATH .venv-m1/bin/python scripts/freeze_m1c.py results/m1c --verify
```

原48项M0/M1测试保留；新增12项覆盖：参数/单位/顺序/原资产不变，接触符号与虚功，共享躯干负载求和，偏置扶手与底盘雅可比差分，解析力LP/任务不可行/无界/预算单调性，双扶手共同twist，移动包含固定解，样条段内极值，延迟与静止边界，有限时间路径与NPZ一致性，12维匹配与六方向评分、示例对不重叠、损坏轨迹被拒绝。

| 检查 | 实际结果 |
|---|---|
| 偏置扶手+底盘J，ε=1e-6 | 最大绝对差 3.40021344e-10 |
| 1512次静力LP最大力矩/等式残差 | 2.84217094e-14 Nm / 0 N |
| 初始42配置最大位置/角误差 | 1.83634109e-08 m / 1.09929371e-07 rad |
| 主1512+敏感性720数值求解异常 | 0 / 0 |
| 发布案例样条重构与约束复核 | 2232，失败 0 |
| 代表配置加密几何采样 | 936，每段8分点 |
| 加密位置/角误差最大值 | 6.69898461e-06 m / 2.50691661e-06 rad |
| 连续坐标速度/加速度最大比 | 0.793016597 / 0.999999401 |
| 加密采样最低记录间隙 | 0.00336557357 m |
| 原冻结数据 | M0 48项 + M1 26项SHA256均一致 |

全部案例找到验收路径包含静止解，不能写成100%非零运动成功率。局部搜索路径拒绝不等于全局不可行；开发TypeError与预检日志保留。密集采样与样条重放仍是同一模型内检查，不是连续实体碰撞证明、硬件误差测量或因果反馈验证。负例测试、正式实验与开发预检分开保存。

源码来源、模型边界、完整数据索引和审阅问题见`M1_CAPABILITY_RESULTS.md`。
