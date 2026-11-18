# C2 数据与接口

冻结 M2.1 checkpoint 的 SHA256 为 `55b895974648a64a98910db5291a87363fe07b985c74ab70cad1ea6c42a89525`。未再训练网络，原状态109维、动作30维、8帧历史和动作时间对齐保持不变。下文路径以原生服务器项目 `/root/gpufree-data/dynadress` 为根；GitHub 位于 `DynaDress/` 下。

## 观测、真值与时序

- `state[t]` 在物理积分前读取，`action[t]` 是之后发出的目标，作用于下一间隔。短段起点 t 的模型输入为 `state[t-7:t+1]`、`action[t-8:t]`；5步候选是 `action[t:t+5]`，比较 `state[t+5]`。只有观测到该终点后才更新；下段起点为 t+5。
- 人体 q/dq、机器人关节/末端、衣物节点均来自原生仿真；它们在 C2 被视为可用观测，并不证明真机可直接测得人体 q/dq 或全部袖口节点。
- 腕载是原 `WristLoad` 由 PhysX incoming wrist reaction 与原生 link velocity/acceleration 做 Newton–Euler 补偿得到的**净外载估计**，世界坐标，腕 incoming joint 原点，力 N、力矩 Nm。包含附件、其他接触及模型误差，不是人体专属接触力或真实 F/T。`solver_contact`、`measured_wrench` 仍不可用。
- 人体保持命令、真实 K/D/μ、旧探测模板编号、工况名、episode ID 和结果不传入辨识器。已知共用 `q_ref=[0,1,0,0.35]` 可传入。真实属性只在环境配置、验证集标定的 oracle、离线评分和图标题中使用。

## 文件

每条原生回合保留于 `artifacts/c2/paired/<episode_id>/`：

| 文件 | 内容 |
| --- | --- |
| `config.json` / `metadata.json` | 环境真值、版本、源码及校准/checkpoint哈希；禁止整份作为辨识输入 |
| `samples.jsonl` | 原生20Hz动作、solver_state、estimated_load、缺失通道与 privileged 真值；沿用M2 schema |
| `observable_sequence.npz` | 仅 state、action、time；用于辨识与预测的紧凑序列，没有属性/模板/工况输入 |
| `belief.jsonl` | 每个不重叠短段起止时间、观测摘要、阶段、人体单通道与完整版本的 before/after 联合和边缘后验 |
| `probe_selection.json` | 唯一探测机会、所选动作、各候选的两类熵下降、归一化分数、IK/增量可行性 |
| `probe_proposals.npz` | 从当前目标、轴与观测历史构建的1秒候选动作；候选预测不等于之后的实测控制 |
| `c2_actions.jsonl` | 探测窗口实际发送的30维动作、跟踪误差与跟踪中断；后验使用这些实际动作 |
| `c2_result.json` / `result.json` | 辨识结果与原几何任务成功/失败分开保存 |
| `cloth_initial.npz` / `cloth_final.npz` | 原FEM初末状态；仅初始化时写入一次节点状态 |

`time` 和 belief 时间为回合相对模拟秒；原 `samples.jsonl` 的时间是 World 模拟时间，首样本通常约0.008333s，两者相差初始化偏置。RGB 的 `frame_index.jsonl` 保留原始模拟时间与物理步。

`artifacts/c2/evaluation/episode_metrics.csv` 是审阅主表，`metrics.json` 包含完整数组和来源哈希；36个 NPZ、beliefs JSON 和 actions JSONL 保留所有成败。评分脚本逐帧核对在线 state/action 与原始 solver/action 日志，避免记录错位。

## 后验与似然

网格按 `itertools.product(K,D,μ)` 排列，共50个联合假设。K为4/8/12/16/20 Nm/rad，D为0.5/2 Nm·s/rad，μ为0.1/0.2/0.3/0.5/0.8。K/D各复制到4关节。联合均匀先验为1/50；p(K,D)为10维，p(μ)为5维。衣物低/中/高有效滑动阻力分别聚合 μ≤0.2、μ=0.3、μ≥0.5；初始类别概率为0.4/0.2/0.4，类别不是均匀先验。

21维观测摘要为终点 q4、dq4、短段 native axial progress 变化1、短段平均有符号腕载12。腕载减去前8个有效观测的基线，模型与实测使用相同基线。似然分别对 q、dq、relative、load 组使用 ν=4 的多元 Student-t，组内对角尺度与共享径向重尾，组间相乘；对数域更新。尺度由旧验证集 oracle 残差的 median(|error|)/0.740697 计算，设置可见的最小模型误差尺度，没有用新36条或旧测试集调尺度。

5步更新和20步候选评分各有独立标定文件 `calibration/`、`probe_calibration/`。先观测到轴向进度>-0.125m，且 q偏移范数>0.002rad 或去基线力范数>0.5N，才开放人体似然；这是接触阶段代理。0.25s内轴向变化≥0.5mm时开放腕载/相对运动项。接触前保持先验。缺失的似然通道跳过；冻结网络仍需要完整历史，历史不完整时推理明确不可用，不填零伪造输入。

## 主动探测与 C3

`OnlineIdentifier.observe(state,t)` 只接收观测；`control(...)` 从同一后验评估回撤重推、暂停推进、滚转差动、常规推进；`sent(action,t,tracking)` 记录真实动作。每回合首个 insert/advance、可观测滑动且跟踪稳定窗口仅触发一次。各模式共享1秒机会、原25秒预算、原低层伺服和成功条件。幅度为原0.025m/s以内、最大2°滚转/±1.8mm差动；既有IK、关节速度和跟踪保护继续执行。

信息评分从后验抽参数及分组 Student-t 噪声，使用同一似然模拟后验，计算人臂及衣物边缘熵下降，各除以 log(10)、log(5) 后相加。候选使用相同随机样本；96次Monte Carlo估计保留有符号结果。优于继续推进不足0.01时继续推进。评分是冻结模型下的估计，不能当作实际获得的信息。候选用固定当前轴构建，实际伺服仍跟随实测几何，二者差异通过实际动作保留。

`dynadress.human.geometry.arm_landmarks(q, shoulder=...)` 返回世界坐标 wrist、elbow、forearm_axis、hand_tip；`axial_progress(q,cuff_center,...)` 返回 `(cuff_center-wrist)·forearm_axis`，单位m。前臂轴从腕指向肘。支持批量数组，不依赖Isaac，可供C3评分使用。C2后验继续使用冻结模型的原生progress递推，未替换成此几何计算。
