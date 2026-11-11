"""Generate reviewer-facing reports from retained M1 data (no new experiment)."""
from pathlib import Path
import json
import subprocess
import numpy as np

out=Path('results/m1/run')
read=lambda p:json.loads(Path(p).read_text())
a=read(out/'asset_audit.json');s=read(out/'summary.json');m=read(out/'metadata.json')
manual=read('assets/h1/manual_evidence.json');replay=read(out/'replay_validation.json')
candidates=read(out/'candidates.json');deriv=read(out/'differential_checks.json');targets=read(out/'targets.json')
code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
fmt=lambda x:', '.join(f'{v:.7g}' for v in x)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])

joint_rows=[]
for j in a['joints']:
    l=j['limit'];bounds='—' if l is None else f"[{l.get('lower','—')}, {l.get('upper','—')}]"
    joint_rows.append([j['name'],j['type'],j['parent']+' → '+j['child'],fmt(j['origin_xyz_m']),fmt(j['origin_rpy_rad']),fmt(j['axis_joint_frame']),bounds,'—' if l is None else l.get('velocity','缺失'),'—' if l is None else l.get('effort','缺失')])
link_rows=[]
for l in a['links']:
    i=l['inertial']
    link_rows.append([l['name'],'缺失' if i is None else f"{i['mass_kg']:.9g}",
                     '—' if i is None else fmt(i['origin_xyz_m']),
                     '—' if i is None else fmt(i['principal_kg_m2']),
                     '—' if i is None else ('零张量' if i['mass_kg']==0 else 'PSD/三角不等式通过'),l['visual_count'],l['collision_count']])
mesh_rows=[[x['link'],x['triangles'],str(x['watertight']),fmt(np.diff(np.array(x['bounds_link_frame_m']),axis=0)[0])] for x in a['meshes'] if x['kind']=='collision']
link_to_joint={j['child']:j['name'] for j in a['joints']}

audit=f'''# M1-A：H1 提供资产审计

日期：2026-11-11。范围：用户提供资产的文件审计及 M1-B 运动学接入；不包含动态施力模型。

## 结论与证据边界

资产可形成完整树并进行条件性运动学计算，但**不能认定它已经是经厂家确认的本机 H1 模型**。用户提供的压缩包名含 `0.8/AR5`，URDF 名为 `Wheel_Robot`；手册为 Helios 系列，未提供对应本机序列号或模型发布说明。第 7 轴定义、质量和力矩字段冲突需要厂家确认。软件加载成功或差分误差小，不会消除这些语义问题。

- 压缩包：`1.5整机urdf-0.8AR5-20260626.zip`，17,096,353 bytes，46 ZIP 条目、解压 49,250,191 bytes；SHA-256 `{m['archive_sha256']}`。
- 原始 URDF SHA-256：`{a['urdf_sha256']}`。原压缩包和 mesh 留在本机 `assets/h1/`，Git 保留审计和哈希，未重复下载资产或执行附件中的脚本。
- 产品手册：DOC-00003457/A，修订表 2026-03-15；58 页，SHA-256 `{manual['sha256']}`。部分偶数页页眉为 DOC-00003456/A，已记录文档内部不一致。
- 手册的数值、页码、语义限定见 `assets/h1/manual_evidence.json`。所有 URDF 原始惯性矩阵、关节信息、mesh 映射、逐文件哈希见 `results/m1/run/asset_audit.json`。
- 数值源码：`{m['source_commit']}`；报告/重放/绘图工具提交：`{code_commit}`。

## 树、运动学链与坐标系

34 links / 33 joints，根为 `base_link`；24 个转动自由度：4 continuous 轮关节、6 个编号 revolute 关节、左右臂各 7 revolute；另有 9 fixed。无浮动底盘关节、无 prismatic、无 mimic、无闭链约束。

左右臂挂接到 `4_Link`：根链 `base_link → 1_Link → 2_Link → 3_Link → 4_Link` 后，经 `left_fixed/right_fixed` 接到各臂 base，再经过 joint_1…joint_7，最后 fixed 法兰。

- 左安装：xyz `[0, 0.0775, 0.1429] m`，rpy `[-1.57, 1.57, 0] rad`。
- 右安装：xyz `[0, -0.0775, 0.1429] m`，rpy `[1.57, 1.57, 0] rad`。
- `5-Joint/6-Joint` 所在分支从 `4_Link` 延伸，经 `6_Link` 后 fixed 到 `7_Link…11_Link`；其与手册“头/颈”的编号映射尚待确认，未据外观自动命名。
- 实际末端 frame 为 `AR5-5_08L-W4C1C9-ZY2_flan_link`、`AR5-5_08R-W4C1C9-ZY2_flan_link`；相对 link7 平移 `[0,0,0.09] m`，无旋转。没有夹爪、扶手、TCP 或接触传感器 frame/mesh。
- 本轮虚拟扶手 frame 与各法兰取恒等变换，属于明确的建模代理，不是已标定的实物扶手。
- URDF 单位按标准取 m/rad/kg/kg·m²；姿态为右手系、固定轴 RPY，`R=Rz(yaw) Ry(pitch) Rx(roll)`。joint axis 在 joint frame；固定关节的零 axis 不参与转动计算。原始近似角 `1.57/1.5708` 未替换为精确 π/2。

下表覆盖全部关节；完整精度保存在 JSON。effort 对本资产的转动关节按 URDF 为 N·m，velocity 为 rad/s；**这些只是文件字段，不是已验证的本机驱动极限**。

{table(['joint','type','parent → child','origin xyz [m]','origin rpy [rad]','axis','position [rad]','velocity [rad/s]','effort [N·m]'],joint_rows)}

## 惯性、质量与 mesh

文件已声明质量合计 **{a['declared_mass_sum_kg']:.9f} kg**。四个轮 link 各为 **144.494198764598 kg**，合计约 577.977 kg；这与手册整机约 190 kg 显著不符。不能把该总和当真机质量，不能据此计算抗倾覆、重力力矩或剩余力裕度，也不擅自缩放到 190 kg。是否导出时重复计入了装配质量需厂家解释。

两个臂 base 缺 inertial；两个法兰为零质量/零惯性 fixed frame。其余声明惯性张量均为对称、半正定且主惯量满足三角不等式（采用 1e-12 kg·m² 数值容差）；数学合法性不能证明参数真实。每侧已声明 7 个臂 link 质量合计 9.466 kg，手册 AR5 约 11 kg；部件覆盖差异未解释。

{table(['link','declared mass [kg]','inertial origin xyz [m]','principal inertia [kg·m²]','algebraic check','visual','collision'],link_rows)}

有 32 个 collision mesh，合计 887,499 个三角形，其中 17 个不是 watertight。visual 与 collision 均有 32 份引用，依靠唯一 basename 可以解析到所附全部 32 个文件，但原包路径**不可直接认为有效的 ROS package URI**：混用了 `Wheel_Robot`、带空格的 `Wheel_Robot 0.8`、`robot_model_v2` 及实际不存在的 `AR5_P_L/AR5_P_R` 子目录。另有 `material name=white` 无顶层定义；不影响本轮几何。

底盘/躯干 mesh 未设 scale，按 m；机械臂显式 `0.001 0.001 0.001`，按原始 CAD 数值缩放到 m。没有自行修正比例。下表是应用各自 URDF scale 和 geometry origin 后、link frame 中的包围盒尺寸；逐引用 URI/实际路径/哈希/上下界见 JSON。

{table(['collision link','triangles','watertight','bbox extent [m]'],mesh_rows)}

## 手册对照与冲突处置

页码同时给出 PDF/印刷正文定位；原表已渲染目视核对。

| 项目 | 手册声明及位置 | URDF/本轮处置 |
| --- | --- | --- |
| 底盘结构 | PDF27/印刷23：4 组舵轮；PDF29/印刷25：全向、3 DOF、最大 1.5 m/s | URDF 只有 4 个固定轴向 continuous 轮旋转，缺舵向、轮地约束/驱动映射。确认“手册声明全向”，仍不能用现 URDF 验证舵轮运动。只做静态 SE(2) 摆放 |
| 整机尺寸 | PDF27/23：680×640×1736 mm（站直）、高度1142 mm（蹲下）、930 mm（运输） | 仅对照，未据此变形/标定 URDF；world z=0 为参考 root，不是地面 |
| 质量 | PDF29/25：约190 kg | 与声明质量739.844 kg冲突，动力学使用受阻 |
| 单臂能力 | PDF29/25：5 kg 负载、650 mm 工作半径；PDF34/30：最大臂展683（该单元未印单位）、约11 kg | 工作半径/最大臂展口径不同，未强行等同；5 kg 不转成任意方向扶持力能力 |
| 臂位置范围 | PDF34/30：±178°, ±120°, ±178°, [-60°,145°], ±178°, ±60°, **±90°** | 前6轴与URDF舍入近似一致，第7轴URDF±1.0472 rad≈±60°。保留URDF较窄范围，未扩大 |
| 臂速度 | PDF34/30：175/175/270/300/240/240/240 °/s | 与URDF臂 velocity 数值换算一致，但并非动态响应/控制频率，未用于本轮轨迹 |
| 关节力矩 | PDF37/33 §3.7.4.4：182/182/134/134/35/35/35 N·m | URDF为113/113/70/70/19/19/19 N·m。两套值都保留；持续/峰值和版本关系未知，不选择有利数值 |
| 手腕载荷 | PDF36/32 §3.7.4.1：轴5允许力矩12 N·m/惯量0.33 kg·m²；轴6为4.5/0.10 | 属负载安装约束，与关节驱动力矩限额语义不同，未混用 |
| 第7轴方向 | PDF34/30 §3.7.2文字：机械零位A1/A3/A5/**A7**沿臂基座+Z，A2/A4/A6沿+Y；PDF35/31有坐标图 | URDF第7轴为local +X，沿其前链在臂零位仍为臂基座+X（前链origin rpy全零）。不能由软件差分验证消除此冲突；未擅自改为Z |
| 传感与接口 | PDF29/25：关节内置扭矩传感器、双编码器、EtherCAT，C++/Python SDK，ROS2 | 不等于已配置末端六维F/T或可用接触力观测；未编写驱动 |
| 控制频率/响应 | 本次所给手册未找到对应数值 | 延迟、控制周期、加速度、力变化率、故障后的负载保持等继续未知 |

`config/joint_names_Wheel_Robot.yaml` 只有轮关节与编号关节，且首项为空，缺全部14个臂关节；不作为完整控制配置。附件中的 launch/CMake/export.log 仅作为资料，不执行。

## 碰撞审计与需要厂家补充的信息

32 个 mesh 对有 496 对组合；55 对基于拓扑排除（同一 fixed 刚性组或直接连接的刚性组），441 对参与检查。完整 pair/reason 在 `collision_policy.json`，**没有因为零位碰撞而自动免检**。先用包围凸体做保守分离检查，距离小于2 mm或重叠时，以原始三角网格做FCL复核；腕部link5/link7的凸包误报及真实约6 mm表面间隙见 development 数据。2 mm是数值筛选条件，不是厂家规定或安全间距。

通过仅表示上述 pair 的所附表面无相交且达到筛选间隙；开口mesh、包容而不穿过表面的情形、被排除的安装相邻体、缺失的夹爪/扶手/地面/床/人体均限制其解释。未验证连续路径，也未评估载荷或接触稳定性。

进入任何动态阶段前，最少需要：确认本机型号/序列号及正确模型版本；确认第7轴方向/零位和限位；修正轮/整机惯性与臂base质量；提供舵轮转向/驱动/轮地模型及躯干编号/限位；解释手册与URDF力矩值关系并给持续/峰值与时长条件；提供SDK控制模式、采样/延迟、加速度与制动约束；夹爪/扶手真实安装变换、载荷与接触测量条件。本轮不等待这些资料来完成文件级审计，但它们阻止把结果当成真实H1能力结论。
'''
Path('docs/M1_ASSET_AUDIT.md').write_text(audit)

accepted=[r for r in candidates if r['status']=='ACCEPTED_CONDITIONAL']; rejected=[r for r in candidates if r['status']=='REJECTED']
fd_rows=[]
for step in sorted({r['epsilon_rad'] for r in deriv},reverse=True):
    entries=[r for r in deriv if r['epsilon_rad']==step]
    fd_rows.append([f'{step:g}',f"{max(r['jacobian_linear_max_abs'] for r in entries):.9g}",f"{max(r['jacobian_angular_max_abs'] for r in entries):.9g}"])
fail_rows=[[r['id'],r['solver']['status'],r['solver']['nfev'],f"{max(r['position_errors_m']):.9g}",f"{max(r['orientation_errors_rad']):.9g}",'; '.join(r['reasons'])] for r in rejected]
versions=', '.join(f"{k} {v}" for k,v in m['versions'].items())
kin=f'''# M1-B：双臂运动学与冗余配置验证

日期：2026-11-11。完成范围是**提供的URDF内的静态几何建模**，不是经真机标定的H1运动学验证。资产存在第7轴方向/限位等冲突，详见 [M1_ASSET_AUDIT.md](M1_ASSET_AUDIT.md)。停在 M1-A/B，不开展动态施力、MPC、RL、VLA、接触仿真或真机控制。

## 源码、环境与复现

分支 `m1/asset-kinematics`，基于已审阅 M0 提交 `d6ede91783c71da8cf5e5f064db3d42640eed225`。
数值实现初始提交 `3b4053c`，正式数值源码提交 **`{m['source_commit']}`**；该次运行 tracked 工作区干净。后续报告/重放/绘图工具提交 `{code_commit}`。完整数据与报告交付提交见本轮 Git 历史（不把文档自引用哈希写入自身）。

Python {m['python']}，{m['platform']}。主要依赖：{versions}。必要依赖锁定在 `requirements-m1.txt`；完整已安装包清单 `results/m1/environment_packages.json`。本机复用了 rl 环境中的MuJoCo/数值库和本地trimesh；仅补缺python-fcl wheel，不使用GPU、服务器或动态步进。

先按 `assets/h1/README.md` 将同一用户资产放在本地。新环境安装（可优先用现有本机环境/缓存）：

```bash
python3.11 -m venv .venv-m1
env -u PYTHONPATH .venv-m1/bin/python -m pip install -r requirements-m1.txt
env -u PYTHONPATH .venv-m1/bin/python -m pip install -e . --no-deps
```

本次实际运行：

```bash
env -u PYTHONPATH .venv-m1/bin/python -m pytest -q
env -u PYTHONPATH .venv-m1/bin/python -u -m h1_kinematics --config configs/m1_kinematics.yaml --output results/m1/run --seed 20261111
env -u PYTHONPATH .venv-m1/bin/python -m h1_kinematics.plots results/m1/run
env -u PYTHONPATH .venv-m1/bin/python scripts/validate_m1.py results/m1/run
.venv/bin/python scripts/verify_frozen_exp0.py --output results/m1/frozen_exp0_integrity.json
```

重现时把 `--output` 改为新的空目录（如 `results/m1_reproduction`），绘图/重放也使用新目录。程序拒绝覆盖既有非空实验目录。`env -u PYTHONPATH` 是为隔离本机ROS Humble的Python3.10路径。运行记录 `results/m1/run.log`、`pytest.log`、`replay.log`、`plots.log`。原始EXP0的48个哈希核对全通过，未覆盖原数据；终端判据与重放含义见 [M0_REVIEW_ADDENDUM.md](M0_REVIEW_ADDENDUM.md)。

## 数学模型与约束

对每条树边，`T_parent,child(q)=T_origin R_axis(q)`，沿树连乘；固定关节不加旋转。库为MuJoCo 3.10.0的URDF导入、`mj_kinematics`、`mj_comPos`、`mj_jacBody`。内存适配仅移除visual/collision并禁用fixed体融合以保留全部frame；原始关节和惯性字段不修改，独立保存在 `kinematics_only.urdf`。MuJoCo编译器会作其内部表示转换；没有用它推断或验证惯性，也没有调用forward/step或设置控制器/执行器。

世界坐标系定义为参考 `base_link` 坐标系，SI、右手系；静态底盘变量 `b=[x_b(m),y_b(m),yaw(rad)]`，`z=roll=pitch=0`。world z=0 **不是地面**。手册支持全向3DOF的产品声明，但现URDF缺转向/驱动映射，因此这里只枚举静态摆放，不能解释为瞬时自由漂移、可执行路径或动力学可达性。

q按 `configurations.npz/joint_names` 的24个URDF转动关节排序，单位rad。IK优化 `1-Joint…4-Joint` 与双臂14个关节，共18个关节；轮角和 `5-Joint/6-Joint` 固定为0。每次IK的底盘摆放固定，由外层采样；同一目标也有12次固定零底盘的多起点尝试。搜索盒 x±0.06 m、y±0.04 m、yaw±0.10 rad，只是数值实验采样域，不是厂家行程/速度限值。

左右虚拟扶手均与其法兰取恒等变换。固定共同目标由YAML中明确的参考q正运动学生成：

- 左位置：`[{fmt(np.array(targets['world_targets'][0])[:3,3])}] m`。
- 右位置：`[{fmt(np.array(targets['world_targets'][1])[:3,3])}] m`。

完整两个4×4目标（**含朝向**）及参考q见 `targets.json`，在所有候选间不改变；左/右自然是不同位置，要求是每一侧在不同机器人配置下保持同一世界位姿。没有为增加成功率逐样本改变目标。

双臂残差为 `e_i=[p_i-p_i*, 0.25 Log(R_i R_i*ᵀ)]`，用有界SciPy TRF least-squares最小化两个残差平方和。Log为世界表达旋转向量，0.25 m是位置/旋转残差的数值权重，不是机器人尺寸或物理阻抗。求解器用三点数值导数；主残差评估预算160次，`ftol=xtol=gtol=1e-10`；SciPy的nfev不含全部数值Jacobian额外函数调用。

后验独立按每侧位置2范数≤1e-5 m、SO(3)角度≤1e-5 rad、原URDF关节限位（1e-10 rad数值容差）、所附mesh表面间隙≥0.002 m筛选。所有试次保存初值、最终q、base、solver原始status/message/nfev、误差、碰撞对/间隙、筛选原因和时间。连续运动路径、环境碰撞、重力/稳定性与厂家真实动态限制不在此判据。

雅可比按 `[world linear; world angular]` 排列。每侧完整J是6×24，臂自身子块6×7；每侧上游4个编号关节参与，另一臂、轮与末端无关分支的列为零。FK与独立按原URDF变换连乘交叉核对；J用中心差分：位置 `(p(q+ε)-p(q-ε))/(2ε)`、角速度 `Log(R(q+ε)R(q-ε)ᵀ)/(2ε)`。检查所有24列，无base导数（base摆放是外层变量）。`jacobians.npz` 保存10个合格随机样本的完整与6×7子块，线性部分单位m/rad，角度部分rad/rad。

碰撞检查详见审计报告：55对安装拓扑排除、441对检查；凸包可分离时用其下界，接近时用原三角表面复核。`min_distance_m`混合保守凸包距离下界及复核表面距离，不能宣称是全模型精确最小净空；相交时-1是哨兵而非穿透深度。部分mesh开口，因此 **ACCEPTED_CONDITIONAL** 仅表示该已声明几何约束检查通过，不是完整实体无碰撞证明。碰撞只做后验筛选，不是IK连续约束优化。

## 自动测试与实际数值

`pytest` **48 passed**，包括原M0的41项和M1的7项。M1覆盖资产完整性/mesh解析、保留源惯性、FK独立实现、双侧J及非作用列、碰撞与间隙的负例、改变base后的双臂IK、不可达目标的局部失败/输入与限位检测、不可变导入及损坏文件检测。单元测试中的负例不是主实验数据。

固定seed={m['seed']}：全URDF关节限位内均匀随机采样18个选定关节，共试 {s['random']['count']} 次，得到10个所声明条件内合格样本，另13个筛除（13个有表面相交，其中1个也有其他pair间隙不足，原因可重叠）。采样在得到10个合格样本后停止（预算250次），不是固定样本量的物理成功率估计。轮/5/6关节固定0，未采样全部硬件状态，也无地面，因此图中某些低位姿态不能作为实机工作姿态。

10个合格样本分别检查3种差分步长，原始30行在 `differential_checks.json`。独立FK全34个frame的最大位置差 {s['validation_maxima']['fk_position_max_m']:.9g} m，最大SO(3)角差 {s['validation_maxima']['fk_rotation_max_rad']:.9g} rad。

{table(['ε [rad]','max |ΔJ_linear| [m/rad]','max |ΔJ_angular| [rad/rad]'],fd_rows)}

48个双臂多起点IK，42个 **ACCEPTED_CONDITIONAL**、6个 **REJECTED**、0个数值异常；42个关节配置之间最小18维L2角距离 {replay['min_accepted_joint_l2_separation_rad']:.9g} rad，包含 {s['distinct_base_count']} 种不同静态base（其中11个合格样本base=0，另31个base不同）。不只是重复保存同一q。

合格样本最大每侧位置误差 **{s['target_position_max_m']:.9g} m**，最大姿态误差 **{s['target_orientation_max_rad']:.9g} rad**。这是数值误差，不是真机精度（手册重复定位指标不与之等同）。正式程序墙钟 {s['elapsed_s']:.6f} s，包含资产审计、随机检查与搜索，不含启动/后续绘图/重放。计时依赖本机负载，不能当实时控制性能。

局部求解41次status=1（gtol），7次status=0（预算耗尽）；其中1次预算耗尽但已满足几何误差要求，按后验判据通过；其余6次均因目标位置和朝向超差筛除。没有把“局部超预算未达标”写成“全局IK不可行”。

{table(['candidate','solver status','nfev','max position [m]','max rotation [rad]','filter reasons'],fail_rows)}

保存后71个记录（48候选+23随机）重新检查，状态/原因不符={replay['status_mismatches']}；NPZ/JSON完全一致。这是同一模型的重放一致性，不是物理验证。离线静态IK没有扰动/反馈信息，不构成EXP0的因果控制验证或H1恢复边界结果。

## 图与原始数据

所有图来自真实保存的q/误差/状态；机器人图用原始mesh的凸包简化显示，碰撞检测的三角复核与图形显示不是一回事。不存在人体或硬件实验曲线。

![共同世界法兰目标下的三个实际求得配置](../results/m1/run/figures/redundant_configurations.png)

![10组随机条件性合格配置](../results/m1/run/figures/random_validation.png)

![双臂差分误差、所有IK状态及静态base分布](../results/m1/run/figures/kinematics_diagnostics.png)

- `asset_audit.json`：逐link/joint/mesh、所有源文件哈希。
- `targets.json`, `config.yaml`, `metadata.json`：目标、参数、seed、源码、依赖。
- `candidates.json/.csv`, `configurations.npz`：全部48个候选、初值、结果及筛选原因。
- `random_validation.json`, `differential_checks.json`, `jacobians.npz`：全部随机尝试、30个差分检查及10组双臂J。
- `collision_policy.json`, `replay_validation.json`：全部检查/排除pair与理由、保存后的一致性。
- `results/m1/development/`：真实开发诊断，独立于正式结果。
- `results/m1/frozen_exp0_integrity.json`：原EXP0产物48项哈希均保持。

## 实际失败、局限与审阅关口

首次启动时实验和测试并发解压同一目录，导致读取未写完的二进制STL（header长度1866400、读取262060），导入失败；完整栈保留在 `development/first_run_asset_read_error.log`。这是资产读取失败，未产生有效实验样本。提交090092e使既有文件校验复用、不再重写，新文件完整写入后原子发布；添加损坏/重复导入测试。修复后正式运行使用新空目录。不是通过添加猜测参数或放宽约束修复。

初版单凸包检查在双侧腕link5/link7产生误报；`development/hull_false_positives.json` 保存实际凸包碰撞而三角表面不相交、约6 mm间隙的结果。最终使用三角复核且保留2 mm条件；没有为好看而关闭该pair。最终随机筛选的13次碰撞拒绝和IK的6次超差拒绝均完整保存。

仅完成文件审计和URDF内静态几何验证。第7轴语义冲突意味着**不能把本次IK/J结果直接作为经确认的H1真机结果**。没有匹配当前施力余量、没有动态响应或恢复能力比较。G0/G1维持用户已限定的M0审阅结论，G2（真实H1配置恢复差异）与G3（创新/强基线优势）仍未验证。

下一步最小任务是审阅本报告及资产冲突，优先向厂家核实第7轴、模型版本、惯性、舵轮与关节参数；本轮在这里停止，等待用户/研究助手审阅，不自行进入动态施力建模。待确认信息清单见资产审计末节，无需4×4090服务器。
'''
Path('docs/M1_KINEMATICS_RESULTS.md').write_text(kin)
print('Wrote docs/M1_ASSET_AUDIT.md and docs/M1_KINEMATICS_RESULTS.md')
