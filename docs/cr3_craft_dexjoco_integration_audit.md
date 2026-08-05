# CR3 + CRAFT DexJoCo 集成审计报告

**日期**: 2026-07-09
**审计方式**: 自动化（代码审查 + 运行时验证）
**范围**: 判断 CR3 + CRAFT hand 是否真正融入 DexJoCo 环境体系，而非独立 demo。

---

## 1. 总体摘要

CR3 + CRAFT hand 已成功作为一等环境接入 DexJoCo。三个 env 类均继承自 `MujocoGymEnv`（后者继承 `gym.Env`），暴露标准的 `reset()`/`step(action)`/`render()`/`close()` 接口，使用统一的 22 维 action 布局（`target_xyz[3] + target_quat_wxyz[4] + craft15[15]`），返回 DexJoCo 风格的字典观测（`state` + `images`）。MuJoCo XML 场景包含真实的 CR3 机械臂运动学（关节 1-6）、CRAFT 手部网格和关节（20 DOF，5 指，远端耦合），以及两者的完整执行器。Panda 机器人已被完全替换——所有 CR3+CRAFT XML 中没有任何 Panda 几何体或执行器。三个 env 中有两个（`Cr3CraftClickMouseEnv`、`Cr3CraftClickMouseShellEnv`）具备完整的 click_mouse 任务逻辑（鼠标、鼠标垫、显示器、点击检测、成功计数、随机化）。Shell env 最为接近生产状态：它复用了 Panda arena 场景（墙壁、地板、桌面纹理、显示器），仅将机器人替换为 CR3+CRAFT。三个 env 均通过 reset/step 冒烟测试。

当前主要短板在于**包装和产品化**，而非环境集成：缺少 Gym registry、遥操作控制质量尚处于实验阶段、Shell env 此前未从 `__init__.py` 导出（本次审计已修复）、缺少标准的冒烟测试。

**一句话结论**: CR3+CRAFT 环境已准备就绪，可通过 22D action 接口接入 policy/teleop。剩余工作在于控制调参、遥操作质量和包装打磨——不需要重建环境层。

---

## 2. 完成度矩阵

| 组件                                | 状态                     | 备注                                                                                     |
| ----------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------- |
| CR3 机械臂资产（网格、运动学）      | **完成**           | 7 个网格（base_link, J1-J6），6 个关节含惯性数据                                         |
| CRAFT 手部资产（网格、关节）        | **完成**           | 20 个 STL 网格（5 指 × 4 段），20 个关节含正确轴向量                                    |
| CR3+CRAFT 组合本体 XML              | **完成**           | `cr3_robot_hand_body.xml`——CR3 链 + CRAFT 手指从 Link6 分支                          |
| MuJoCo XML: reach_debug 场景        | **完成**           | 简单桌面 + 机器人，无任务物体                                                            |
| MuJoCo XML: click_mouse 场景        | **完成**           | 内联鼠标/显示器/鼠标垫 + 机器人                                                          |
| MuJoCo XML: click_mouse_shell 场景  | **完成**           | Panda arena（地板/墙壁/桌面）+ CR3+CRAFT 机器人替换                                      |
| CRAFT 远端耦合等式约束              | **完成**           | 全部 5 指：PIP → DIP 关节耦合，XML 和 Python 层均有                                     |
| CRAFT 自碰撞排除                    | **完成**           | 逐指相邻连杆排除（20 对），XML 层                                                        |
| DexJoCo env 类（MujocoGymEnv 子类） | **完成**           | 全部 3 个 env 继承自`MujocoGymEnv`                                                     |
| `action_space`（22D Box）         | **完成**           | `[xyz(3), wxyz(4), craft15(15)]`——三个 env 完全一致                                  |
| `observation_space`（Dict）       | **完成**           | `state.{tcp_pose, arm_qpos, craft_qpos, target_tcp_pose}` + `images.front`           |
| `reset()` / `step(action)`      | **完成**           | step 返回标准 5 元组，info 字典已填充                                                    |
| `render()` / `close()`          | **完成**           | 来自基类`MujocoGymEnv`，运行正常（冒烟测试中已验证渲染）                               |
| `get_initial_action()`            | **完成**           | 全部 3 个 env，返回 (22,) np 数组                                                        |
| `get_end_effector_pose_matrix()`  | **完成**           | ReachDebug 和 Shell 中有，返回 4×4 齐次矩阵                                             |
| click_mouse 任务逻辑                | **完成** (2/3 env) | 鼠标/垫/显示器物体、点击检测、成功计数、随机化——ClickMouse 和 Shell env 具备           |
| 桌面高度随机化                      | **完成**           | ClickMouse 和 Shell 均随机化桌面高度                                                     |
| 鼠标/鼠标垫/显示器随机化            | **完成**           | Shell 做完整随机化（位置、偏航角、显示器位置）；ClickMouse 做鼠标位置+偏航角             |
| 桌面纹理随机化                      | **完成** (Shell)   | Shell 有 18 种桌面材质变体（来自 Panda arena）                                           |
| XML 中无 Panda 依赖                 | **完成**           | 所有 CR3+CRAFT XML 中零 Panda 机器人几何体或执行器                                       |
| 包导出（`__init__.py`）           | **完成**（已修复） | Shell env 此前缺失，本次审计已补上                                                       |
| Gym registry（`gym.make`）        | **缺失**           | 项目中无任何`gym.register`；所有 env 均为直接 import                                   |
| 遥操作包装器（HaMeR）               | **实验态**         | `hamer_cr3_craft_click_mouse_shell.py` 脚本存在，连接 HaMeR → env，但控制质量未经验证 |
| 遥操作包装器（MediaPipe）           | **实验态**         | `mediapipe_cr3_craft_click_mouse_shell.py` 存在；同实验态                              |
| 手动控制脚本                        | **实验态**         | 多个`manual_cr3_craft_*.py` 脚本用于调试                                               |
| 冒烟测试                            | **缺失**           | 无`pytest`/`unittest` 测试文件覆盖 CR3+CRAFT env                                     |
| 文档                                | **部分完成**       | `docs/cr3_craft_backend.md` 存在但简短；无正式 env API 文档                            |

---

## 3. 环境清单

### Cr3CraftReachDebugEnv

- **用途**: 最小化调试/开发环境，用于独立测试 CR3 机械臂 IK + CRAFT 手部控制。
- **场景**: 简单桌面 + 机器人。无鼠标、显示器或鼠标垫。有绿色目标标记（`target_marker` 物体）用于可视化调试。
- **控制器**: 机械臂使用 IK（TCP 雅可比上的阻尼最小二乘法）；手部使用位置伺服。
- **机械臂执行器**: 位置执行器（位置伺服）。
- **任务逻辑**: 无。奖励 = 负 TCP 位置误差。除时间限制外无成功/终止条件。
- **状态**: 稳定的调试工具，非任务 env。

### Cr3CraftClickMouseEnv

- **用途**: 第一代 CR3+CRAFT click_mouse 任务。独立场景（不复用 Panda arena）。
- **场景**: 自定义桌面、带点击关节的鼠标、显示器、鼠标垫圆柱体。所有物体在 XML 中内联定义。
- **控制器**: 与 ReachDebug 相同的 IK + 位置伺服。
- **机械臂执行器**: 位置执行器。
- **任务逻辑**: 完整 click_mouse——mouse_in_mousepad 检测、点击检测（滑动关节传感器）、点击后显示器变蓝、持续触发后判定成功（连续 10 步）。reset 时随机化鼠标 x/y/偏航角。随机化桌面高度。奖励 = -TCP 位置误差。
- **状态**: 功能正常的任务 env。视觉丰富度低于 Shell。控制器比 Shell 简单。

### Cr3CraftClickMouseShellEnv（推荐）

- **用途**: 生产级 CR3+CRAFT click_mouse 环境。复用 Panda arena 场景，仅替换机器人为 CR3+CRAFT。
- **场景**: 完整 Panda arena（墙壁、带纹理的地板、含 18 种随机化纹理变体的桌面、鼠标、显示器、鼠标垫）。机器人安装在 Panda 基座位置 `(-0.8, 0, 0.9)` 的底座上。
- **控制器**: 机械臂使用操作空间控制器（`opspace`，来自 `dm_robotics`）进行力矩控制；手部使用位置伺服。
- **机械臂执行器**: 电机执行器（力矩控制）——物理真实度更高。
- **任务逻辑**: 与 ClickMouseEnv 相同的 click_mouse 逻辑，外加显示器位置随机化（`_PLANT_XY_BOUNDS`）。桌面纹理随机化（18 种材质）。
- **状态**: 最完整、最接近生产环境的 env。推荐作为后续主线环境。

---

## 4. 主线环境推荐

**推荐将 `Cr3CraftClickMouseShellEnv` 作为主线环境。**

理由：

1. **最接近 Panda 生产环境**: 复用了 Panda arena 场景（`arena_arm_hand_monitor_mouse.xml`），可直接与基于 Panda 的任务对比。
2. **力矩控制机械臂**: 使用电机执行器 + 操作空间控制器，与 Panda env 的物理真实度一致。
3. **更丰富的随机化**: 桌面高度、鼠标位置/偏航角、鼠标垫位置、显示器位置和桌面纹理全部随机化——与 Panda click_mouse 一致。
4. **更好的视觉质量**: Panda arena 墙壁、地板纹理、合适的灯光。
5. **已预留底座**: 在 Panda 基座坐标处有 `cr3_pedestal_mount` 物体，便于调整安装高度。
6. **遥操作脚本已对接**: `hamer_cr3_craft_click_mouse_shell.py` 和 `mediapipe_cr3_craft_click_mouse_shell.py` 连接的是这个 env。

`Cr3CraftReachDebugEnv` 应保留作为快速调试/测试环境。`Cr3CraftClickMouseEnv` 可逐步废弃或保留为简化替代方案。

---

## 5. 接口就绪状态

### Action 接口——就绪

Policy/teleop 可通过 **22 维 action 向量**接入：

```
action[0:3]   → target_xyz（机械臂末端目标位置，世界坐标系）
action[3:7]   → target_quat_wxyz（机械臂末端目标姿态）
action[7:22]  → craft15（CRAFT 手指关节指令，0 到 2π 映射）
```

craft15 的排列顺序为：无名指(PIP, MCP前向, MCP侧向)、食指(PIP, MCP前向, MCP侧向)、拇指(PIP, MCP前向, MCP侧向)、中指(PIP, MCP前向, MCP侧向)、小指(PIP, MCP前向, MCP侧向)。

取值范围 `[0, 2π]`，线性映射到各关节的运动范围。远端关节（DIP）通过等式约束从动于 PIP——它们不在 15D 指令中。

### Observation 接口——就绪

```python
obs = {
    "state": {
        "tcp_pose": np.ndarray(7,),       # TCP site 的 xyz + wxyz
        "arm_qpos": np.ndarray(6,),       # CR3 关节位置 (rad)
        "craft_qpos": np.ndarray(20,),    # CRAFT 关节位置 (rad)
        "target_tcp_pose": np.ndarray(7,),# 上次指令目标
    },
    "images": {
        "front": np.ndarray(H, W, 3),     # uint8 RGB
    },
}
```

### 辅助方法——已有

| 方法                                        | ReachDebug | ClickMouse | Shell |
| ------------------------------------------- | ---------- | ---------- | ----- |
| `get_initial_action()` → (22,)           | ✓         | ✓         | ✓    |
| `get_end_effector_pose_matrix()` → (4,4) | ✓         | ✗         | ✓    |

### 各 Env 控制方式差异

| 方面             | ReachDebug & ClickMouse          | Shell            |
| ---------------- | -------------------------------- | ---------------- |
| 机械臂控制       | IK + 位置伺服（关节位置目标）    | 操作空间力矩控制 |
| 手部控制         | 位置伺服                         | 位置伺服（相同） |
| 机械臂执行器类型 | `<position>`                   | `<motor>`      |
| 控制速率         | `_arm_ctrl_step = 0.02` rad/步 | 每子步计算力矩   |

---

## 6. 尚未完成的部分

以下为影响产品化的高层缺口，不影响核心集成：

1. **缺少 Gym registry**——CR3+CRAFT env 无法通过 `gym.make("Cr3CraftClickMouseShellEnv-v0")` 创建。所有使用均为直接 Python import。这与 Panda env 的情况相同（项目中完全没有 registry），属于项目级缺口，非 CR3 特有。
2. **缺少冒烟测试**——没有 `tests/test_cr3_craft_*.py` 验证 reset/step 确定性、action 边界、observation 形状或 click_mouse 成功逻辑。
3. **Shell env 此前未导出**——本次审计已修复（`__init__.py` 现已包含 `Cr3CraftClickMouseShellEnv`）。
4. **遥操作控制质量为实验态**——HaMeR 和 MediaPipe 脚本已存在并连接到 env，但没有系统性地评估跟踪精度或任务成功率。IK 和操作空间控制器功能正确，但增益的手动调参仍在进行中。
5. **CR3 安装位置可能需要调整**——Shell env 将机器人安装在 `(-0.8, 0.0, 0.9)`（Panda 基座位置）。相对于桌面（`(-0.15, 0, 0)`），其工作空间覆盖可能需要调整。
6. **ClickMouseEnv 缺少 `get_end_effector_pose_matrix()`**——ReachDebug 和 Shell 有，ClickMouseEnv 没有。轻微的 API 不一致。
7. **缺少正式文档**——`docs/cr3_craft_backend.md` 存在但简短。无 API 参考、无快速入门、无架构图。
8. **Click mouse 随机化对齐度**——Shell env 的随机化已接近 Panda click_mouse，但需要与 `panda_click_mouse_env.py` 做 diff 才能确认 100% 对齐。

---

## 7. 运行验证证据

**环境**: WSL, Python 3.x, MuJoCo EGL 渲染

**命令**:

```bash
cd <DEXJOCO_ROOT>
source ~/.venv-hamer/bin/activate
export PYTHONPATH=<DEXJOCO_ROOT>/dexjoco
export MUJOCO_GL=egl
python -c '...'  # 完整脚本见审计说明
```

### Cr3CraftReachDebugEnv

```
action_space shape: (22,)
initial_action shape: (22,)
obs keys: ['state', 'images']
state keys: ['tcp_pose', 'arm_qpos', 'craft_qpos', 'target_tcp_pose']
image keys: ['front']
image front shape: (640, 640, 3)
info keys (reset): []
step ok, reward: -0.0
terminated: False, truncated: False
info2 keys: ['tcp_error', 'target_tcp_pose', 'ncon', 'intervene_action']
close ok
```

### Cr3CraftClickMouseEnv

```
action_space shape: (22,)
initial_action shape: (22,)
obs keys: ['state', 'images']
state keys: ['tcp_pose', 'arm_qpos', 'craft_qpos', 'target_tcp_pose']
image keys: ['front']
image front shape: (640, 640, 3)
info keys (reset): ['succeed']
step ok, reward: -0.003856...
terminated: False, truncated: False
info2 keys: ['tcp_error', 'ncon', 'succeed', 'mouse_in_pad', 'click_detected', 'display_blue']
close ok
```

### Cr3CraftClickMouseShellEnv

```
action_space shape: (22,)
initial_action shape: (22,)
obs keys: ['state', 'images']
state keys: ['tcp_pose', 'arm_qpos', 'craft_qpos', 'target_tcp_pose']
image keys: ['front']
image front shape: (640, 640, 3)
info keys (reset): ['succeed']
step ok, reward: -0.0
terminated: False, truncated: False
info2 keys: ['tcp_error', 'ncon', 'succeed', 'mouse_in_pad', 'click_detected', 'display_blue']
close ok
```

### 包导入验证

```
from dexjoco.sim.envs import Cr3CraftClickMouseEnv, Cr3CraftClickMouseShellEnv, Cr3CraftReachDebugEnv
All three importable from dexjoco.sim.envs: OK
```

**结果**: 全部 3 个 env 通过 reset/step/render/close，形状正确，info 字典已填充。无错误。

---

## 8. 最终判定

**CR3 + CRAFT 已经作为 DexJoCo 环境完成基础接入。**

### 完成度估算

| 层级       | 完成度        | 备注                                                                                                              |
| ---------- | ------------- | ----------------------------------------------------------------------------------------------------------------- |
| 资产 / XML | **95%** | 网格、运动学、执行器、等式约束、碰撞排除均已到位。安装位置可能需微调。                                            |
| Env 集成   | **90%** | 标准 Gym 接口、22D action、DexJoCo 观测字典、render/close。缺：ClickMouseEnv 的`get_end_effector_pose_matrix`。 |
| 任务集成   | **85%** | Click mouse 逻辑端到端可运行（检测、成功计数、随机化）。与 Panda env 可能存在轻微对齐差距。                       |
| 遥操作集成 | **70%** | 脚本将 HaMeR/MediaPipe 通过 22D action 接入 env。控制质量未验证。手到机器人重映射质量未知。                       |
| 产品化     | **40%** | 无 gym registry、无冒烟测试、文档极少、无 CI。这与 Panda env 相同的差距——属于项目级问题。                       |

### 对核心问题的回答

**"现在项目是不是已经完成 CR3 + CRAFT 场景和 DexJoCo 接口层，只是 teleop 还存在问题？"**

**是的，有限定条件。** 环境层——XML 场景、Gym env 类、action/observation 接口、click_mouse 任务逻辑——是稳固的且已通过运行时验证。CR3+CRAFT 环境是真正的 DexJoCo env，不是独立 demo。22D action 接口在三个 env 之间完全统一，任何输出 `(target_xyz, target_quat, craft15)` 的 policy 或 teleop 系统都可以立即接入。

当前瓶颈在于**遥操作控制质量和手到机器人的映射**，而非环境是否存在或能否工作。遥操作脚本已存在，但产出的轨迹质量为实验级。这对于新的机器人+手部组合是正常的，属于调参和标定问题，不是集成问题。

---

## 本次审计所做的修改

| 文件                                     | 修改内容                                                         | 原因                           |
| ---------------------------------------- | ---------------------------------------------------------------- | ------------------------------ |
| `dexjoco/dexjoco/sim/envs/__init__.py` | 添加`Cr3CraftClickMouseShellEnv` 的 import 和 `__all__` 条目 | Shell env 已实现但未从包中导出 |

验证：修改后 `from dexjoco.sim.envs import Cr3CraftClickMouseShellEnv` 成功。
