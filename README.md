# CR3 × CRAFT Teleoperation Showcase

CR3 六轴机械臂与 CRAFT 灵巧手在 DexJoCo/MuJoCo 中的集成、任务环境和遥操作验证仓库。

> 本仓库是从完整实验工作区整理出的 **可读展示版 / 交接版**。它保存了 CR3+CRAFT 的环境适配、MuJoCo 场景、遥操作脚本、测试和实验文档；完整的 DexJoCo 主工程、第三方模型权重、CAD 文件和本地运行环境不包含在这里。

[![GitHub](https://img.shields.io/badge/GitHub-fslee2%2Fcr3--craft--teleop--showcase-181717?logo=github)](https://github.com/fslee2/cr3-craft-teleop-showcase)
[![Simulation](https://img.shields.io/badge/Simulation-MuJoCo%20%2F%20DexJoCo-1f6feb)](#项目定位)
[![Input](https://img.shields.io/badge/Teleoperation-MediaPipe%20%2B%20Dual--Camera-2ea44f)](#遥操作链路)

![CR3 + CRAFT camera grid](assets/images/cr3_craft_camera_grid.png)

![CR3 + CRAFT x-axis teleoperation](assets/gifs/cr3_craft_x_axis_teleop.gif)

## 项目定位

这个项目解决的是一个具体的系统集成问题：把原本面向 Panda/Allegro 类机器人的 DexJoCo 任务，替换成 CR3 机械臂和 CRAFT 灵巧手，并让不同的人类输入方式都能够转换成同一套机器人动作接口。

它不是单独的机械臂模型，也不是完整的真实硬件控制程序，而是连接以下几层的仿真与遥操作验证平台：

```text
MediaPipe / 键盘 / 双摄像头 / Quest WebXR
                    ↓
       末端位姿 + CRAFT 手指动作
                    ↓
             统一 22D action
                    ↓
       CR3 + CRAFT DexJoCo 环境
                    ↓
       MuJoCo 物理仿真与任务反馈
```

## 核心接口

### 22 维动作接口

所有环境共享同一套动作布局：

```text
action[0:3]   = target_xyz        # 机械臂末端目标位置
action[3:7]   = target_quat_wxyz  # 末端目标姿态，四元数顺序为 wxyz
action[7:22]  = craft15           # CRAFT 五指的 15 个直接控制量
```

`craft15` 并不是 20 个手部自由度的全部关节，而是每根手指的 3 个主要输入；远端 DIP 关节通过 MuJoCo equality 约束与 PIP 关节耦合。因此，外部遥操作器只需要输出 15 个手指控制量，环境内部负责展开到 CRAFT 的完整关节/执行器。

### 观测接口

环境提供 DexJoCo 风格的字典观测，核心内容包括：

```text
observation["state"]["tcp_pose"]         # 当前末端位置 + wxyz 姿态
observation["state"]["arm_qpos"]         # CR3 六个关节角
observation["state"]["craft_qpos"]       # CRAFT 手部关节状态
observation["state"]["target_tcp_pose"]  # 当前目标末端位姿
observation["images"]["front"]           # 前视角图像
```

## 三个仿真环境

| 环境 | 用途 | 特点 |
| --- | --- | --- |
| `Cr3CraftReachDebugEnv` | 最小化调试 | 用于验证 CR3 末端 IK、位置伺服和 CRAFT 手指响应 |
| `Cr3CraftClickMouseEnv` | 独立点击任务 | 自带桌面、鼠标、鼠标垫和显示器，结构简单，适合快速调试 |
| `Cr3CraftClickMouseShellEnv` | 主线任务环境 | 保留 DexJoCo/Panda arena 的桌面、纹理、显示器和任务逻辑，只替换机器人为 CR3+CRAFT |

推荐后续实验优先使用 `Cr3CraftClickMouseShellEnv`。它最接近完整任务场景，并且同时包含：鼠标位置随机化、鼠标垫约束、点击检测、显示器反馈和成功计数。

## 遥操作链路

### 1. MediaPipe 单摄像头

当前最轻量、最容易在 Windows 上运行的方案：

- 手掌左右/上下移动映射到 TCP 的 Y/Z；
- 手指弯曲映射到 CRAFT 15 维手指控制；
- 键盘补偿单目摄像头不容易获得的 TCP X 深度；
- 支持重新校准、滤波、死区和最大步长限制。

这是一条“可运行的实用原型”，不是严格的三维手部重建。单摄像头对前后深度的估计是它的主要限制。

### 2. 双摄像头 MediaPipe

双摄像头脚本把深度问题拆开：

```text
正面摄像头 → TCP Y/Z + CRAFT 手指
侧面摄像头 → TCP X
```

它不是经过标定的双目视觉重建，而是两个视角的工程化 2D 映射，适合快速验证遥操作可行性。

### 3. 键盘控制

键盘脚本用于不接摄像头时调试环境和任务逻辑：

```text
Z / Numpad 7  → TCP X-
X / Numpad 9  → TCP X+
H / Numpad 5  → 清除 X 偏移
R             → 重新校准
Q / Esc       → 退出
```

### 4. Quest / WebXR

`src/tasks/quest_teleop.py` 是 Quest 3 WebXR 的 MVP 桥接层，负责接收 Quest 位姿、处理四元数转换、计算相对运动、平滑位置、映射抓握量，并通过 WebSocket 与仿真循环交换状态。这部分目前应视为接口原型，不能描述成已经完成的 Quest 产品级控制系统。

## 仓库结构

```text
.
├── assets/                         # README 展示用图片和 GIF
├── docs/                           # 集成审计、运行交接和硬件输入说明
├── src/
│   ├── envs/                       # CR3+CRAFT Gym/DexJoCo 环境类
│   ├── scripts/                    # 键盘、MediaPipe、双摄像头、Quest 入口
│   ├── tasks/                      # Quest 状态接收和动作映射
│   ├── tests/                      # 环境与 Quest 映射的轻量测试
│   ├── xmls/                       # CR3+CRAFT MuJoCo 场景定义
│   └── requirements-windows-mediapipe.txt
├── tools/                          # 渲染展示媒体的辅助脚本
├── PROJECT_SCOPE.md                # 发布边界和未包含内容
└── README.md                       # 项目总览和运行入口
```

## 运行前提

这个展示仓库不是完全独立的 Python 包。环境类中的相对导入和 MuJoCo XML include 仍依赖完整 DexJoCo 主工程中的基础类、控制器、机器人网格和任务资源。因此运行前需要：

1. 准备完整的 DexJoCo/CRAFT 主工程；
2. 将本仓库中的对应 `src/` 内容合并或映射到主工程的 `dexjoco/` 包中；
3. 确保 `cr3_craft/models/`、基础 `mujoco_gym_env.py`、`controllers/opspace.py` 以及任务 XML 资源可被找到；
4. Windows MediaPipe 方案额外准备 `hand_landmarker.task`。

下面命令中的 `<DEXJOCO_ROOT>` 代表完整 DexJoCo 主工程目录，不是本展示仓库的目录。

## 快速验证

### 环境冒烟测试

```powershell
cd <DEXJOCO_ROOT>
python scripts/smoke_cr3_craft_envs.py --env all --steps 2
python scripts/smoke_cr3_craft_envs.py --env click_mouse_shell --steps 5 --render-check
```

### Windows MediaPipe 单摄像头

```powershell
cd <DEXJOCO_ROOT>
.\scripts\setup_windows_uv_mediapipe.ps1
.\scripts\run_mediapipe_cr3_windows_uv.ps1 -CameraId 0
```

等价的 Python 入口：

```powershell
.\.venv-mediapipe-win\Scripts\python.exe scripts\mediapipe_cr3_craft_click_mouse_shell.py `
  --camera-id 0 `
  --viewer `
  --preview `
  --keyboard-x-step 0.01
```

### 双摄像头原型

```powershell
.\scripts\run_dual_mediapipe_cr3_windows_uv.ps1 `
  -FrontCameraId 0 `
  -SideCameraId 1
```

如果侧面摄像头的 X 方向相反：

```powershell
.\scripts\run_dual_mediapipe_cr3_windows_uv.ps1 -SideXSign -1
```

## 完成度与边界

### 已完成或基本完成

- CR3 六轴机械臂与 CRAFT 手部 MuJoCo 资产的组合场景；
- 三个 CR3+CRAFT 环境类；
- 统一 22D action 和标准 `reset / step / render / close` 接口；
- CRAFT 远端关节耦合和执行器控制；
- click-mouse 任务逻辑及 Shell 场景适配；
- MediaPipe、键盘和双摄像头遥操作入口；
- Quest/WebXR 的状态和动作映射原型；
- 冒烟测试、交接文档和展示媒体。

### 仍属于实验阶段

- MediaPipe 手到机器人坐标的标定与自然度；
- 单目深度和双摄像头 X 方向映射；
- Quest 真实浏览器连接、TLS 和网络部署；
- HaMeR/MobileHand 等重型视觉模型的完整接入；
- 真实 CR3 + CRAFT 硬件闭环和 Sim2Real 安全验证；
- 面向策略训练的数据采集、评估和大规模复现实验。

## 不包含的内容

- 第三方 DexJoCo、HaMeR、MobileHand 完整仓库；
- 模型权重、缓存和本地虚拟环境；
- SolidWorks/CAD 与 3D 打印源文件；
- Quest Developer Hub 等安装程序；
- 论文原始数据、训练日志和大体积实验产物。

## 文档导航

- [架构与代码导览](docs/architecture_guide.md)
- [DexJoCo 集成状态](docs/cr3_craft_dexjoco_integration_status.md)
- [集成审计报告](docs/cr3_craft_dexjoco_integration_audit.md)
- [遥操作交接记录](docs/cr3_craft_teleop_handoff.md)
- [Windows + uv + MediaPipe](docs/windows_uv_mediapipe_env.md)
- [Quest 3 WebXR MVP](docs/quest3_teleop_mvp.md)
- [发布边界](PROJECT_SCOPE.md)

## 一句话总结

这是一个把 **CR3 机械臂、CRAFT 灵巧手、MuJoCo 任务环境和多种人类输入方式** 接起来的研究型仿真与遥操作平台；环境接口已经建立，遥操作质量和真实硬件闭环是下一阶段重点。
