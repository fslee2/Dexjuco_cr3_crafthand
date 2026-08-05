# CR3 + CRAFT 仓库架构导览

这份文档面向第一次接手仓库的人，解释每一层负责什么、数据如何流动，以及哪些内容已经稳定、哪些内容仍在实验。

## 1. 总体分层

```text
输入层
├── keyboard_*.py
├── mediapipe_*.py
├── dual_mediapipe_*.py
└── quest_*.py
        ↓
动作映射层
├── 位置 / 姿态滤波
├── 坐标系与四元数转换
├── CRAFT 手指 retargeting
└── 22D action 拼接
        ↓
环境层
├── Cr3CraftReachDebugEnv
├── Cr3CraftClickMouseEnv
└── Cr3CraftClickMouseShellEnv
        ↓
仿真层
├── MuJoCo XML
├── CR3 六轴关节和执行器
├── CRAFT 20 个手部关节
└── equality 耦合与任务物体
```

## 2. 环境层的区别

### `Cr3CraftReachDebugEnv`

最小调试环境。它不强调任务场景，而是用来快速确认 CR3 关节和 TCP site、阻尼最小二乘 IK、CRAFT 手指响应，以及动作/观测空间形状。

### `Cr3CraftClickMouseEnv`

独立实现的点击鼠标任务。场景资源较少，便于理解和调试任务逻辑，包括鼠标、鼠标垫、显示器、随机化和成功判定。

### `Cr3CraftClickMouseShellEnv`

当前主线环境。它保留 DexJoCo/Panda arena 的场景外壳和视觉布局，将机器人替换为 CR3+CRAFT，并使用 `opspace` 操作空间控制器和位置伺服手部控制。它适合后续遥操作和策略实验。

## 3. 22D action 的内部处理

```text
target_xyz
    ↓
目标 TCP 位置 → workspace 限制 → IK / opspace → CR3 执行器

target_quat_wxyz
    ↓
目标 TCP 姿态 → 四元数归一化 → 姿态误差 → CR3 执行器

craft15
    ↓
15 个主控制关节 → 关节范围裁剪 → DIP equality 耦合 → CRAFT 执行器
```

这里的 15 个 CRAFT 输入按五根手指排列，每根手指使用 3 个主要控制量。MuJoCo XML 中额外的 distal 关节通过 equality 约束跟随对应的 PIP 关节，所以“15D 外部指令”和“20D 内部关节状态”并不矛盾。

## 4. 遥操作脚本如何连接环境

MediaPipe 脚本的主要流程是：

1. 打开摄像头并取得视频帧；
2. 运行 MediaPipe hand detector；
3. 用手掌中心和手部尺度计算相对位移；
4. 用手指关键点夹角计算手指弯曲量；
5. 将位移、保持的 TCP 姿态和 CRAFT 15D 拼成 action；
6. 调用环境 `step(action)`；
7. 通过 MuJoCo viewer 或 OpenCV preview 显示结果。

双摄像头版本复用同一套手指映射和滤波逻辑，只把 TCP X 的输入换成侧面摄像头的手掌水平位移。

## 5. MuJoCo XML 的依赖关系

当前仓库中的 XML 是展示版场景文件，但其中部分 `<include>` 仍指向完整 DexJoCo 工程中的机器人网格和任务资源，例如：

```text
cr3_craft/models/cr3_robot_hand_body.xml
mouse_nolight.xml
display.xml
```

因此 clone 本仓库后，不能把 `src/xmls/` 当成完全自包含的资产包。正确的使用方式是将这些文件放回完整 DexJoCo 的相应资源目录，或在本地工程中配置等价的 XML/mesh 搜索路径。

## 6. 当前最重要的技术结论

- 环境层比遥操作层成熟：CR3+CRAFT 已经有统一接口和三类任务环境；
- 最大的输入问题是单摄像头无法稳定提供前后深度；
- 双摄像头方案是工程化折中，不是严格的 stereo reconstruction；
- HaMeR/MobileHand 属于重型视觉方向，不能和当前轻量 MediaPipe demo 混为一谈；
- 真实硬件控制之前必须增加工作空间、速度、碰撞、电机限位和急停保护。

## 7. 建议的后续开发顺序

1. 先固定 `Cr3CraftClickMouseShellEnv` 的 XML 资源和环境 API；
2. 统一 MediaPipe、键盘和 Quest 的坐标系、增益、滤波和校准接口；
3. 建立录制 action/observation 的数据格式；
4. 在仿真中做可重复的遥操作评估；
5. 最后再连接真实 CR3 + CRAFT，并加入安全控制层。
