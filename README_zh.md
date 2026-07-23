# 网球比赛视频视觉分析

> 一个用于自动分析网球比赛视频的计算机视觉流水线——直接从原始比赛录像中检测球场、追踪球员姿态并识别球员动作（静止 / 正手 / 反手 / 发球 / 移动）。

[English](./README.md) | 简体中文

---

## 概述

本项目是一个面向网球单打比赛的端到端视觉分析流水线，分为三个连续阶段：

```
原始比赛视频
   │
   ├─[1] 球场检测       YOLO 关键点 → 14 个球场点 → 单应性变换（俯视坐标）
   │
   ├─[2] 球员姿态追踪    YOLO-pose 追踪近/远端球员 → 17 点骨架 → EMA 平滑与缺口填补
   │
   └─[3] 动作识别        自定义 MSTFormer → 5 类动作 + 关键帧检测（双头）
```

第 3 阶段 **MSTFormer**（多流 Transformer）是本项目的核心贡献：它融合了三路信息流——球员**姿态序列**、**球场几何**与**多流视觉裁剪图**——并使用 Transformer 联合完成**动作分类**与**关键帧检测**。

## 模块

| 模块 | 路径 | 说明 |
| --- | --- | --- |
| **球场检测** | `src/court_detector.py`, `src/pipeline/` | YOLO 14 点关键点模型；检测球场点并计算单应性变换 |
| **姿态追踪** | `src/pose_tracker.py` | 使用 YOLO-pose 追踪近/远端球员，配合 EMA 平滑、缺口填补，以及多项打分机制（置信度、追踪惯性、球场邻近度、局部运动量）以剔除裁判、捡球童等非球员对象 |
| **球追踪** | `src/ball_tracker.py`, `src/ball_tracker_tracknet.py`, `src/tracknet/` | 两种可互换的后端：经典 CV 追踪器（背景减除 + 卡尔曼滤波，无需额外配置）以及基于 [TrackNet](https://github.com/yastrebksv/TrackNet) 的追踪器（精度更高，需下载模型文件/权重——见[使用方法](#使用方法)） |
| **音视频-球融合** | `src/audio_video_fusion.py` | 带通滤波的音频冲击（onset）检测 + WAITING/POINT_ACTIVE 滞回状态机；供 `main.py` 的 `"fusion"` 分段模式使用，将音频、球员运动与球活动信号融合以确定回合边界 |
| **动作识别（核心）** | `src/model/mst/` | MSTFormer：双头结构（5 类动作 + 关键帧）、三路视觉 token 融合、姿态/裁剪图消融开关 |
| **人物分类** | `src/model/yolo/`, `src/training/` | 用于区分近端/远端球员的 YOLO 分类器 |
| **批处理流水线** | `src/main.py` | 将比赛切分为回合（三种可互换模式：固定机位、广播 CLIP 场景分类，或音频/运动/球融合），通过 FFmpeg 切割并合成回合片段，并可选地为每个片段叠加姿态骨架与球轨迹进行重新渲染 |
| **离线精确追踪** | `src/pipeline/offline_tennis_tracker.py` | 独立的追踪流水线（球场单应性、球员边界框追踪、雷达视图），支持断点续跑——与 `main.py` 相比是一个独立的、更重量级的工具 |
| **可视化演示** | `src/demo/` | PyQt5 桌面应用：视频播放 + 三行时间轴（真值/预测/帧）+ 实时推理叠加 |
| **标注与数据工具** | `src/utils/` | 动作时间轴标注工具（Flask 网页版）、球场关键点标注工具（GUI）、球员边界框标注工具、数据集划分等 |

> 各文件的具体职责详见 [`docs/architecture_zh.md`](./docs/architecture_zh.md)。

## 仓库结构

```
tennis-vision-analysis/
├── src/                    源代码
│   ├── main.py             批处理视频入口（分段 + 切割 + 标注）
│   ├── court_detector.py   球场 ROI 检测器（供 pose_tracker 用于近/远端球员区域划分）
│   ├── pose_tracker.py     姿态追踪器
│   ├── ball_tracker.py     经典 CV 球追踪器（背景减除 + 卡尔曼滤波）
│   ├── ball_tracker_tracknet.py  基于 TrackNet 的球追踪器封装（精度更高）
│   ├── audio_video_fusion.py     音频冲击检测 + 滞回状态机（融合模式）
│   ├── tracknet/           TrackNet 模型文件（需单独下载，见"使用方法"）+ 预训练权重
│   ├── train_court_pipeline.py  球场模型训练入口
│   ├── pipeline/           离线追踪、数据集准备、标注精修工具
│   ├── model/
│   │   ├── mst/            MSTFormer 模型、训练与评估
│   │   └── yolo/           人物分类模型
│   ├── demo/               PyQt5 可视化演示
│   ├── utils/              标注与数据处理脚本
│   └── training/           人物检测/分类训练脚本
├── configs/                YAML 配置文件（球场、人物、MSTFormer 主配置/消融/超参/组件）
├── docs/
│   ├── architecture_zh.md  详细的分文件说明及模块依赖关系
│   └── figures/             实验结果图表
├── requirements.txt
├── LICENSE
└── README.md / README_zh.md
```

## 安装

```bash
# 推荐 Python 3.10+（开发环境为 3.11 / 3.12）
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

> **PyTorch / CUDA**：请参照[官方指南](https://pytorch.org/get-started/locally/)安装与你的 CUDA 版本匹配的 `torch` / `torchvision`，再安装其余依赖。

## 使用方法

> ⚠️ 本仓库**仅包含源代码、配置文件与文档**。原始比赛视频、标注数据集与模型权重（总计数百 GB）未纳入版本控制，请自行准备并分别放置于 `videos/`、`data/`、`models/` 目录下。路径约定详见 `src/config_legacy.py` 与 `configs/`。

```bash
# 1) 批处理视频：切分为回合、通过 FFmpeg 切割合成，并（默认）为每个回合片段
#    叠加姿态骨架与球轨迹进行重新渲染。
#    可在 src/main.py 底部编辑 mode/annotate/fusion_weights 等参数：
#      mode="static"    —— 固定机位画面，通过球员边界框速度分段
#      mode="broadcast" —— 电视转播画面，通过 PySceneDetect + CLIP 场景分类分段
#      mode="fusion"    —— 通过音频冲击检测、球员运动与球活动的加权组合，
#                          经滞回状态机分段
python src/main.py

# 2) 训练球场关键点检测模型
python src/train_court_pipeline.py

# 3) 训练 MSTFormer 动作识别模型（选择一个配置文件）
python src/model/mst/train.py --config configs/main.yaml

# 4) 启动动作时间轴标注工具（打开 http://localhost:5000）
python src/utils/action_annotator.py

# 5) 启动可视化演示
python src/demo/main.py --rally <rally_dir>
```

### 可选：TrackNet 球追踪后端

`src/ball_tracker.py`（经典 CV，背景减除 + 卡尔曼滤波）开箱即用。
如需更高精度的球追踪，`src/main.py` 可改用 [TrackNet](https://github.com/yastrebksv/TrackNet)
（在 `src/main.py` 顶部切换 `BALL_TRACKER_BACKEND`）——由于预训练权重未随本仓库打包，
需要进行一次性手动配置：

1. 从 [TrackNet 仓库](https://github.com/yastrebksv/TrackNet)下载 `model.py` 与 `general.py`
   至 `src/tracknet/`，并添加一个空的 `src/tracknet/__init__.py`。
2. 下载该仓库 README 中链接的预训练权重，并保存为
   `src/models/tracknet/model_best.pt`。

如果这些文件不存在，`main.py` 会自动回退到经典 CV 追踪器，并在控制台给出提示，而不会报错中断。

## 数据格式

> **说明：** `src/main.py` 当前的输出为视频片段（`rally_XXX.mp4`、`all_rallies_combined.mp4`，以及在启用标注时对应的 `_annotated` 版本），以及在 `"fusion"` 模式下内部计算得到的音频/运动/球活动分数。下方的 `pose_data.json` / `annotations.json` 格式由离线流水线（`src/pipeline/`）与标注工具（`src/utils/`）生成/消费，用于模型训练，而非直接由 `main.py` 产生。

**动作标注 `annotations.json`** —— 每个时间片段一条记录：

```json
[
  {"start_time": 0.0,   "end_time": 4.837, "action_name": "idle",  "action_id": 0},
  {"start_time": 4.837, "end_time": 12.78, "action_name": "serve", "action_id": 3}
]
```

动作类别：`idle 静止(0)`、`forehand 正手(1)`、`backhand 反手(2)`、`serve 发球(3)`、`movement 移动(4)`。

**姿态数据 `pose_data.json`** —— 每帧一条记录：

```json
{
  "frame": 0,
  "court": [[x, y, conf], ...],
  "near_player": {"bbox": [x1,y1,x2,y2], "keypoints": [[x,y,conf], ...]},
  "far_player":  {"bbox": [x1,y1,x2,y2], "keypoints": [[x,y,conf], ...]}
}
```

- `court`：14 个球场关键点；置信度低于 0.3 时在特征向量中置零
- `near_player` / `far_player`：17 点 COCO 骨架

## 实验结果

| 训练曲线 | 主混淆矩阵 |
| --- | --- |
| ![训练曲线](./docs/figures/fig1_main_training_curve.png) | ![混淆矩阵](./docs/figures/fig7_confusion_matrix_main.png) |

更多图表（消融实验、超参数扫描、组件对比、关键帧检测曲线）见 [`docs/figures/`](./docs/figures/)。

## 技术栈

- **深度学习**：PyTorch、Ultralytics YOLO11（检测 / 姿态 / 关键点）、[TrackNet](https://github.com/yastrebksv/TrackNet)（可选球追踪后端）
- **自定义模型**：MSTFormer（融合姿态 + 球场几何 + 视觉裁剪图的多流 Transformer）
- **CV / 数值计算**：OpenCV、NumPy、SciPy（同时用于音频冲击检测中的带通滤波）、scikit-learn
- **音视频**：FFmpeg（音频提取、片段切割/拼接）
- **应用 / 标注**：PyQt5（桌面演示）、Flask（网页版标注工具）

## 许可证

基于 [MIT 许可证](./LICENSE) 发布，Copyright © 2026 Da_233。

---

> 本项目源于一篇本科毕业设计。开源发布内容仅包含代码与文档；论文正文及受版权保护的参考文献未包含在内。