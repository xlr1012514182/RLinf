<div align="center">

# Fibocom π0.5 VLA Stack

**冻结基座适配 · 连续动作推理 · 机器人接口**

基于 RLinf

[English](README.md) · [实现指南](examples/embodiment/fibocom_vla/README_ZH.md) · [测试方法](docs/fibocom_vla/VALIDATION_ZH.md) · [来源与许可](docs/fibocom_vla/SOURCES.md)

</div>

## 项目简介

这套代码将冻结的 π0.5 策略与残差后训练、动作块推理及机器人控制接口连接起来。模型契约与部署契约相互分离：权重、归一化、相机顺序和动作维度保持明确，策略适配与硬件接入通过独立模块完成。

仓库分发**源码、运行资产清单、配置模板、测试和基准程序**。模型权重、任务数据、部署标定及运行产物单独提供。

## 整体架构

```text
同步相机 + 机器人状态 + 指令
             │
      观测 / 模型契约
             │
   ┌─────────┼──────────┐
   │         │          │
冻结 π0.5  π0.5 +     π0.5 + 已审批
 基础策略   有界残差    Draft / 验证器
   └─────────┼──────────┘
             │
      动作块 / 异步规划
             │
     标定适配器 / 安全检查
             │
         机器人后端
```

三个策略分支择一使用；残差与推测模式互斥。原生模型空间 RTC 仅用于保留对应动作血缘的路径，包装策略与非线性重定向使用明确标记的主机动作空间回退。

## 核心组件

| 模块 | 接口与适用范围 |
|---|---|
| **残差学习** | 有界 Actor、独立价值头和 Twin-Q 辅助 Critic、GAE 与单轮 PPO；初始化和更新 CLI 接收外部采集的 rollout 归档。 |
| **推测式推理** | 连续动作 Draft、单次 prefill 并行验证、连续前缀接受与同轮主模型接管；π0.5 部署需要兼容且经过训练、评测和签名审批的 Draft。 |
| **RTC** | 动作块异步生成、已承诺前缀冻结、重叠引导，以及分离的观测时间与生成时间检查。 |
| **CUDA Graph** | 对形状稳定的视觉塔和投影层捕获计算图，包含逐位一致性门控和 eager 回退；RTC、推测与梯度上下文绕过此路径。 |
| **TensorRT** | 引擎清单、构建/运行与数值一致性诊断；需自行提供导出模型或网络，不附整套 π0.5 导出器或引擎。 |
| **机器人接口** | SO101、Dobot Nova TCP、ROS2 关节后端，以及 OpenCV、RealSense、ROS2 相机；真机部署需任务专用映射与硬件标定。 |

## 快速开始：CPU / 无硬件

在 **Ubuntu 或 WSL2 x86_64** 的仓库根目录执行。需 Bash、Git、`python3`、`pip` 和网络：

```bash
bash requirements/fibocom_vla_quickstart.sh

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli validate-config \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli mock-smoke \
  --config examples/embodiment/fibocom_vla/config/mock.json --steps 16

.venv-fibocom/bin/python -m pytest -q tests/unit_tests/projects/fibocom_vla
```

安装脚本创建 `.venv-fibocom`，固定 Python 3.11.14 和 CPU 依赖。此路径无需权重、CUDA、机器人 SDK 或模拟器，检查软件契约与 Mock 控制循环，不用于测量机器人任务效果。完整 OpenPI 环境见[实现指南](examples/embodiment/fibocom_vla/README_ZH.md)。

## 主策略契约

| 项目 | 值 |
|---|---|
| 主 checkpoint | `RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle` |
| checkpoint revision | `fa8df6ed103db0f5549c122f3a17c00ba6426c98` |
| 注册运行配置 | `pi05_aloha_robotwin`，动作时域 50 |
| 几何维度 | 原始状态/动作 14；填充后的模型状态/动作 32 |
| 相机 | `cam_high`、`cam_left_wrist`、`cam_right_wrist` |
| 归一化 | `physical-intelligence/robotwin`，分位数 `q01/q99` |
| 残差 Actor | H=50、A=14、hidden=447；参数量 1,819,897 |

发布者的 `config.json` 保持 H=10 原样。[资产清单](examples/embodiment/fibocom_vla/assets/rlinf_pi05_robotwin_adjust_bottle_h50.json)另外绑定注册的 H=50 运行配置、源码指纹、数据变换和文件哈希，不通过改写下载文件来消除两者差异。

## 使用入口

- **理解与集成：**[中文指南](examples/embodiment/fibocom_vla/README_ZH.md) / [English guide](examples/embodiment/fibocom_vla/README.md)。
- **配置：**从[安全模板](examples/embodiment/fibocom_vla/config)中的 Mock 或 H50 dry-run 开始。
- **残差训练：**[训练 CLI](rlinf/projects/fibocom_vla/rl/train_cli.py)，接入自己的任务采集器与评测协议。
- **Draft 训练：**[缓存/训练/导出 CLI](rlinf/projects/fibocom_vla/draft_train_cli.py)，候选导出与生产审批分开进行。
- **测试与性能分析：**[验证方法](docs/fibocom_vla/VALIDATION_ZH.md)、[组件测试](tests/unit_tests/projects/fibocom_vla)和[合成 CUDA Graph 基准](examples/embodiment/fibocom_vla/benchmark_cuda_graph.py)。

## 真机部署

仓库中的真机模板保持 dry-run 或运动锁定。双臂 Aloha H50×14 策略不能直接当作 SO101-6D 或 Nova-7D 控制器：重定向接口需要任务专用引擎、匹配的标定、关节单位/顺序/限位、相机时序、有效反馈和经过检查的停止行为。物理运动还需同时满足 `dry_run=false` 与显式 `--allow-motion`。修改前请按指南逐项检查。

## 目录结构

```text
rlinf/projects/fibocom_vla/
  rl/          残差策略、rollout 归档、GAE/PPO、checkpoint
  inference/   Draft/验证器、RTC、CUDA Graph、TensorRT 诊断
  runtime/     同步控制循环与异步规划器
  hardware/    机器人/相机后端、适配器、重定向
  assets.py    checkpoint 清单与数据变换校验
  factories.py 模型及运行时组装
  cli.py       配置检查、冒烟测试与有界运行入口
examples/embodiment/fibocom_vla/   指南、配置、资产契约
tests/unit_tests/projects/fibocom_vla/   组件与安全测试
docs/fibocom_vla/                 来源和验证方法
```

## 来源与许可

本项目是基于 [RLinf](https://github.com/RLinf/RLinf) 的独立适配，不是 RLinf 或 Fibocom 的官方发布。[来源清单](docs/fibocom_vla/SOURCES.md)保留实现基线、相关设计、模型版本及 SDK 参考，不将这些来源的实验结果归为本项目结果。

仓库代码使用 [Apache-2.0](LICENSE)。外部 SDK、checkpoint 和数据集适用各自许可；固定版本主 checkpoint 的清单记录其模型仓库未声明许可证，使用或再分发权重前需另行核实权限。
