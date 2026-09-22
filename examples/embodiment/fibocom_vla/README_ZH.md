# Fibocom π0.5 VLA Stack — 实现指南

[English](README.md) · [项目首页](../../../README_ZH.md) · [测试方法](../../../docs/fibocom_vla/VALIDATION_ZH.md) · [来源与许可](../../../docs/fibocom_vla/SOURCES.md)

以下命令均在仓库根目录执行。CPU 示例无需硬件；模型与训练示例需要各节明确列出的外部资产。`/opt/fibocom-assets` 等绝对路径是示例，请替换为自己的实际目录。

## 1. 环境

### CPU 组件

Ubuntu/WSL2 x86_64 安装脚本将 Python 3.11.14 和锁定依赖安装到 `.venv-fibocom`：

```bash
bash requirements/fibocom_vla_quickstart.sh
.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli validate-config \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json
.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli actor-report \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json
.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli mock-smoke \
  --config examples/embodiment/fibocom_vla/config/mock.json --steps 16
.venv-fibocom/bin/python -m pytest -q tests/unit_tests/projects/fibocom_vla
```

`mock-smoke` 使用 Mock 策略，并要求机器人和相机均为 Mock 后端，不加载 π0.5。安装脚本面向 Linux，不是 Windows 原生全栈安装器。

### OpenPI / GPU

完整 RLinf/OpenPI 栈使用独立的 Linux 环境：

```bash
bash requirements/install.sh embodied --model openpi --env robotwin --install-rlinf
```

安装器绑定 `rlinf-openpi==0.1.1` 与 [OpenPI 模型依赖](../../../requirements/embodied/models/openpi.txt)。后续 `python` 指此模型环境，不自动指向 CPU venv。选择对应后端时再单独安装机器人 SDK、ROS2 与相机驱动；不要脱离固定环境随意升级模型依赖。

## 2. 模型契约与资产

| 项目 | 主策略 |
|---|---|
| checkpoint | `RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle` |
| revision | `fa8df6ed103db0f5549c122f3a17c00ba6426c98` |
| 注册配置 | `pi05_aloha_robotwin` |
| 运行几何 | H=50；原始状态/动作=14；模型状态/动作=32 |
| 统计文件 | `physical-intelligence/robotwin/norm_stats.json`，分位数 `q01/q99` |
| 观测相机 | `cam_high`、`cam_left_wrist`、`cam_right_wrist` |
| 模型图像键 | `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb` |
| 环境输出 | 逆 delta 和 Aloha 变换后的绝对关节/夹爪目标 |

输入输出链包含 Aloha 关节/夹爪转换、状态归一化与填充，并对 12 个机械臂关节使用 delta 变换，两个夹爪不使用。发布者的 H=10 `config.json` 保持原始字节不变。[主模型清单](assets/rlinf_pi05_robotwin_adjust_bottle_h50.json)另行固定注册的 H=50 配置、RoboTwin YAML 与变换源码。工厂拒绝模型或栈维度不匹配；不能替换为不相关的 state-15/action-17 `fibocom_mobile` 统计。

在模型环境中准备好 Hugging Face CLI 后，下载完整固定版本。清单覆盖 17 个文件，包括全部权重分片：

```bash
MAIN_ROOT=/opt/fibocom-assets/RLinf-Pi05-RoboTwin-SFT-adjust_bottle-fa8df6e
hf download RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
  --revision fa8df6ed103db0f5549c122f3a17c00ba6426c98 \
  --local-dir "$MAIN_ROOT"

python -m rlinf.projects.fibocom_vla.cli verify-checkpoint-assets \
  --manifest examples/embodiment/fibocom_vla/assets/rlinf_pi05_robotwin_adjust_bottle_h50.json \
  --root "$MAIN_ROOT" \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json
```

验证覆盖大小、SHA-256、归一化/配置维度、源码指纹与栈语义。这是资产校验，不执行模型前向。代码许可证不等于权重许可证，见[来源与许可](../../../docs/fibocom_vla/SOURCES.md)。

## 3. 不驱动真机的主模型推理

将已校验资产绑定到默认工厂：

```bash
export FIBOCOM_PI05_MODEL_PATH="$MAIN_ROOT"
export FIBOCOM_PI05_ASSET_MANIFEST="$PWD/examples/embodiment/fibocom_vla/assets/rlinf_pi05_robotwin_adjust_bottle_h50.json"
export FIBOCOM_PI05_CONFIG_NAME=pi05_aloha_robotwin
export FIBOCOM_PI05_DEVICE=cuda
export FIBOCOM_PI05_DENOISE_STEPS=5

python -m rlinf.projects.fibocom_vla.cli openpi-checkpoint-smoke \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json \
  --seed 17

python -m rlinf.projects.fibocom_vla.cli run \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json \
  --instruction "adjust the bottle" --steps 16
```

第一条使用合成图像/状态检查模型加载与前向；第二条将真实模型接到 Mock 机器人/相机。两者都不是 RoboTwin 任务评测。manifest 模式可以从清单推导配置名称；显式指定时必须一致。不要使用 `FIBOCOM_WRIST_CAMERA` 覆盖清单中的相机顺序。

## 4. 策略模式与 RTC

| 模式 | 配置 | 所需资产 |
|---|---|---|
| 基础策略 | `residual_rl.enabled=false`，`speculative.enabled=false` | 校验过的主模型 |
| 残差策略 | `residual_rl.enabled=true`，`speculative.enabled=false` | 匹配的残差 checkpoint 与校验侧文件 |
| 推测策略 | `residual_rl.enabled=false`，`speculative.enabled=true` | π0.5 Draft 候选、评测、审批及可信公钥 |

训练、采集与部署须保持配置一致，运行时检查 checkpoint/config 哈希。残差与推测模式不能同时开启。

RTC 默认执行 8 步、队列阈值 2 步、重叠时域 16 步；规划器并行生成与执行动作块，并冻结已承诺前缀。原生 OpenPI RTC 将同次前向的归一化动作保存在 `ActionChunk.model_values`，用于模型空间重叠引导。残差/推测包装器与非线性重定向走主机输出空间回退，不等于原生 VJP 引导。无法及时停止的在途加速器调用，需要重启其所属进程后再使用对应模型/GPU 上下文。

## 5. 残差后训练

基座保持冻结，有界残差 Actor 修正基础动作块；独立 V 头和 Twin-Q 辅助 Critic 支持 GAE 与单轮 PPO。H50×14 配置对应 1,819,897 个 Actor 参数，hidden=447、bottleneck=512、feature dimension=2048。

从 H50 dry-run 模板复制自己的 `artifacts/residual_config.json`，设置 `residual_rl.enabled=true`，保持推测关闭、硬件为 Mock/dry-run，然后验证配置。`artifacts/` 默认不进入 Git。初始化、采集、更新和模型组装均使用**同一份配置**。

```bash
python -m rlinf.projects.fibocom_vla.cli validate-config \
  --config artifacts/residual_config.json

python -m rlinf.projects.fibocom_vla.rl.train_cli initialize \
  --config artifacts/residual_config.json \
  --output artifacts/residual_init.pt \
  --base-model RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle \
  --base-model-revision fa8df6ed103db0f5549c122f3a17c00ba6426c98 \
  --seed 17 --device cpu

# 使用该行为 checkpoint 采集兼容 rollout 后执行：
python -m rlinf.projects.fibocom_vla.rl.train_cli update \
  --config artifacts/residual_config.json \
  --input-checkpoint artifacts/residual_init.pt \
  --rollout artifacts/rollout.npz \
  --output artifacts/residual_updated.pt --device cpu
```

这两个训练命令仅操作残差状态和归档，不连接配置中的机器人，也不加载 π0.5 权重。基座 ID/revision 是操作者提供的元数据，需绑定到另外校验过的主模型资产。更新前会检查配置、行为 checkpoint、基座身份及 rollout 几何是否匹配。

[Rollout 记录器](../../../rlinf/projects/fibocom_vla/rl/collector.py)是集成 API，不是自动任务采集程序。需由任务采集器提供观测、奖励、终止/截断及行为策略数据。训练采集应使用随机残差采样并保留 log-probability；部署工厂默认确定性残差推理。加载更新后的 checkpoint 时，将 `FIBOCOM_RESIDUAL_CHECKPOINT` 设为其绝对路径，并保留生成的校验侧文件。基线与更新策略应使用相同任务协议评测。

## 6. Draft 训练与部署

[公开 Draft 注册表](assets/dexmal_flash_pi0_libero_drafts.json)描述的是 **π0 LIBERO**，不是可直接用于 π0.5 RoboTwin 的 Draft：

```bash
DRAFT_ROOT=/opt/fibocom-assets/Dexmal-RealtimeVLA-Flash-77b9a6f
hf download Dexmal/RealtimeVLA-Flash draft_libero_10.pt \
  --revision 77b9a6f88fb100230bc78cb4cb361bd2e586f9fb \
  --local-dir "$DRAFT_ROOT"

python -m rlinf.projects.fibocom_vla.cli verify-draft-asset \
  --registry examples/embodiment/fibocom_vla/assets/dexmal_flash_pi0_libero_drafts.json \
  --root "$DRAFT_ROOT" --suite libero_10
```

其他 suite 为 `libero_goal`、`libero_object`、`libero_spatial`。验证原始资产后会返回 `pi05_direct_use: false`；派生 π0.5 Draft 时只复制兼容的 warm-start 参数，并初始化新的动作头。

[Draft CLI](../../../rlinf/projects/fibocom_vla/draft_train_cli.py)提供分离的阶段：

| 命令 | 输入 / 输出 |
|---|---|
| `materialize` | 任务数据适配器（`module:function`）、数据清单、主模型持有的 teacher 与划分种子 → teacher cache |
| `verify-cache` | cache manifest → 内容与血缘检查 |
| `train` | cache manifest、`DraftTrainingConfig`、原始 Draft 注册表/根目录/suite → 优化器训练记录 |
| `verify-run` | run manifest → checkpoint 与运行检查 |
| `export-candidate` | 已验证的 cache/run 与 warm-start 身份 → 未签名候选 |

```bash
python -m rlinf.projects.fibocom_vla.draft_train_cli materialize --help
python -m rlinf.projects.fibocom_vla.draft_train_cli train --help
python -m rlinf.projects.fibocom_vla.draft_train_cli export-candidate --help
```

需提供任务数据适配器及符合[训练 schema](../../../rlinf/projects/fibocom_vla/inference/pi05_draft_training.py)的 JSON。训练代码版本必须对应干净 Git 工作树；从 ZIP 开始时，先将经过检查的源码纳入真实 Git 提交。候选导出设置 `production_authorized=false`，CLI 不签署审批。生产加载器另行要求绑定内容的留出集评测和独立 Ed25519 审批。

对已审批的 π0.5 候选，开启推测模式并设置：

```bash
export FIBOCOM_PI05_DRAFT_MANIFEST=/absolute/path/candidate/manifest.json
export FIBOCOM_PI05_DRAFT_EVALUATION=/absolute/path/evaluation.json
export FIBOCOM_PI05_DRAFT_APPROVAL=/absolute/path/approval.json
export FIBOCOM_PI05_DRAFT_PUBLIC_KEY=/absolute/path/ed25519_public_key.raw
export FIBOCOM_PI05_DRAFT_SIGNING_KEY_ID=reviewed-key-id
```

公钥为 32 字节原始文件。可选默认值为 `FIBOCOM_PI05_DRAFT_DTYPE=bfloat16`、`FIBOCOM_PI05_SPECULATIVE_BACKEND=auto`、`FIBOCOM_PI05_SPECULATIVE_VERIFICATION_TIMES=0.10,0.05`。工厂自行构造与 checkpoint 绑定的并行验证器，此生产路径无需额外的 Draft/verifier 插件工厂。运行时按连续前缀接受动作，接受不足时由主模型同轮接管。

## 7. CUDA Graph 与 TensorRT

视觉捕获仅包装 PaliGemma 视觉塔及投影层。需要 CUDA，并使用符合条件的非 RTC、非推测、纯推理配置：

```bash
export FIBOCOM_PI05_CUDA_GRAPH_VISUAL=true
export FIBOCOM_PI05_CUDA_GRAPH_CACHE_CAPACITY=4
export FIBOCOM_PI05_CUDA_GRAPH_WARMUP=3
```

形状检查、逐位一致性门控与 eager 回退始终保留。仓库的 H50 模板默认开启 RTC，因此只设置该环境变量不会激活图捕获。TensorRT 工具接收外部提供的 ONNX 图或网络，构建引擎清单并输出 parity/ULP/逐层差异诊断；源码包不附整策略 π0.5 一键导出路径或引擎。合成基准与实际模型计时分开，见[验证方法](../../../docs/fibocom_vla/VALIDATION_ZH.md)。

## 8. 机器人接入

| 模板 | 用途 |
|---|---|
| [mock.json](config/mock.json) | CPU Mock 组件与控制循环 |
| [robotwin_pi05_h50_dry_run.json](config/robotwin_pi05_h50_dry_run.json) | H50×14 主策略契约，机器人/相机为 Mock |
| [SO101 H50 重定向](config/robotwin_pi05_h50_so101_6d_retargeting_locked.json) | 锁定的 H50×14 → SO101 六坐标接口 |
| [Nova H50 重定向](config/robotwin_pi05_h50_nova_7d_retargeting_locked.json) | 锁定的 H50×14 → Nova 七坐标接口 |
| [so101_realsense.json](config/so101_realsense.json) | SO101 / RealSense 后端模板 |
| [dobot_nova_ros2.json](config/dobot_nova_ros2.json) | Dobot TCP 后端与 ROS2 相机 |
| [ros2_nova5_arm.json](config/ros2_nova5_arm.json) | 六轴 ROS2 关节后端模板 |

SO101 是五个机械臂关节加一个夹爪。Nova 七坐标路径是六个机械臂关节加夹爪，并要求独立数字输入夹爪反馈。通用仿射映射不能把双臂 Aloha 动作直接变为已标定的单臂控制器；需要在既有适配边界实现任务专用重定向、状态编码、FK/IK、可行性检查及夹爪转换。

### 部署检查

1. 核对机器人身份、SDK/串口/TCP 端点、ROS2 topic/service 和相机流。硬件 `dry_run` 仍可能连接并读取设备。
2. 核对相机顺序、H50×14 状态/动作语义、机器人关节顺序、单位、scale/offset、限位与控制周期。
3. 提供真实标定，仅在审查后设置适配器 validated，并匹配重定向引擎的 calibration ID。
4. 与硬件核对限位后，才可解除 `limits_require_hardware_validation`；不要用未经验证的值替换锁定占位项。
5. 在部署负载下检查相机偏差、状态/相机偏差、观测新鲜度和动作生成年龄。
6. 检查有确认的停止/急停行为、控制器互锁及实体急停，保持操作者在场。
7. 先进行有界 dry-run 读取；满足以上条件后，同时设置 `robot.dry_run=false` 并传入 `--allow-motion`，才授权有界物理运动。

这些检查属于运行时的一部分。配置检查通过不等于完成标定或获得运动授权。

## 9. 测试与源码分发

[验证方法](../../../docs/fibocom_vla/VALIDATION_ZH.md)分别描述组件测试、模型冒烟、合成性能分析和任务评测。源码包保留测试/基准程序，不携带运行结果归档；自己的不可变运行记录应另行保存。[来源与许可](../../../docs/fibocom_vla/SOURCES.md)保留上游及资产版本。
