# Fibocom π0.5 VLA 后训练与推理运行时栈

[English](README.md) | [简体中文](README_ZH.md)

**冻结基座残差 RL · 连续动作 speculative inference · CUDA/TensorRT 诊断 · RTC · 真机接入**

本分支把来源锁定的 RLinf π0.5 RoboTwin 策略组织成可审计的后训练
与部署栈。

## 已实现内容

| 技术主线 | 实现 | 当前边界 |
|---|---|---|
| **残差 RL 后训练** | 冻结 π0.5 参考策略、H=50/A=14 下 1,819,897 参数有界残差 actor、独立 V head、Twin-Q 辅助损失、轨迹级 GAE、单轮 PPO、rollout/checkpoint | 组件已验证 |
| **CUDA Graph / TensorRT** | factory 接线的 vision tower/projector graph capture、bit-exact 门与 eager 回退；TensorRT 10 build/runtime manifest、parity、ULP 和逐层差异诊断 | 契约已验证 |
| **Speculative inference** | 签名 π0.5 Draft 门、单次 prefill 的 B×K OpenPI verifier、连续 prefix 接受、同轮主模型接管、Torch/Triton 后处理 | production 装配路径已实现 |
| **RTC** | 异步 action-chunk planner、已承诺 prefix 硬冻结、模型空间 overlap VJP、指数衰减权重和分离计时 | 合成组件已验证 |
| **机器人接入** | SO101、Dobot Nova、ROS2 和相机 backend；双 Aloha H50×14 到 SO101-6D/Nova-7D 严格 retargeting 边界 | adapter 与锁定模板已验证 |

## 来源锁定的模型目标

| 字段 | 契约 |
|---|---|
| 主权重 | `RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle@fa8df6ed103db0f5549c122f3a17c00ba6426c98` |
| 已注册运行时 | `pi05_aloha_robotwin`，H=50 |
| 归一化统计 | `physical-intelligence/robotwin`，quantile normalization |
| 几何 | raw state/action 14；归一化后 model state/action 32 |
| 相机 | `cam_high`、`cam_left_wrist`、`cam_right_wrist` |
| Draft 来源 | `Dexmal/RealtimeVLA-Flash@77b9a6f8` |

Publisher checkpoint metadata 记录的是 H=10。本仓不修改这些字节，而是核验
原始权重，再通过 SHA-256/源码指纹单独绑定 RLinf 已注册 H=50 运行时、
官方 RoboTwin YAML、norm stats、相机映射和 Aloha transform。

## 全新 Clone 后直接运行

以下代码块面向全新 Ubuntu/WSL2 x86_64 检出，仅使用 CPU。前置要求为
Git、Bash、带 `pip` 的 `python3` 和网络连接；不需要 CUDA、模型
checkpoint、ROS2 或机器人 SDK。请在 Bash 中完整执行：

```bash
git clone --branch feat/fibocom-vla-stack --single-branch \
  https://github.com/xlr1012514182/RLinf.git
cd RLinf

bash requirements/fibocom_vla_quickstart.sh

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli validate-config \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli mock-smoke \
  --config examples/embodiment/fibocom_vla/config/mock.json --steps 16

.venv-fibocom/bin/python -m pytest -q tests/unit_tests/projects/fibocom_vla
```

Bootstrap 会在仓库本地的 `.venv-fibocom` 中锁定 Python 3.11.14、
CPU PyTorch、RLinf OpenPI Transformers fork 以及该 smoke/test 路径使用的
全部依赖，不需要激活 shell。该路径只验证配置、Mock runtime 和单元测试
plumbing。完整 OpenPI/RoboTwin、CUDA/TensorRT、checkpoint、ROS2 与真机
部署仍需满足完整实现指南中的平台专用前置条件。

## 文档与证据

- [English 完整实现指南](examples/embodiment/fibocom_vla/README.md)
- [中文完整实现指南](examples/embodiment/fibocom_vla/README_ZH.md)
- [声明—证据边界](docs/fibocom_vla_evidence/claim_boundary.md)
- [复现矩阵](docs/fibocom_vla_evidence/reproduction_matrix.csv)
- [固定来源身份](docs/fibocom_vla_evidence/source_lock.json)
- [当前本地验证记录](docs/fibocom_vla_evidence/local_validation_20260826.md)
- [当前远程验证记录](docs/fibocom_vla_evidence/remote_validation_20260826.md)

## 上游血缘与许可

实现基座来自官方 [RLinf 仓库](https://github.com/RLinf/RLinf)。从当前 RLinf
选择的 OpenPI/RoboTwin 契约使用独立来源锁定，详见完整指南。
本分支代码沿用父项目 [Apache-2.0 License](LICENSE)。第三方 checkpoint
仍遵守其自身条款，本仓不重新发布或重新许可它们的权重。
