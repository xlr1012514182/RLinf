# Source inventory / 来源与许可

[English overview](../../README.md) · [中文首页](../../README_ZH.md) · [Machine-readable lock](sources.lock.json)

This inventory pins source identities, not experimental outcomes. The dependency lockfiles remain the installation authority; a reference revision here is not a requirement to install every referenced project.

本清单固定来源身份，不记录实验结论。安装依赖以 requirements 锁定文件为准；列出参考版本不代表需安装全部参考项目。

## Implementation and design references / 实现与设计来源

| Source | Pinned revision | Role / 用途 |
|---|---|---|
| [RLinf release/v0.2](https://github.com/RLinf/RLinf/tree/46213e88caa910a4a52e68bde4fb96416c1efa55) | `46213e88caa910a4a52e68bde4fb96416c1efa55` | Implementation base / 实现基线 |
| [RLinf reference snapshot](https://github.com/RLinf/RLinf/tree/12881eda376c8407bd85e4329e64b008d6aa9640) | `12881eda376c8407bd85e4329e64b008d6aa9640` | π0.5 config, transform and dependency comparison / 配置、变换及依赖参考 |
| [FlashRT](https://github.com/flashrt-project/FlashRT/tree/f72192b263b267994edd7bbff0a8c62c6da98948) | `f72192b263b267994edd7bbff0a8c62c6da98948` | CUDA Graph/exactness and RTC design reference / 相关运行时设计参考 |
| [realtime-vla-flash](https://github.com/dexmal/realtime-vla-flash/tree/da6ceccad603695a8a3d6fa14dd410c3aadb536f) | `da6ceccad603695a8a3d6fa14dd410c3aadb536f` | Draft, parallel verification, prefix acceptance and fallback reference / 推测推理设计参考 |

## Model assets / 模型资产

| Source | Revision | Contract |
|---|---|---|
| [RLinf π0.5 RoboTwin](https://huggingface.co/RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle/tree/fa8df6ed103db0f5549c122f3a17c00ba6426c98) | `fa8df6ed103db0f5549c122f3a17c00ba6426c98` | [Main manifest](../../examples/embodiment/fibocom_vla/assets/rlinf_pi05_robotwin_adjust_bottle_h50.json) |
| [Dexmal RealtimeVLA-Flash](https://huggingface.co/Dexmal/RealtimeVLA-Flash/tree/77b9a6f88fb100230bc78cb4cb361bd2e586f9fb) | `77b9a6f88fb100230bc78cb4cb361bd2e586f9fb` | [π0 LIBERO registry](../../examples/embodiment/fibocom_vla/assets/dexmal_flash_pi0_libero_drafts.json); warm-start only for π0.5 / 对 π0.5 仅作初始化来源 |

These manifests are runtime contracts and are included in the source distribution. Model weight files are not. Their hashes, model-family checks and approval gates must not be removed when preparing a deployment.

以上 manifest 属于运行契约，随源码分发；权重文件不在源码包中。部署时应保留文件哈希、模型族匹配及审批检查。

## Hardware API references / 硬件 API 参考

| Project | Runtime/reference revision |
|---|---|
| [LeRobot v0.4.4](https://github.com/huggingface/lerobot/tree/8fff0fde7c79f23a93d845d1a50e985de01f8b8a) | `8fff0fde7c79f23a93d845d1a50e985de01f8b8a` |
| [Dobot TCP-IP-Python-V4](https://github.com/Dobot-Arm/TCP-IP-Python-V4/tree/55ec1ec82201aaf2e6d54aa47b6272bbaa14c815) | `55ec1ec82201aaf2e6d54aa47b6272bbaa14c815` |
| [Dobot ROS2 V4](https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4/tree/def21d05149576b9aed261f38e62ea236a5d8ec5) | `def21d05149576b9aed261f38e62ea236a5d8ec5` |
| [RealSense ROS](https://github.com/realsenseai/realsense-ros/tree/60c850958d651130fc2cc3d10efb37ff5be93da5) | `60c850958d651130fc2cc3d10efb37ff5be93da5` |

SDK versions do not certify a physical setup. Robot firmware, calibration, endpoints, units, limits, camera timing and stop behavior belong to the deployment configuration.

SDK 版本不能替代真机部署检查；固件、标定、端点、单位、限位、相机时序与停止行为需要在目标设备上确认。

## License and attribution / 许可与归属

- This is an independent adaptation, not an official RLinf or Fibocom release. Existing copyright headers and the parent [Apache-2.0 license](../../LICENSE) are retained.
- Referenced projects, optional SDKs, weights and datasets retain their own licenses. Source citations do not transfer redistribution rights or imply endorsement.
- The main model manifest records `not_declared_by_checkpoint_repository` for its pinned checkpoint. Verify permission separately before using or redistributing that asset.
- Upstream performance numbers and related projects' experiments are not results of this source distribution.

本项目为独立适配，保留上游版权头与代码许可。外部软件、权重、数据适用各自条款，来源引用不代表再分发授权或官方背书。主 checkpoint 的固定版本清单记录模型仓库未声明许可，使用前需另行核实；不将上游或参考项目成绩当作本项目成绩。
