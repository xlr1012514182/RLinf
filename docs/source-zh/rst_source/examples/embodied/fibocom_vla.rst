Fibocom π0.5 VLA Stack
================================

本示例将冻结的 π0.5 策略接入残差后训练、连续动作推理与机器人接口。
源码包包含运行契约、配置模板、测试及基准程序；权重、任务数据、部署标定与运行产物单独提供。

架构
----

* 冻结基座、有界残差、已审批 Draft/验证器是三个可选策略分支，残差与推测模式互斥。
* 异步规划器重叠动作块生成与执行。原生 RTC 要求模型空间动作血缘完整；包装器和非线性重定向使用明确的主机空间回退。
* 视觉 CUDA Graph 仅捕获视觉塔/投影层，不满足条件时回退 eager；RTC、推测与梯度上下文绕过捕获。
* TensorRT 工具接收外部提供的导出图/网络，仓库不附整策略 π0.5 导出器或引擎。

CPU 快速开始
------------

在 Ubuntu/WSL2 x86_64 的仓库根目录执行：

.. code-block:: bash

   bash requirements/fibocom_vla_quickstart.sh
   .venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli validate-config \
     --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json
   .venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli mock-smoke \
     --config examples/embodiment/fibocom_vla/config/mock.json --steps 16
   .venv-fibocom/bin/python -m pytest -q tests/unit_tests/projects/fibocom_vla

此路径检查 CPU 组件与 Mock 循环，无需权重或硬件。实际模型运行使用单独的 RLinf OpenPI 环境。

模型组装
--------

主模型为 ``RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle``，固定版本
``fa8df6ed103db0f5549c122f3a17c00ba6426c98``。注册的
``pi05_aloha_robotwin`` 使用 H=50、原始状态/动作 14、模型状态/动作 32，
以及固定顺序的高位、左腕、右腕相机。发布者 H=10 配置保持原样，由独立清单绑定运行契约与变换源码指纹。

资产验证后绑定 ``FIBOCOM_PI05_MODEL_PATH`` 与
``FIBOCOM_PI05_ASSET_MANIFEST``；显式指定的 ``FIBOCOM_PI05_CONFIG_NAME``
必须与清单一致。残差组装另需 ``FIBOCOM_RESIDUAL_CHECKPOINT``。
推测组装需 ``FIBOCOM_PI05_DRAFT_MANIFEST``、
``FIBOCOM_PI05_DRAFT_EVALUATION``、``FIBOCOM_PI05_DRAFT_APPROVAL``、
``FIBOCOM_PI05_DRAFT_PUBLIC_KEY`` 及 ``FIBOCOM_PI05_DRAFT_SIGNING_KEY_ID``，
工厂自行创建绑定的验证器。公开 π0 LIBERO Draft 对该 π0.5 目标仅作 warm-start 来源。

训练与硬件
----------

残差初始化/更新 CLI 消费外部 rollout 归档，任务采集与评测由集成方接入。
Draft CLI 生成 teacher cache、训练并导出未签名候选，留出集评测及独立审批另行进行。

机器人接口覆盖 SO101、Dobot Nova TCP、ROS2 关节，以及 OpenCV、RealSense、ROS2 相机。
真机模板保持 dry-run 或运动锁定。双臂 Aloha H50×14 接到单臂需任务专用重定向引擎，
并审查标定、单位、限位、反馈新鲜度和停止行为。运动还要求
``dry_run=false`` 与 ``--allow-motion``；配置检查通过不等于获得运动授权。

进一步阅读
----------

* :download:`实现指南 <../../../../../examples/embodiment/fibocom_vla/README_ZH.md>`
* :download:`验证方法 <../../../../fibocom_vla/VALIDATION_ZH.md>`
* :download:`来源与许可 <../../../../fibocom_vla/SOURCES.md>`

本项目为独立适配，不是 RLinf 或 Fibocom 官方发布。
