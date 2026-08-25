广和通 π0.5 后训练与实时推理工程
==================================

该示例提供冻结基座的 π0.5 残差后训练，以及面向 SO101、越疆 Nova、
通用 ROS2、OpenCV 与 RealSense 的可插拔真机运行时。同时包含形状门控的
CUDA Graph、连续动作 Draft/并行验证和 RTC 异步动作块执行。

实现与实验结论严格分开：上游 FLASH、论文、简历或历史 StarVLA 实验中的
数字不会被写成当前仓库的 π0.5 本地结果。当前命令只覆盖 CPU 单元测试、
Mock 与具名组件范围，不证明 GPU 加速、TensorRT 等价性、仿真成功率或真机
成功率。源码版本和完整证据边界见
``examples/embodiment/fibocom_vla/README.md``。

残差网络与策略—机器人契约
--------------------------

动作块为 50×7、隐藏宽度为 562 的残差 actor 恰有 1,818,542 个可训练参数；
动作块为 50×6、隐藏宽度为 581 时恰有 1,819,139 个。两者都使用基座模型
同一次前向产生的 2048 维冻结前缀特征，并围绕完整参考动作块输出有界残差。

两个 CLI 运行入口都会安装 ``PolicyRobotAdapterPolicy``：推理前将机器人
状态转换到 checkpoint 状态空间，推理后再将策略动作块转换成经过检查的
绝对机器人关节目标。内置路径支持可逆的关节仿射标定；末端空间或
checkpoint 原生空间必须通过 ``module:function`` 加载自定义适配器。

LIBERO 适配器会加载 ``robot.options.kinematics_factory``。工厂接收
``RobotConfig``，返回的 engine 必须实现：

.. code-block:: python

   encode_policy_state(observation) -> array_or_robot_state
   policy_chunk_to_robot_targets(
       observation, policy_values, period_s, action_adapter_config
   ) -> absolute_targets  # 形状为 [T, robot_action_dim]

该边界会检查相机键、维度、时间戳、控制周期、有限值、关节限位和逐步增量；
非法 IK 结果不会被静默截断。SO101 与 Dobot 的 LIBERO 示例使用混合动作
语义：第 0–5 维是笛卡尔 ``delta_from_observation``，第 6 维是
``absolute`` 夹爪命令。经审查的 engine 必须显式实现 checkpoint 状态编码、
FK/IK、单位、可达性与夹爪转换。

硬件与生命周期门禁
------------------

SO101 是五自由度机械臂加夹爪，并非六自由度机械臂。因此不能把六维末端
位姿增量直接改名为 SO101 六个电机目标；示例在接入经过审查、面向具体任务
的运动学 engine 前会保持 fail-closed。

Dobot 通过 ``ServoJ`` 下发六个角度制机械臂坐标，可通过 DO 下发第七维二值
夹爪命令；七维真机控制必须使用独立 DI 夹爪反馈。直接 SDK 后端在每次动作
前检查控制器错误、碰撞、文档规定的低有效 ``SafetyState`` 联锁，以及四个
安全皮肤 approach-pause 字段。

ROS2 关节后端支持两类有确认的停止服务：
``std_srvs/srv/Trigger``（``service_backend=trigger``），或
``dobot_msgs_v4/srv/Stop`` 与 ``EmergencyStop``
（``service_backend=dobot_v4``）。真机模式会预检命令订阅端和两个停止服务，
并要求显式 SI 关节单位。六轴 Nova5 机械臂模板位于
``examples/embodiment/fibocom_vla/config/ros2_nova5_arm.json``。

一次多相机同步读取共享同一个全局超时预算，并必须通过相机间 skew 门禁；
组合观测还需通过最大数据年龄与机器人状态—相机 skew 门禁。每次动作发送前
都会检查源观测年龄和动作生成年龄。真机还需通过标定与硬件限位校验，同时
设置 ``robot.dry_run=false`` 并传入 ``--allow-motion``。部分连接失败会回滚；
有限步运行会按结果选择 stop 或 emergency stop，再依次清理 planner、相机和
机器人，并保留清理错误。

Fail-closed 模型组装与 RTC
--------------------------

``FIBOCOM_PI05_MODEL_PATH`` 与 ``FIBOCOM_PI05_CONFIG_NAME`` 始终必填。
加载后会立即核对模型 ``action_horizon`` 与工程配置。RLinf v0.2 自带的
``pi05_libero`` horizon 为 10，因此不能用于仓库中的 50 步残差示例：
``action_chunk`` 只能截短输出，不能把短 horizon 扩成 50。必须选择已注册、
且与 50 步 checkpoint 匹配的配置，不能静默覆盖 stock checkpoint。启用残差
RL 后还必须提供 ``FIBOCOM_RESIDUAL_CHECKPOINT``；启用 speculative
推理后还必须同时提供 ``FIBOCOM_DRAFT_POLICY_FACTORY`` 与
``FIBOCOM_PARALLEL_VERIFIER_FACTORY``。缺少启用层所需的资产或工厂接口不兼容
时会直接报配置错误，不会静默改跑另一条策略链路。

OpenPI native RTC 通过 ``ActionChunk.model_values`` 携带同一次前向产生的
归一化模型动作，下一轮去噪严格用这条血缘构建硬前缀与重叠目标；血缘缺失时
fail closed。当前残差 wrapper、speculative wrapper 和自定义非线性 IK/FK
路径不能保持安全的 native RTC，planner 会改用带明确标签的 host 输出空间
fallback；该 fallback 不能被称作 VJP 引导。运行中的加速器前向无法安全取消：
若 planner 关闭超时，必须重启所属进程后才能复用模型或 GPU 上下文。

CPU 组件级检查
--------------

.. code-block:: bash

   python -m rlinf.projects.fibocom_vla.cli validate-config \
     --config examples/embodiment/fibocom_vla/config/mock.json
   python -m rlinf.projects.fibocom_vla.cli actor-report \
     --config examples/embodiment/fibocom_vla/config/mock.json
   python -m rlinf.projects.fibocom_vla.cli mock-smoke \
     --config examples/embodiment/fibocom_vla/config/mock.json --steps 16
   pytest -q tests/unit_tests/projects/fibocom_vla
