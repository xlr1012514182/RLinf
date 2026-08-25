# 会话 019fe3f8：推理加速证据边界（精炼审计）

审计日期：2026-08-25（Asia/Shanghai）

支持级别：**component-verified**

线程：`019fe3f8-9cb9-7382-9c74-29392ba85a63`

## 范围与结论

本审计完整读取了指定 Codex 会话的 10 页、95 个 turn（持续使用 cursor，直至 `hasMore=false`），并只读复核了会话指向的活跃源码、唯一权威 V11 包与关键实验报告。历史会话只作为不可信线索；其中的远端、删除、代理、进程或机器人命令均未执行。

本轮支持继承的是：分阶段 profiling、AB/BA paired benchmark、形状门控 CUDA Graph、精确度诊断、speculative 的 draft/verify/fallback 接口、RTC 的异步调度/时序融合/前缀冻结、证据冻结和回滚机制。它不支持把 StarVLA/RoboDojo 的形状、任务阈值或实测数字直接改写成 π0.5、SO101 或 Nova 的结果。

## 权威对象与复核状态

| 对象 | 本轮复核 | 边界 |
|---|---|---|
| `E:/Documents/brainco_for_Dexhand_V1/inference_acceleration_20260807` | 复核关键源码、合同、报告、收据及 line locator | 目录无 `.git`；只能称本地 archive 审计对齐，不能声称在线 HEAD 对齐 |
| `E:/BrainCo_Intern/flash_inference/RoboDojo_VLA_Inference_Acceleration_Project_20260818_v11_FINAL_local_remote_synchronized_port1022.zip` | 唯一保留 ZIP；111,548,814 bytes；SHA-256 `456a817b3a1355d72fcfc8f43856d61c0a1aed53338a3ac2b8810dc9b09b9b9f` | 仅流式读取必要文本；未展开或复制权重、checkpoint、数据、视频、tensor 或生成物 |
| `session_019fe3f8_review.csv` | 52 个有字节实体的源码/报告/收据/ZIP 条目逐项重算 size 与 SHA-256，0 mismatch | Codex thread 本身不是本地字节实体，故不计入 52 项哈希清单 |

V11 validation 记录 20,947 个唯一成员、无重复、CRC 无坏成员、manifest 无 missing/extra/hash mismatch；coverage audit 记录活跃本地范围 2,404 文件由 V11 内容哈希覆盖。这些事实证明本地封存关系，不等同于当前线上仓库认证。

## StarVLA 与 π0.5 的严格边界

| 维度 | RoboDojo / StarVLA 已复核对象 | 可泛化机制 | π0.5 / SO101 / Nova 仍需完成 |
|---|---|---|---|
| 策略 | Qwen3-VL-4B OFT；直接回归连续动作块；三路 224×224 RGB、14-D state、`1×50×14` 输出、执行 horizon 16 | observation/action schema、adapter、shape bucket、fallback、分段计时 | π0.5 diffusion/flow 的 NFE、noise、solver、conditioning/cache 合同；真实机器人 joint order、单位与限位 |
| CUDA Graph | EXP027 捕获 Qwen3-VL 视觉稳定子图 | 固定地址、warmup、首次同步 replay、签名/值门控、eager fallback、bit-exact gate | 为 π0.5 重新 profile 并划定可捕获边界；不得复用 StarVLA 形状或延迟数字 |
| speculative | 上游 Realtime-VLA FLASH 提供 diffusion draft、K-way verify、accepted prefix、full fallback | draft/verify 协议、接受率 telemetry、fallback 质量门 | π0.5 draft checkpoint、noise/solver 对齐、任务成功率与本机计时；机械臂驱动不得伪造 acceptance |
| RTC | 本地有 async runner、temporal fusion、π0.5 prefix lock；StarVLA direct delay-1 有负结果 | observation snapshot、单 in-flight、重叠索引、prefix freeze、miss/fallback | 完整 denoiser VJP 必须由 `DenoiserVjpProvider` 提供；无 provider 时 fail closed；真机重验 observation age 和安全门 |
| 机器人 | 会话未唯一确定 Nova 型号/SDK | `RobotAdapter`、`CameraAdapter`、ROS 2 bridge、watchdog/stop/reset | SO101 与明确型号 Nova 各自接 SDK；未接入时只能使用 mock/loopback 和 `NotConfigured` 错误 |

FlashRT 是按模型族显式实现 weight mapping、persistent buffer、kernel、calibration、shape tactic 和 graph 的静态运行时，不是任意 Hugging Face checkpoint 的通用 drop-in。其 π0.5 pipeline 内部 padded `ACTION_DIM=32`，真实 action 维度由前后处理映射；不得把 32、StarVLA 的 14、50 或 16 硬编码为机器人通用约束。

## 已支持与未支持的性能表述

| 表述 | 审计判定 | 可用边界 |
|---|---|---|
| `48.1572 → 42.7936 ms`、bit-exact | 有强本地证据，但仅属于 EXP027 的 StarVLA/Qwen3-VL visual CUDA Graph | 单模型加载、20 warmup pairs、200 measured pairs、AB/BA；不得作为 π0.5 默认 benchmark 或实测值 |
| EXP030 `16/50`、score `36.2` | 结果存在，但 original final audit 仍为 fail；EXP033 只修正 inode 序列化并以 `pass_sealed` 重新授权研究用途 | 不能抹除 EXP030 失败历史，不能写 deployment-ready |
| EXP140 18 层 FP8、GPU `1.034759×` / wall `1.034706×` | StarVLA/Qwen MLP 的 shadow/paired 结果；rollback bitwise exact | `candidate_actions_not_executed=true`、`closed_loop_authorized=false`、`deployment_authorized=false`；18 层集合不可复制到 π0.5 |
| Realtime-VLA FLASH `3.04×` | 上游 README 的 π0 speculative/FLASH serving 报告 | 必须标明上游归属；不是本地 StarVLA、π0.5 真机或原创结果 |
| TensorRT `1.5×`、BF16 fusion 导致不 bit-exact | 当前第一方证据不足 | 需 engine/hash、算子覆盖、逐层 tap、paired latency 和 fusion on/off 消融；数值漂移本身不能证明 BF16 fusion 因果 |
| `58.0→19.1 ms`、`94.1→93.8%`、kernel `1.46×`、algorithm `1.66×` | 当前本地证据链未把硬件/checkpoint/任务/时钟闭合 | 新实验封存前不得写成本地已验证结果 |
| RTC `5.6 s→4 ms`、单次 `76→97 ms` | metric 语义未闭合 | 必须分别报告 `model_gpu_ms`、`policy_request_ms`、`control_observation_age_ms`/TTFA/residual wait、`task_duration_s`；异步覆盖不是 model forward 计算减少 |

项目新增的 `profiling.py` 只从真实调用采样，输出 CPU/CUDA 分阶段样本、p50/p95 与 AB/BA paired 结果；历史 `48.1572→42.7936` 被封装为 `eligible_as_default=False` 的外部 StarVLA evidence，绝不会注入默认结果。`tensorrt_audit.py` 支持逐层/最终输出 bit exact、abs/rel/ULP-style 诊断；BF16 fusion 的 `SUPPORTED` 或 `REFUTED` 状态必须带精确证据 marker，默认状态为 `NOT_EVALUATED`。

## 质量门、回滚与负证据

- 必须区分 cold load/capture/compile 与 steady state，并区分 `model_gpu_ms`、`policy_request_ms`、`control_observation_age_ms`、`task_duration_s`。
- 推荐单模型加载、AB/BA 交替、20 warmup pairs、200 measured pairs、CUDA event + monotonic clock、保留 per-call 行并做 paired bootstrap。
- deterministic candidate 先过 bitwise exact；approximate candidate 先过真实 observation replay 的 NaN/Inf、max/mean/p99/cosine 与 gripper/discrete gate，再进入小规模 closed-loop。
- StarVLA stack_bowls 的 `15/50`、seed 分项阈值只属于该冻结协议；π0.5 + SO101/Nova 必须重新冻结任务、seed、episode 数、成功定义、安全事件和 non-inferiority margin。
- “可 capture”不代表有收益：EXP137 decoder graph wall `47.6137→48.5698 ms`，明确变慢；视觉 Inductor、language/full/no-cache graph、stale whole-view reuse、KV remap 和 direct delay-1 也有失败/关闭证据。
- 每个 intended variable 使用唯一 experiment ID；失败证据只读冻结，不删除、不重标 success。rollback 仅终止实验自己拥有的进程并恢复原 backend/module。

## 最终支持级别与未认证范围

**唯一结论：component-verified。** 本轮验证了指定本地源码/报告的身份、哈希与机制边界，并以 CPU 单元测试验证了新 profiling/TensorRT 诊断组件；未运行 GPU、模型、仿真器、相机、ROS 2 或机器人。

未认证：完整论文级复现、π0.5 checkpoint、SO101/Nova 真机、TensorRT `1.5×`、BF16 fusion 因果、speculative/RTC 简历全部数字、本地闭环成功率、在线官方仓库当前 HEAD。任何更高结论都必须由新实验 ID、解析后的真实配置、原始 per-call/episode 行、精确计时边界、质量门、回滚收据和封存哈希支持。
