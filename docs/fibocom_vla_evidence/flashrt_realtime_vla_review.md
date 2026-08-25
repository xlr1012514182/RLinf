# FlashRT / realtime-vla-flash 固定提交审计（仓库内精炼版）

审计日期：2026-08-25（Asia/Shanghai）

审计类型：固定提交、只读、第一方源码与接口审计

执行结论：`component-verified`（仅静态源码/接口级）；`execution_status=not-run`

## 证据来源与完整性

本文件是工作区完整审计报告的仓库内精炼副本。完整逐文件 CSV 体积约
0.98 MB，未复制入项目树；仓库内统计摘要见
`flashrt_realtime_vla_files.csv`。

| 证据 | 工作区路径 | SHA-256 | 大小/记录 |
|---|---|---|---:|
| 完整审计报告 | `work/fibocom_vla_stack/audit/flashrt_realtime_vla_review.md` | `3af8e1864b919d4b402e34c20566e5638adfa0f5e3fd83acbbd4b7645bb70aab` | 27,592 bytes |
| 完整逐文件 manifest | `work/fibocom_vla_stack/audit/manifests/flashrt_realtime_vla_files.csv` | `84810af4dd50dae3ce92bd6eb84f33f48c21046c267c6996b108e94ab8bf7785` | 981,547 bytes / 2,552 records |

完整 manifest 每行字段为：

`repo,commit,path,mode,git_oid,bytes,lines,sha256,scope,reviewed,exclusion_reason`

枚举以 `git ls-files -s` 为准；普通文件按原始字节计算 SHA-256，gitlink
按 `gitlink:<git_oid>` 计算。`reviewed=yes` 表示逐行静态阅读，不表示成功
导入、编译或执行。

## 固定仓库身份

| 仓库 | origin | 固定提交 | 审计时工作树 |
|---|---|---|---|
| FlashRT | `https://github.com/flashrt-project/FlashRT.git` | `f72192b263b267994edd7bbff0a8c62c6da98948` | clean |
| realtime-vla-flash | `https://github.com/dexmal/realtime-vla-flash.git` | `da6ceccad603695a8a3d6fa14dd410c3aadb536f` | clean |

`realtime-vla-flash` 固定了两个外部 gitlink：ALOHA
`d1dc83afd89ded4379851257fe5d85632d31d5ec` 和 LIBERO
`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。本审计读取外层接口、调用点和
固定对象；嵌套实现不计入第一方逐行语义审查。

## 范围闭环

| 仓库 | tracked 条目 | reviewed 文本 | reviewed 行 | 排除条目 |
|---|---:|---:|---:|---:|
| FlashRT | 2,374 | 1,625 | 444,492 | 749 |
| realtime-vla-flash | 178 | 166 | 48,065 | 12 |
| 合计 | 2,552 | 1,791 | 492,557 | 761 |

明确排除：746 个 FlashRT upstream vendored 实现、11 个二进制/生成资产、
1 个视觉资产、1 个空占位和 2 个外部 submodule gitlink。排除项仍被枚举、
哈希并记录归属/接口边界；没有把生成资产或二进制伪称为语义阅读对象。

## 核心结论

1. `realtime-vla-flash` 已实现连续动作 draft、flow 插值、主模型 `B×K`
   并行验证、min-K 连续 prefix 接受、verifier tail 均值拼接，以及对应
   Triton/CUDA Graph 路径。这是简历第 ③ 项可复用的主要上游。
2. 上游 full fallback 的实际语义是下一轮执行：本轮仍返回 stitch 后动作，
   只设置 `pending_full_fallback`。简历所说“低接受率时主模型接管去噪”若指
   同轮替换，必须作为项目差异显式实现和打标。
3. FlashRT Pi0.5 捕获整 pipeline 和 decoder-only graph；它并不是“只捕获
   形状稳定视觉子图，动态分支 eager 回退”。底层 capture/fixed-buffer/
   subgraph hook 可复用，捕获边界和失败恢复必须重写。
4. FlashRT RTC 有 decoder 每步前后硬写 prefix，也有 host 侧异步指数输出
   融合；二者都不是 denoise-loop overlap VJP guidance。其 stock Pi0.5
   `rtc_vjp_guided.py` 明确缺少 provider，不能把 host 后融合冒充 native
   guidance。
5. 两仓没有可直接接入的 SO101、Nova、ROS2 驱动。ALOHA 真机样例使用
   ROS1 `rospy`/Interbotix；机器人 SDK、相机同步、单位/关节顺序、限位、
   watchdog 和 e-stop 必须在项目层补齐。
6. 两个固定提交不足以证明简历中的全部性能数字。本审计没有运行 GPU、
   CUDA、Triton、TensorRT、模型权重、仿真或真机，不声称性能复现。

## Continuous speculative 的精确语义

`src/openpi/models_pytorch/spec_pi0_pytorch.py` 的验证路径是：

1. 选择 K 个近终点验证时刻，默认 `(0.10, 0.05)`；
2. 构造 `x_t = t * noise + (1 - t) * x0_draft`；
3. 将 observation/KV/状态扩展成 `B×K`，一次主 denoiser 前向；
4. 由 `x0_hat = x_t - t * v_theta(x_t,t)` 得到 K 条 clean-action 估计；
5. 在下一次 replan 前的 `eval_h` 内计算按动作维数归一化的 L2/RMS 距离；
6. `D >= 7` 时半径最多使用前 6 维，gripper 使用单独 crossing guard；
7. 每个 K 独立求从第 0 步开始的连续合规 prefix，再取
   `min_k(prefix_len_k)`；
8. prefix 使用 draft，tail 使用 K 个 `x0_hat` 的均值。

上游两级 gripper guard 会在 verifier 任一 K 发生阈值跨越时清零 prefix，
并在 post-verify 阶段把已接受 prefix 截到首次 crossing。跨机器人移植时必须
先定义 gripper 维度、极性和阈值，不能默认套用第 6 维。

项目实现保留既有绝对/相对双阈值配置；因此它不是逐位复制上游单一
`tau_radius`。min-K、连续 prefix 和 tail stitch 顺序与上游一致，并在
diagnostics 中区分：

- `resume_override=True`：低接受率同轮完整主模型接管，属于简历要求；
- `resume_override=False`：本轮返回 stitch，下一轮强制 full main，属于上游
  reference state machine。

## CUDA Graph 与 Triton 边界

FlashRT `flash_rt/core/cuda_graph.py` 直接调用 CUDA Runtime API 完成 stream
capture、instantiate、launch 和 synchronize，可作为 framework-agnostic
handle。但现有类缺少 graph/exec/stream destroy 与异常清理；频繁 recapture
可能积累 native 资源。

FlashRT Pi0.5 先 warmup/capture 完整 pipeline，再 capture decoder-only，最后
调用 subgraph hooks。graph replay 抛错没有 catch-and-eager 恢复。若项目要支撑
“仅稳定子图捕获”，至少需要：

- 按 prompt shape、相机数、batch、horizon、dtype 建 graph key；
- 动态分支明确 eager；capture/replay 失败可观测回退；
- 固定 buffer 生命周期、graph destroy 和失败清理；
- 对 graph/eager 做输出 parity，而不是只测延迟。

`realtime-vla-flash/scripts/spec/triton/pi0_spec_infer.py` 提供动作 mix、
accept/metrics、tail stitch 和 time-bias 插值 kernel。注意 time-bias 插值不是
动作 flow 插值。fast verifier 绑定特定 32→1024→2560/4096→32 shape，draft
还要求 `num_kv_heads=1`；其他模型必须 capability gate 到 generic path，不能
静默误用 fast kernel。依赖/cache 缺失时上游明确报错，不允许 identity verify
伪回退。

## RTC 语义边界与项目补齐

FlashRT decoder 在每个 Euler/残差步的开始和结束都写回固定 prefix，提供严格
hard freeze；prefix 长度在 graph capture 中是结构常量。host 侧
`AsyncTemporalFusionRunner` 则按 controller time 对齐多个 chunk，以
`exp(-decay * distance)` 融合最终输出，并提供 block/hold-last deadline 策略。

两者之间缺少真正的 overlap guidance。项目新增的 PyTorch reference 必须在
每个 denoise step 内执行：

`x0_hat = x_t - t * v_theta(x_t,t)`

`L = 1/2 * sum_i w_i ||x0_hat[i] - old_chunk[i]||^2`

`w_i = w_0 * decay^i`

通过 `autograd.grad(L, x_t)` 计算穿过 denoiser 的 VJP，并在高时间到低时间的
Euler 更新中施加 guidance；hard prefix 必须在模型调用前和 update 后均写回。
host 输出 blending 仅可作为接线/dry-run fallback，并必须标记
`rtc_native_guidance=False`。

## 模型、机器人与相机接口

FlashRT runtime export 提供 images `(views,224,224,3)`、model action/noise
`(chunk,32)`、robot action `(chunk,robot_action_dim)`、text/state 与 RTC ports；
`robot_action_dim` 限制在 `[1,32]`。这是模型边界，不是 robot SDK。

realtime-vla-flash 有 ALOHA、LIBERO、DROID transforms 和 WebSocket
`BasePolicy.infer(dict)->dict` / `ActionChunkBroker`，但 WebSocket server 不是
ROS2 node。项目层至少应有：

- `RobotDriver`: connect/reset/read_state/send_action/stop/health/close；
- `CameraDriver`: 单调时间戳、frame id、内外参；
- `ObservationAssembler`: 多相机/state/prompt 同步和缺帧策略；
- `ActionAdapter`: joint order、单位、absolute/delta/velocity、gripper 极性、
  limit/rate-limit；
- `SafetySupervisor`: stale-action 拒绝、watchdog、e-stop、断线 hold/stop；
- SO101/Nova calibration schema、ROS2 lifecycle/QoS/cancellation 和 deterministic
  mock/replay driver。

## 依赖与官方入口（未执行）

FlashRT 声明 Python `>=3.10`，目标为 NVIDIA CUDA + POSIX Linux；CUDA kernels
需要 CMake 构建，并需要 CUTLASS v4.4.2。当前 CUTLASS 不是固定 submodule，正式
构建应锁完整 commit 和 SHA。

realtime-vla-flash 声明 Python `>=3.11`、`jax[cuda12]==0.5.3`、
`torch==2.7.1`、`triton==3.3.1`、`transformers==4.53.2`，LeRobot 固定到
`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`。README 报告 inference 需要
大于 8 GB 显存、建议 RTX 4090、测试 OS 为 Ubuntu 22.04。这些是上游声明，
不是本机验证结果。

## 可复用、必须重写与证据判定

| 目标 | 可复用 | 必须重写/补齐 |
|---|---|---|
| CUDA Graph | raw graph、subgraph hook、fixed buffers、runtime export | 稳定子图边界、动态 eager、异常回退、资源释放 |
| Speculative | draft head、flow/BK、min-K、stitch、gripper 思路 | same-round takeover、目标机器人 capability gate、draft 重训 |
| Triton | fused accept/stitch、fast/generic、graph fallback | 目标权重 shape gate、数值 parity、真实 profiling |
| RTC | async/deadline、controller alignment、hard-prefix graph | denoise-loop overlap VJP、动态 graph key、stream/控制周期协调 |
| I/O | native_v2、openpi transforms、WebSocket | SO101/Nova/ROS2、相机同步、action schema、安全层 |
| TensorRT | 其他模型的比较思路 | Pi0.5 engine、逐层 dump、BF16 fusion/rounding 消融 |

本地证据只能确认官网页面中的 `58.0 ms → 19.1 ms`、`3.04×` 和平均成功率
下降 0.3 个百分点；没有原始实验包。本提交内未找到可复算的
`48.2→42.8 ms`、`1.46×`、`1.66×`、`94.1→93.8%`、`5.6 s→4 ms` 或
`76→97 ms` 日志。所有简历指标都应绑定 commit、checkpoint、配置、任务列表、
raw log 和统计脚本。

## 许可证与归属风险

- 两仓根许可证均为 Apache-2.0；模型权重不自动继承代码许可证。
  `LICENSE_GEMMA.txt` 对 Gemma 模型另有条款。
- 发布时必须区分 Physical Intelligence/openpi 基线、Dexmal/FLASH 增量、
  FlashRT 和本项目修改，保留 Apache notices，不能把上游代码表述为原创。
- ALOHA 样例声明主要复制自 ACT；ALOHA/LIBERO submodule 需要独立核对 LICENSE/
  NOTICE。
- FlashRT vendor metadata 包含 BSD-3-Clause 的 FA2/CUTLASS/FA4、Apache-2.0
  的 FlashInfer XQA 和外部 openpi snapshot，发行物必须保留归属。
- `csrc/attention/sage2/**` 缺独立目录级 provenance；应补精确上游 commit、
  patch diff 和 NOTICE。CUTLASS 也应从 tag 提升为 commit 锁定。
- 根 Apache 许可证不授予机器人/相机 SDK、数据集、论文名或商标的再分发权。

## 验证门槛

1. 无 GPU：复核 manifest/commit/gitlink/SBOM；运行 flow direction、B×K、
   min-K/stitch、same-round/next-round fallback、hard-prefix、VJP、无 overlap 和
   `guidance_weight=0` 测试；robot adapter 做 deterministic replay。
2. 单 GPU correctness：锁 GPU/driver/CUDA/Torch/Triton/checkpoint；比较 PyTorch
   reference、Triton generic/fast、graph/eager 的 max-abs/relative/cosine，
   bit-exact 声明另做 raw-bit；覆盖阈值等号、首步/中途失败、K 分歧和 gripper
   crossing；连续 replay 至少 1,000 次检查 buffer 污染和资源增长。
3. 性能：CUDA Events 测 GPU region、单调 host clock 测端到端；报告 warmup、
   样本数和 p50/p90/p95/p99。TensorRT 必须同输入/preprocess/noise/steps，并逐层
   定位第一个数值差异。
4. 仿真/真机：固定任务、seed、episode、成功标准、控制频率、horizon 和
   fallback 模式；接 SDK 前先过 joint order、单位、限位、gripper polarity、
   timestamp freshness、watchdog、e-stop、断线恢复。RTC 分开报告模型计算、
   执行覆盖后的可见等待、冷启动和 deadline miss。

最终边界：这两个固定提交足以作为设计与复用来源，但不足以单独证明简历中的
全部机制与数值。项目实现和实验报告必须持续保留“上游已有、项目重写、尚未
验证”三种状态，不能混为同一证据等级。
