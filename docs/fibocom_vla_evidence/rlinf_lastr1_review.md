# RLinf / LaST-R1 / Stack-One 审计报告

审计日期：2026-08-25
审计性质：只读、源码与既有产物核验；未修改 vendor 或 `E:` 原件，未在本机复跑 IsaacLab/GPU 训练。
逐文件证据清单：`audit/manifests/rlinf_lastr1_files.csv`

## 1. 结论先行

1. **上游官方 RLT 与本地 `112_success` 不是同一算法。** 上游 commit `230ea79b4b0e1c1cbe13a3079baa14b93b441dac` 的 Stage 2 是冻结 Stage 1 特征模型后训练小型 off-policy actor-critic；TD3 变体使用 replay、direct reference-conditioned actor、Twin-Q 和 `-Q + BC`，没有 PPO/GAE。官方文档还明确说明当前 AC Stage 2 不是标准 maximum-entropy SAC（`docs/source-en/rst_source/examples/embodied/rlt.rst:205-225,330-340,631-643`；`fsdp_rlt_td3_policy_worker.py:109-174`）。
2. **本地 `112_success` 是可定位、可复用的第一方定制链。** 它把冻结 OpenPI/pi0.5 的 prefix mean 当作 2048 维 `z_rl`，再用 reference-conditioned passthrough actor、双 Q critic 和 LaST-R1 的 GAE/PPO 数学函数组成一个自定义混合训练器。它是“RLinf 环境/模型 + LaST-R1 PPO core + 本地 Stack-One 路由/奖励”的集成，不是上游官方 RLT Stage 2。
3. **“冻结 π0.5”成立。** 本地模型明确 `openpi.eval()` 且逐参数 `requires_grad_(False)`（`openpi_pi05_rlt_td3_official_pass_copy.py:342-345`）；上游当前实现也在 rollout worker 对 `rlt_feature_model` 执行 `eval()` 和 `requires_grad_(False)`（`huggingface_worker.py:153-160`）。
4. **“1.82M 残差 actor 与双 Q critic”必须改写。** 精确可训练参数为 actor **611,107**、双 Q critic **1,204,738**，合计 **1,815,845**；日志也记录 `trainable_params=1815845`。因此 1.82M 只能修饰“actor+critic 小头总计”，不能修饰 actor。actor 的实现名是 `direct_reference_passthrough_conditioned`，计算为 `atanh(ref)+tanh(learned)*scale` 后再 squash，语义上是参考动作附近的局部调整，但不是类名为 residual 的简单 `ref + delta`（模型 `81-217` 行）。
5. **“参考动作锚定”成立且有两层。** actor 零初始化后从参考动作 passthrough 起步；训练损失还包含 reference chunk-sum L2 与 executed-action chunk-sum L2（`trajectory_lapo_v2_terminal_guard.py:355-385`）。
6. **“chunk 级二值”成立；“稀疏、无需人工奖励塑形”不成立。** `stage_binary_guard` 对每个 active 且未触发 regression 的 chunk 给 1，否则给 0（helper `1659-1711,1852-1858`）。训练窗口 reward mean 为 `0.8257-0.9979`，正 reward chunk 占绝大多数，故它是**稠密二值守卫奖励**，不是稀疏奖励。回报依赖 grasp/goal/hold/drop/success 几何阈值、阶段状态和持久 regression，属于明确的人工任务塑形/门控。
7. **“失败尾部清零并定向负强化”大体成立，但细节应说准。** 从未 first-hit 的失败轨迹会整条清零；hit-then-drop/hold-fail 会从 drop/hold-fail/first-hit 对应尾段清零（helper `2131-2177`）。零 reward 的强负样本再获得 critic 权重 3；updater 将这些位置的 advantage 强制为 `-abs(A)` 且设 floor 0.15 后重白化（updater `213-228`）。这不是自然由二值 reward 自动得到，而是显式 advantage 改写。
8. **“轨迹级 GAE、单轮 PPO”可保留但必须限定。** updater 确实把每个环境 episode 组成 trajectory，调用 LaST-R1 `compute_gae_advantage_return`，配置 `ppo_epochs=1`。但每轮拆成 4 个 mini-batch；112 条链的每个采样窗口实际有 4 或 5 次 optimizer step，所以“单轮”只能表示一个 epoch/pass，不能表示一次梯度更新。
9. **112 条训练轨迹数字成立。** 首段 `5×16=80` episode，resume `2×16=32` episode，合计 112；对应 chunk/transition 为 `478+499+454+450+468+505+537=3391`。这些是模型闭环采集，但执行策略是 reference/actor 路由的混合策略，不是纯 actor 自采。
10. **75.0%→96.9% 的既有仿真产物成立，但不能写“真机”。** 基线 summary 是 `96/128=75.0%`、`30/128 hit_then_drop`；iter7 独立 sampler-only 评测四 shard 是 `32+30+30+32=124/128=96.875%`、`2/128 hit_then_drop`。然而基线 seed 为 93042、候选为 93107，且候选评测启用了 midchunk reference clamp、pre-hit reference pulse、late actor rescue。因此该数字证明的是 **IsaacLab/Franka 仿真中的 checkpoint + reference/heuristic route 组合策略**，不是裸 actor，更不是 SO101/Nova 真机。可达链只出现 `Isaac-Stack-One-Franka-IK-Rel-Visuomotor-Rewarded-v0`、IsaacSim cameras 和 `pi05_isaaclab_stack_cube`；没有 SO101、Nova、ROS2、真实相机或机械臂 SDK 调用。

## 2. 仓库身份与可比性

### 2.1 上游固定点

- 路径：`work/fibocom_vla_stack/vendor/RLinf`
- HEAD：`230ea79b4b0e1c1cbe13a3079baa14b93b441dac`
- origin：`https://github.com/RLinf/RLinf.git`
- 工作树：审计时 clean。

### 2.2 本地定制树

- 路径：`E:/fibocom_intern/bag/RL_rlt_Last-R1/RLinf`
- HEAD：`ce455aa6fdd01c7f4080ec26646be250ce5f0a27`
- commit 时间/标题：`2026-04-07T16:51:47+08:00`，`feat(behavior): support online/offline object/pose randomization (#976)`
- origin 同为官方 RLinf。
- 该 object database **不含**上游目标 commit；不能做可信的 `git diff target...local`。
- 本地工作树高度定制：可枚举到至少 8,834 个状态项（20 deleted、37 modified、8,777 untracked），且若干 result 深路径被 Windows long-path 跳过。因此比较采用“固定上游目标文件全文 + 本地可达成功链全文/语义等价 base+diff”方法，而非把 dirty tree 当成单一可审查 commit。
- 当前上游的 `rlinf/algorithms/rlt/`、RLT token transformer、RLT MLP policy/worker 和 RLT 文档在本地 Apr-2026 基线中均不存在；本地成功链是后置的 untracked/custom 文件族。

## 3. 两条算法链的实际结构

### 3.1 上游官方 RLT（Aug-2026）

可达入口 `examples/embodiment/train_embodied_agent.py:59-74` 根据 `algorithm.adv_type` 选择 RLT AC 或 RLT TD3 worker。Stage 1 用 pi0.5 + `RLTTokenTransformer` 学习 RLT 表示；Stage 2 rollout worker加载并冻结 feature model，route 产生：

`{z_rl, proprio, ref_chunk} -> small actor -> action chunk`

TD3 路径的事实：

- replay buffer warmup 与离策略更新；默认示例含 10k transition warmup 和 30k post-collect update（config `44-45`）。
- `gamma=0.99`、`tau=0.005`，actor objective 明写为 `-q_weight*Q + bc_weight*BC`（config `50-76`）。
- 2048 维 `z_rl`、9 维 proprio、10×8 action、两层 256 hidden、TwinQCritic（config `330-364`）。
- actor 直接拼接 feature 与 reference，MLP 输出完整 action 后 clamp；reference 通过输入与 BC loss 锚定，不是 residual passthrough（`rlt_td3_mlp_policy.py:45-100,125-150`）。
- 没有 PPO ratio、GAE 或“112 条轨迹”协议。

### 3.2 本地 `112_success`（May-2026）

可达调用链：

1. `launch_stackone_bridge_v2b_frombase_pi05_80eps...sh` 启动 5 个 16-env window。
2. `stackone_bridge_train_openpi_pi05_v2_terminal_guard.py` 动态加载本地 helper 和 policy，仅给 actor/critic 建 Adam optimizer（`168-180`）。
3. helper 在 IsaacLab Stack-One/Franka 中采样完整 episode，按 `near_place_or_success_hold` 在 frozen pi0.5 reference 和 actor 间路由，并生成 reward、失败尾部标签、critic 权重。
4. `StackOneTrajectoryLAPOUpdater` 用 LaST-R1 `core_algos.py` 的 GAE/PPO 公式更新 actor 与双 Q critic。
5. resume launcher 从 iter5 checkpoint 再跑 2 个 16-env window，得到 iter7。
6. v2d sampler-only launcher加载 iter7，关闭训练，启用三种动作守卫，做 `32 env × 4 shard` 独立评测。

模型细节：

- frozen OpenPI 同一次 forward 同时给 reference action 和 prefix output。
- 所谓 `z_rl` 是 frozen OpenPI prefix token 的均值池化，维度 2048；`encoder/decoder` 是 `Identity`（模型 `354-358,438-478`）。成功链并未调用当前上游的 `RLTTokenTransformer`，所以最稳妥称呼是 **RL-token-style frozen VLA prefix feature**，而不是“训练了外置 RL Token transformer”。
- 拼 7D proprio 得 2055 维 feature；拼 5×7 reference chunk 得 actor 输入 2090。
- actor：`2090 -> 256 -> 256 -> 35`，每层带 LayerNorm，共 611,107 参数。
- critic：两个 `2090 -> 256 -> 256 -> 1` Q head，共 1,204,738 参数。
- target actor/critic 是冻结 copy，不计 trainable；总可训练参数 1,815,845。

## 4. 简历主张逐项核验

| 主张 | 判定 | 最可信表述 |
|---|---|---|
| 基于 RLinf 框架 | 部分成立 | 复用 RLinf IsaacLab 环境、OpenPI wrapper、replay/helper；本地成功链不是目标 commit 的官方 RLT worker。 |
| 冻结 π0.5 全部参数 | 成立 | `eval()+requires_grad_(False)`，只优化外置 actor/critic。 |
| 外置 1.82M 残差 actor 与双 Q critic | 数字修饰对象错误 | 外置参考动作局部调整 actor 61.1 万 + 双 Q critic 120.5 万，总计 181.6 万。 |
| 参考动作作锚、抑制漂移 | 成立 | passthrough 初始化 + reference BCSUM + executed-action BC；但没有对“基座能力退化”做真机/跨任务量化。 |
| chunk 级二值稀疏奖励 | 一半成立 | 是 chunk 级 0/1；实际正奖励很密集，应称“二值守卫奖励”。 |
| 无人工奖励塑形 | 不成立 | 大量几何阈值、阶段、drop/hold/success regression 与后处理规则。 |
| 二次失败轨迹尾部清零 | 基本成立但术语含糊 | no-hit 整轨清零；hit-drop/hold-fail 从失败或 first-hit 尾段清零。 |
| 优势翻负成定向负强化 | 成立但属显式规则 | `strong_negative -> -abs(A).clamp(floor=0.15)`，随后重白化。 |
| 轨迹级 GAE | 代码存在，但有算法问题 | LaST-R1 GAE 被调用；输入的 “value” 实际是 `Q(s,a)` 聚合值，不是 `V(s)`。 |
| 单轮 PPO | 限定成立 | `ppo_epochs=1`，但每窗口 4/5 个 mini-batch optimizer step。 |
| 112 条模型自采样轨迹 | 成立但为混合策略 | 80+32 episode，3391 chunk；reference 控制大部分 chunk，actor 只在 handover 区间接管。 |
| 75.0%→96.9%，掉落降低 | 仿真产物支持 | 96/128→124/128，hit-drop 30→2；不同 seed，候选含 eval-time guards。 |
| 真机方块堆叠 | 不成立 | 所有可达代码、命令和 config 均为 IsaacLab/Franka 仿真，没有真机 SDK/ROS2 证据。 |

## 5. 必须修正的算法冲突

### 5.1 GAE 把 Q 当作 V

`_compute_old_values()` 调用 `critic.q_aggregate(feature, action)`，随后把结果作为 `values` 送入 GAE（updater `190-228`）。GAE 的标准基线应是状态价值 `V(s)`；当前值依赖 replay action，是 Q。接着双 Q 又以 GAE returns 为监督做 MSE（`337-345`），因此既不是标准 PPO value head，也不是标准 TD3 Bellman critic。

### 5.2 PPO 的 old log-prob 不是行为策略记录

代码在更新时构造：

`old_log_prob = Normal(mean=executed_action, std=fixed_std).log_prob(executed_action)`

即每个旧动作都位于其临时分布的均值（updater `247`）。真实 rollout 是 deterministic actor/reference 路由，部分 step 还被 clamp/pulse/rescue 改写；它没有在采样时保存真实 behavior distribution/log-prob。故 PPO ratio 是“当前 actor 离 executed action 多远”的伪似然，而非可信 importance ratio。

### 5.3 reference/heuristic 动作也进入 full actor PPO

成功 launcher 显式 `bridge_actor_ppo_mask_mode=all`。reference-controlled 和 v2d 守卫改写动作并非 actor 样本，却都可能被当作 actor PPO 数据；代码头注释/metric 的 `terminal_confidence` 文案与实际 config 不一致。七个训练 window 的 `bridge_action_pg_clipfrac` 全为 0，说明 clip 逻辑存在但在这条链上从未激活。

### 5.4 双 Q 没有参与成功链的 actor Q objective

本地成功 updater 的 actor loss是：

`0.50 * PPO + 0.25 * reference_BCSUM + 0.10 * executed_action_BCSUM`

demo 比例为 0。没有 `-Q(actor)` 项；双 Q 只提供伪 value baseline并拟合 GAE return。不能把这条链描述成官方 TD3+BC，也不能把 twin-Q 当作 PPO 必需组件。

### 5.5 评测策略与 checkpoint 不可混称

iter7 v2d eval 的 config 明确：

- `midchunk_success_clamp=true`, source=`reference`；
- `prehit_reference_pulse=true`；
- `late_actor_rescue=true`；
- `sampler_only=true`, `training_allowed=false`。

因此 124/128 属于 composite control policy。公平报告应并列 raw actor、actor+reference route、actor+route+guards 三组，而不能把最强组合归给 actor 权重。

## 6. 实验产物核算

### 6.1 训练预算

| window | episode | chunk | final | success-once | hit-drop |
|---:|---:|---:|---:|---:|---:|
| iter1 | 16 | 478 | 15/16 | 16/16 | 1 |
| iter2 | 16 | 499 | 8/16 | 15/16 | 7 |
| iter3 | 16 | 454 | 11/16 | 16/16 | 5 |
| iter4 | 16 | 450 | 10/16 | 16/16 | 6 |
| iter5 | 16 | 468 | 10/16 | 16/16 | 6 |
| iter6 | 16 | 505 | 8/16 | 15/16 | 7 |
| iter7 | 16 | 537 | 11/16 | 16/16 | 5 |
| **合计** | **112** | **3391** | **73/112** | **110/112** | **37** |

训练日志的 reward mean 为 0.8257–0.9979，直接反证“稀疏”。每个窗口记录 `training_allowed=true`，eval 四个 shard 全为 false。

### 6.2 headline 结果

- 基线 artifact：`episodes=128`, `success_once=126`, `final_success=96`, `hit_then_drop=30`, `use_actor=false`, `reference_policy=pi05`, `elapsed=1053.421s`。
- 候选 artifact：四 shard final `32/32, 30/32, 30/32, 32/32`，合计 `124/128=96.875%`；success-once `126/128`，hit-drop `2/128`。
- 候选总 eval transitions：`942+1007+972+934=3855`。
- 该结果与本地总结 `stackone_last_r1_frombase_iter_cost_curve_20260519.md` 和 `xvla_112_success_codepath_audit_20260529.md` 一致。
- 后续继续训练到 144/176/208/240/272 episode 的 standalone final 分别为 122/118/122/118/120，未单调提升；112 episode 是该 from-base only-LaST-R1 链的局部最佳，不是稳定样本曲线终点。

## 7. 可直接复用与不应直接搬运

### 7.1 优先从上游复用

- `rlinf/algorithms/rlt/{transition,route,rollout}.py`：统一 transition/route 语义，避免本地脚本继续复制采样逻辑。
- `huggingface_worker.py` 的 frozen feature-model 装载与 route 接入。
- `RLTTokenTransformer` 及其 unit test：如果简历要严格声称“RL Token”，应真的接入该模块并提供 Stage 1 checkpoint，而不是只把 prefix mean 命名为 token。
- `rlt_td3_mlp_policy.py`、TD3 worker、配置和 e2e config：若选择官方 off-policy `-Q+BC` 路线，直接以此为基线。

### 7.2 可从本地成功链抽取

- `openpi_pi05_rlt_td3_official_pass_copy.py`：frozen OpenPI 同源 reference/feature wrapper、reference passthrough actor 和精确参数结构。
- helper 中 strict Stack-One 成功判定、episode/chunk lineage、失败尾部标注与指标；应拆成独立 reward/collector 模块，不继续维护 3,282 行单文件。
- `trajectory_lapo_v2_terminal_guard.py`：trajectory padding、GAE/PPO/BC 框架骨架；必须先修正第 5 节的 value/logprob/mask 问题。
- 112-success launch lineage、config、metrics：可作为回归 fixture 和 golden metadata，不作为真机测试。
- `scripts/stackone_strict_success.py`：仿真 strict-success 口径可复用，但不能代替真实机械臂成功判定。

### 7.3 不直接搬运

- bridge 下完整 vendored `transformers/`、LaST-R1 无关模块和 `external_refs/` 镜像；应以依赖/固定 commit 管理。
- 大量 dated `*_copy.py`、backup/malformed 文件；只保留被 launcher/command 实际引用的 copy-on-write lineage。
- checkpoint、replay、9.6MB step trace；它们是产物，不是源码依赖。
- v2d 三种 eval-time guard 不应默认混入“模型成功率”实现；应成为可开关 safety/controller 层并单独报告。

## 8. 最可信的重实现方案

如果目标是最大限度保留简历的 PPO/GAE 叙事，同时让算法自洽，建议新实现名为 `ReferenceAnchoredChunkPPO`，不要冒充官方 RLT TD3：

1. 使用上游 RLinf 的 frozen feature-model/route/transition 结构；pi0.5 forward 输出 `z_rl` 与 reference chunk。
2. 明确 feature 选择：要么接入并训练/加载官方 `RLTTokenTransformer`，要么诚实命名 `frozen_pi05_prefix_mean`。
3. 保留 611,107 参数 passthrough actor；增加独立标量 `V(s)`（或双 V ensemble）供 GAE。Twin-Q 可保留为 failure-risk/TD auxiliary，但不要把 `Q(s,a)` 当 V。
4. rollout 时保存 actor mean/std/sample/log-prob、route source、实际执行动作和任何 guard rewrite。只有真正由 actor 分布采样且未被修改的 chunk进入 PPO；reference/guard chunk只进 BC/critic，或显式建模 mixture-policy likelihood。
5. `ppo_epochs=1` 可保留；日志同时报告 optimizer-step 数，避免“单轮=单步”歧义。
6. reward 若保持当前规则，应命名 `dense_binary_regression_guard` 并把全部阈值配置化。若坚持“稀疏”，只在严格成功/失败事件上发稀疏信号，并单独说明 strong-negative advantage override。
7. 保留 no-hit 整轨、hit-drop 尾段标签；显式单测 chunk 边界、drop/hold/first-hit 对齐和 advantage floor。
8. 评测至少输出 raw actor、reference-routed actor、routed+guards 三栏，并使用相同 reset manifest/seed 与 paired baseline。
9. 真机接入前另外实现 SO101/Nova/ROS2/camera adapter、动作空间/时间同步和 e-stop；当前结果只能标为 IsaacLab simulation。

若目标反而是“严格跟上游官方 RLT”，则删除 PPO/GAE 叙事，直接采用 upstream TD3 replay + Twin-Q + `-Q+BC` worker，并用上游 configs/tests 做基准。这两条路线不应在命名上混为一条。

### 8.1 当前 fork 已落实的算法闭环（非历史产物证据）

审计后已在 `project/rlinf/projects/fibocom_vla/rl/` 按上述可信方案实现修复；这部分是当前 fork 的代码状态，不反向证明 E 盘历史实验采用了同样算法：

- `trajectory.py:28-166` 定义稳定的 `ActionSource` 枚举和 `policy_mask`。`REFERENCE/GUARD/CLAMP/RESCUE/PADDING` 以及未采样的 deterministic policy action 均不得拥有 PPO behavior likelihood；其 `old_log_probs` 必须为 NaN，只有真实 stochastic `POLICY` 位必须为有限值。
- `torch_modules.py:140-174,227-280,334-370` 从 reference-conditioned bounded tanh-Normal 直接采样残差并同时产生 Jacobian-corrected chunk log-prob；越出该分布 support 的执行改写动作拒绝计算密度。deterministic mode 显式 `policy_mask=False`。新增独立 `StateValueCritic(features, reference)`，不读取执行动作或 actor mode。
- `trainer.py:117-144,153-312` 用独立 `V(s)` 产生 trajectory GAE；advantage 只在 `policy_mask` 上归一化。更新前会以当前 behavior actor 复算已采样 bounded residual 的 log-prob，不一致时在任何 optimizer step 前失败。PPO ratio、reference-anchor actor 梯度和 auxiliary `-min Q` actor 梯度均只读取 policy transitions；Twin-Q 与 V 可读取全部 valid executed transitions。全非 policy batch 不执行 actor optimizer。
- 修复了原实现中 anchor loss 对录制 residual 常量求值、因而无法给 actor 梯度的问题；当前 anchor 约束 actor 的实时 bounded mode residual。
- `test_residual_trainer.py:221-381` 覆盖四类 execution intervention 不进入 PPO、全非 policy batch actor 参数不变、伪造 old log-prob 在更新前被拒绝、GAE 对 Twin-Q 任意扰动保持不变及 provenance 契约。2026-08-25 最终定点复核在可用 PyTorch/pytest 环境执行 Fibocom 单测为 **39 passed**；环境缺 `omegaconf`，测试进程仅注入无副作用 resolver stub，未安装或修改依赖。静态 `compileall` 通过。

仍需在真实采集器接入时保证每次 controller rewrite 同步写入实际 `action_source`；当前合同会阻止错误数据进入 PPO，但不会自行推断外部 SDK 做过的动作改写。

## 9. 测试、验证与证据边界

- 上游纳入 `test_rlt_token_transformer.py` 和两份 e2e config；它们验证模块 shape/config 路径，不证明本地 112 链。
- E 盘历史成功 bridge 没有独立 unit test：无 actor/critic 参数计数测试、reward tail 表驱动测试、真实 behavior log-prob 测试、route-mask 测试或 paired eval test。当前 fork 已补齐其中参数计数、behavior log-prob 和 route-mask 测试，但这不改变历史链的证据缺口。
- 本地 `scripts/verify_pi05_official_rlt_alignment.py` 是 388 行文本锚点检查；本次只读执行结果为 `40/40 passed`。它核验的是另一条 VLARLKit-style TD3 alignment lineage，不是 112-success PPO bridge 的语义正确性。
- 本报告只认证：固定源身份、目标范围的逐行源码/配置/日志审阅、静态算法推导与既有产物交叉核算。未认证依赖可安装、GPU/Isaac runtime 可复跑、checkpoint 可加载、paper-level reproduction、跨 seed 稳定性或真实机器人表现。

## 10. 排除项

排除项及理由也列在 CSV 末尾：

- vendored `transformers/`、LaST-R1/RL-Token/VLARLKit/FlashRT 镜像：第三方依赖，不是第一方定制；仅审阅成功链实际调用的 LaST-R1 GAE/PPO 函数范围。
- `.git/`、venv、cache、build、assets：版本元数据或生成依赖。
- `.pt/.pth/.ckpt/.safetensors`、replay buffer：二进制产物；本审计使用其旁路 config/log/metrics，不反序列化不可信 pickle。
- `chunk_reward_trace.jsonl`（约 9.6MB）与 episode trace：逐 step bulk trace；四 shard summary/metrics、config、command 和相关 reward 源码已覆盖 headline 核算。
- 无关 result、dated backup、malformed 和失败分支：不在原始 base→iter7→v2d 的 112-success 可达链；关键对照/成本总结已纳入。
- RLinf 其余算法、环境、通用测试与 docs：不含 RLT/π0.5/Stack-One/actor/critic/reward/rollout 目标逻辑。
