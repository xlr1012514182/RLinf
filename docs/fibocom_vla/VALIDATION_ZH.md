# 验证方法

[English](VALIDATION.md) · [实现指南](../../examples/embodiment/fibocom_vla/README_ZH.md)

本页说明如何执行检查及各项检查的范围，不是历史运行报告。命令均在仓库根目录执行；需要保留的输出放入独立、受控的运行归档。

## CPU 组件

完成 CPU 环境安装后执行：

```bash
for config in examples/embodiment/fibocom_vla/config/*.json; do
  .venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli validate-config \
    --config "$config" || exit 1
done

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli actor-report \
  --config examples/embodiment/fibocom_vla/config/robotwin_pi05_h50_dry_run.json

.venv-fibocom/bin/python -m rlinf.projects.fibocom_vla.cli mock-smoke \
  --config examples/embodiment/fibocom_vla/config/mock.json --steps 16

.venv-fibocom/bin/python -m pytest -q -rs tests/unit_tests/projects/fibocom_vla
```

配置验证不连接机器人；`mock-smoke` 只使用 Mock 后端。[单元测试](../../tests/unit_tests/projects/fibocom_vla)覆盖资产契约、残差更新、Draft 血缘、数值参考、规划生命周期、新鲜度、适配器及运动门控。应单独检查跳过项，跳过不算 GPU 或硬件检查通过。

## 需要外部条件的检查

| 检查 | 前置条件 | 输出含义 |
|---|---|---|
| `verify-checkpoint-assets` | 完整 checkpoint 下载与清单 | 文件身份、变换和模型/栈契约；不执行前向 |
| `openpi-checkpoint-smoke` | 已验证权重与兼容 OpenPI/GPU 环境 | 模型加载及一次合成前向；不代表任务成功或延迟成绩 |
| 使用 H50 Mock 模板的 `run` | 同上 | 模型/规划器/Mock 集成；不代表物理部署 |
| 合成 CUDA Graph 基准 | CUDA 主机与支持的 PyTorch | 合成组件计时/一致性，不是 π0.5 端到端计时 |
| 模型加速对比 | 实际模型输入、符合条件的配置、配对 eager/优化运行 | 仅覆盖明确测量的计时边界及数值指标 |
| 任务评测 | 任务环境/数据、精确策略资产与评测程序 | 仅覆盖声明的任务、episode、种子及成功定义 |
| 真机部署检查 | 标定硬件、已审查适配器及操作者 | 仅覆盖实际监督执行的机器人配置 |

资产和模型命令见实现指南。完整任务评测与硬件标定需要任务专用程序和协议，不能用 Mock CLI 替代。

## 合成 CUDA Graph 基准

在 CUDA 环境执行：

```bash
PYTHONPATH=. python examples/embodiment/fibocom_vla/benchmark_cuda_graph.py \
  --output artifacts/cuda_graph_component.json \
  --batch-size 8 --width 256 --depth 4 --dtype float16 \
  --capture-warmup 3 --warmup 5 --repetitions 50
```

保留输出中的 synthetic 标记。程序测量自带的合成网络，不是下载的 π0.5 checkpoint。实际模型对比应记录输入身份、模型/引擎哈希、精度、设备/驱动、图捕获条件、预热/捕获计时方式、重复次数、同步方法，以及计时是否包含预处理、传输、排队或控制。使用相同输入和数值容差，报告分布统计而不只报告最快一次。

## 任务评测记录

在评测前确定协议：

1. 固定任务/模拟器/机器人版本、数据划分、episode 列表、种子与策略 checkpoint 哈希。
2. 定义成功、失败、超时、重置条件及安全停止，不根据有利结果挑选 episode。
3. 冻结基座与适配策略使用可比任务条件，训练/筛选数据与留出评测数据分离。
4. 保存逐 episode 结果、分母、失败运行和不确定性；必要时保存可核对标签的视频/轨迹引用。
5. 残差学习绑定 rollout 行为 checkpoint 与配置身份；Draft 部署保留被评测候选、协议摘要及独立签名审批。

源码包不提供这些成绩。代码检查、合成冒烟和候选导出应分别按其实际范围描述。

## 源码分发

运行 manifest、依赖锁、测试、基准程序、许可和硬件安全检查保留在源码树中。下载权重、生成的 checkpoint/引擎、任务数据、标定秘密、日志和运行报告放在公开源码包之外。根目录 `artifacts/`、`runs/`、`checkpoints/` 及测试缓存默认被忽略；`.gitignore` 不会移除已跟踪文件，也不能代替秘密扫描。
