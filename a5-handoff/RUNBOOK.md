# A5 执行规则

本文件是项目本地 `A5-INTERNAL-AGENT-RUNBOOK.md` 的内外交接版。具体可执行命令以当前 CURRENT.md 为准，本文件不授权连续执行所有步骤。

## 成功定义

以包含 TorchTitan PR #4025 的主线固定提交，使用 Kimi-K3 原生 FlexAttention 打通 debugmodel 训练链路：单卡 2 step，然后四卡 FSDP2 2 step。loss 与 grad norm 有限，所有 rank 正常退出，无 OOM、HCCL、storage 或算子错误。这不是完整规模模型验收。

## 门禁顺序

| 门禁 | 内容 | 放行证据 |
| --- | --- | --- |
| A5-0 | 冻结实际运行环境、版本与代码身份 | SoC、驱动、固件、CANN、OS/架构、Python、torch、torch_npu、Triton-Ascend、设备健康/占用、固定 SHA、安装前依赖快照 |
| A5-1 | NPU tensor -> world-size 1 HCCL -> 原生 Flex 微测 | 依次成功；HCCL init/all-reduce/destroy；Flex forward/backward 与有限梯度 |
| A5-2 | 固定 TorchTitan 主线 checkout 与代码测试 | 确认包含 PR #4025、kimi_k3_debugmodel、默认 Flex；现有相关 CPU/unit 测试结果 |
| A5-3 | 单卡 pristine 2 step | 原生入口、未经兼容修改；两步有限 loss/grad norm、退出码 0、正常 teardown |
| A5-4 | 四卡 FSDP2 pristine 2 step | 所有 rank 完成并退出；必要时单独记录最小 compat 重试 |
| A5-5 | 5-10 step 无 profiler 稳定性验证 | 冻结首个 A5 baseline，环境、代码与 workload 可追溯 |
| A5-6 | 后续可观测性 | 经单独任务授权后再做 profiler、memory snapshot |

每次仅执行 CURRENT.md 的一个任务。前项失败不得进入后项。

## 依赖部署

- 优先用 A5 内部已验证的完整版本组；torch 和 torch_npu 必须配套发布或同一内部构建批次，Triton-Ascend 使用该方案指定版本。
- 版本关系不明确就停止，向用户索取配套依据，不单包追新。
- 安装必须另行授权并使用新 venv，不修改系统 Python，不删除旧环境。安装前后保存依赖快照与 diff，记录安装 wheel 的 SHA256；原始快照留在内网并检查敏感地址。
- 910B3 的 torch 2.12 / torch_npu 2.12 / Triton-Ascend 3.6 探索组合不是 A5 推荐矩阵。

## 代码与兼容

- 使用独立 clone/worktree，记录完整 SHA 和工作区状态，不以移动的 main 作为报告身份。
- 从当前 checkout 的 README、帮助和现有测试确定 CLI/测试路径，不凭旧命令猜参数。
- 首先使用原生入口和默认 Flex。不预打 SDPA、DeferredLine shim、set_timeout no-op 或 FSDP 补丁。
- 单卡成功后先跑四卡 pristine。只有真实复现 final norm/lm_head 共享 FSDP state、unallocated storage 或相关两阶段 backward 错误，排除 Flex、HCCL、OOM 且确认当前主线仍缺修复后，才评估最小 FSDP 分离补丁。
- compat 必须新分支/worktree、先单测再同配置重试，不带 SDPA 改动。分别记录 pristine FAIL 与 compat PASS/FAIL。

## 运行与错误

- 每次 NPU 操作前确认授权卡组、健康、占用和所需端口；不动他人进程或共享环境。
- 每次使用唯一 run ID 和全新目录；保留命令、配置、退出码、日志、版本、SHA 与 patch 身份，不覆盖失败产物。
- Flex 失败保留首个 import/API、compile/codegen、runtime/operator 或数值错误；NoTritonConfigsError 可能只是汇总，必须找首个编译器错误。
- 四卡检查所有 rank，不仅 rank 0；检查两步数值、正常 teardown 和残留进程，不自行清理。
- 短 smoke 优先前台并设置合理超时；长任务需另行下发带日志、PID/状态和校验的启动方案。

## 禁止事项

- 不用 SDPA compat 宣称原生 Flex 成功，不用 910B3 结果替代 A5。
- 不同时改多个依赖再直接归因，不在 pristine 未复现时预打兼容补丁。
- 首轮不叠加 profiler、snapshot、FullAC、swap、HSDP 或 EP，不做性能调优。
- 不提交内网原始日志、凭据、内部地址、数据或权重；只返回脱敏结果。
