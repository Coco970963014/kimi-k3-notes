# 一键环境检查

`a5_doctor.py` 是唯一需要转入内网的脚本，只依赖 Python 标准库启动。请在同事容器内、准备用于训练的 Python 环境执行；不在宿主机执行 Docker 操作。

## 先确认空闲卡

先用 `npu-smi info` 并向同事确认可用卡。脚本不自动预留设备，不根据显存数字猜测设备是否空闲。

以下 `0` 指当前 Python 进程可见设备中的逻辑编号，不保证等于宿主机 npu-smi 的物理编号。容器设备映射、ASCEND_RT_VISIBLE_DEVICES 已设置时，应先确认对应关系；脚本不会覆盖这些设置。

## 一键执行

已确认逻辑 0 卡获准使用、空闲且本次测试期间独占后，在脚本所在目录执行：

```bash
python a5_doctor.py --compute --device 0 --confirm-idle
```

有 TorchTitan checkout 时可以补充 `--repo /实际路径/torchtitan`，只读取 HEAD、v0.3.0 tag SHA 和工作区状态，不 fetch、checkout 或改代码。缺少本地 tag 会明确失败，不凭目录名推定基线。

不能确认设备授权/空闲时，只执行信息与导入检查，不分配测试 tensor：

```bash
python a5_doctor.py
```

导入 torch_npu 本身可能加载驱动动态库；此模式不启动 tensor/HCCL/Flex 测试。

## 检查范围

| 阶段 | 验证内容 |
| --- | --- |
| inventory | Python、CPU 架构、包发行版本、已知位置的 CANN/驱动/固件版本文件、设备可见性；不读取凭据或 pip 配置 |
| npu_smi_before | 记录容器设备健康与占用，详细表由用户在内网查看 |
| imports | torch、torch_npu、triton 实际导入与文件位置；FlexAttention、FSDP2 API 是否可导入 |
| tensor | FP32 基础结果校验、BF16 小矩阵乘前反向、输出和梯度有限 |
| hccl | world-size 1 的 init/all-reduce/destroy，动态绑定本机回环端口 |
| flex | 原生 FlexAttention，BF16、causal mask、shape=(1,2,128,64)，Inductor fullgraph 编译、前反向与有限梯度 |
| repo（可选） | HEAD 是否等于本地 v0.3.0 tag、工作区是否干净；不导入或执行模型仓库代码 |
| npu_smi_after | 记录运行后占用；残留进程是否正常需人工核对，不自动清理 |

基础诊断会尽量收集完整；计算阶段按 tensor -> HCCL -> Flex 顺序，前项失败则跳过后项。torch 版本偏离 2.14.0、Python 不在 3.11/3.12 或同时存在两种 Triton 发行包会警告，但允许诊断现有组件能力，不擅自修复。

单项默认超时 180 秒，Flex 首编译最多 600 秒；可用 `--timeout` / `--flex-timeout` 调整。超时只终止脚本自己启动的进程组，保留日志，不碰其他用户进程。超时后需人工确认设备状态，再决定重试。

## 结果与口述

每次在 `/tmp/a5-doctor/` 创建独立目录，内含 `SUMMARY.txt`、`report.json`、各阶段日志以及编译缓存/调试目录。需要长期保留时用 `--output-root /个人持久化目录` 指定位置。原始输出可能含内网路径等信息，必须留在内网，不提交本仓。

你只需口述：

1. torch、torch-npu、triton、triton-ascend 的版本。
2. 最后的 OVERALL、tensor/hccl/flex 的 PASS/FAIL/SKIP。
3. 若失败，FIRST_FAILURE 及对应日志中的首个错误关键词；不用传文件或抄整段日志。

`triton=NOT_INSTALLED` 指标准发行包不存在，不等于 `import triton` 失败；同时看 `loaded_triton` 和 imports 阶段，Triton-Ascend 也可提供该模块。

退出码：0 为请求的组件检查通过且无自动警告；1 为至少一项失败；2 为仅采集模式或存在警告。参数错误也由 argparse 返回 2，需看终端是否实际生成 SUMMARY.txt。没有摘要或中断不得当作成功。

`PASS_COMPONENT_SMOKE` 仅证明本次小型组件测试通过，不等于配套版本认证或模型跑通。脚本不做四卡通信/FSDP2、不启动 Kimi-K3 训练、不加载数据/权重、不接 profiler，不验证全模型数值对齐或已安装组件是否带同事的私有补丁。后续仍需固定代码、测试单测与单卡/四卡训练。

## 本地自测（无 NPU）

```bash
python -B -m unittest discover -s a5-handoff -p 'test_a5_doctor.py' -v
```

本地自测只验证编排、退出码、失败跳过和超时，不替代 A5 上的真实组件测试。
