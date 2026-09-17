# A5 内外交接

本目录用于外网教师 agent 与内网 A5 执行 agent 之间传递任务和脱敏结果，用户负责转交。不是自动同步或远程执行系统，不要求 A5 能访问 GitHub。

## 入口

- [CURRENT.md](CURRENT.md)：唯一有效的当前任务，由教师 agent 更新。
- [RUNBOOK.md](RUNBOOK.md)：目标、验收顺序与禁止事项。
- [RESULT-TEMPLATE.md](RESULT-TEMPLATE.md)：内网 agent 回传格式。
- `results/<task-id>-<run-id>.md`：后续收到并脱敏的实际结果；尚无回传时不创建占位成功报告。

## 交接方式

1. 教师 agent 发布当前任务并告知文档 commit SHA。
2. 用户将该版本的 CURRENT.md、RUNBOOK.md、RESULT-TEMPLATE.md 转入内网。
3. 内网 agent 核对任务 ID，只执行 CURRENT.md 的一个步骤；不执行整个 RUNBOOK。
4. 内网 agent 保留本地完整日志，按模板生成脱敏摘要，由用户传回。不得自行 push、发布日志或申请外网权限。
5. 教师 agent 核对结果，保存到 results/，更新当前状态；未通过时只下发本步骤的修正任务，不跳关。

给内网 agent 的固定提示：

```text
阅读 a5-handoff/CURRENT.md 和 RUNBOOK.md。只执行 CURRENT.md 指定的任务，不自行安装、升级、打补丁或进入后续步骤。按 RESULT-TEMPLATE.md 回传脱敏结果；无法执行时说明阻塞，不猜测成功。执行结束立即停止，等待下一份任务。
```

## 信息边界

- 只提交任务、版本号、代码 SHA、脱敏诊断与验收结论。
- 不提交 token、密码、私钥、内部镜像/包源地址、内部 IP、主机名、账号或敏感业务数据。
- 原始日志、环境变量全集、pip 配置、训练数据、权重和 profiler 产物留在内网。错误片段与路径返回前也必须脱敏。
- 机器用稳定别名，例如 `A5-01`；敏感路径用 `<A5_ROOT>` 替代，内网保留映射。
- 本仓不放模型代码或补丁；代码仍通过 TorchTitan 仓库的可审查 commit/patch 传递。

## 当前证据

截至本目录初始化，尚未收到 A5 环境实测结果。此前 910B3 的 SDPA/compat 实验不计入 A5 原生 FlexAttention 验收。
