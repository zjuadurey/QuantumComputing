# PROJECT_INTENT.md

This document records the project's high-level research motivation and
longer-term direction. It is intentionally broader than the current
implementation, so it may lag behind the latest frozen design.
For current semantics and execution details, prefer
`docs/v04_fullcontract_spec.md` and `docs/v04_guide.md`.

Minimal context-recovery order across machines:
1. `docs/v04_fullcontract_spec.md`
2. `docs/v04_guide.md`
3. `docs/research_log.md`
4. This file

## 项目名称

**openqasm-rtir**

## 这个项目要解决什么问题

这个项目不是要做一个“大而全的量子编译器”，而是要做一个**小型、受控范围的可验证编译原型**，用最小闭环回答下面这个核心研究问题：

> **带 timing、pulse、measurement-feedback 的量子程序，如何被精确定义、可靠 lowering，并被证明在关键语义与安全性质上是正确的？**

更具体地说，我们关心的不是单纯“电路等价”，而是：

- 这类实时量子控制程序的**形式化语义**是什么；
- 如何把它们 lowering 到一个显式表达**时间、资源、依赖、反馈**的 IR；
- 如何证明 lowering 保持关键性质，而不只是“功能大概没变”；
- 如何通过回归测试和自动检查发现规范—实现不一致、资源冲突或反馈因果错误。

这个方向的研究动机是：OpenQASM 3 已经把控制流、时序、脉冲与校准统一放进同一语言框架，因此形式化语义与可验证编译不再只针对门级电路等价，而必须覆盖**时序、资源和反馈正确性**。同时，这个方向对小课题组友好，因为它主要依赖形式化工具、测试基准和仿真，而不是昂贵硬件。 

## 研究定位

这是课题 A 的工程化研究原型：

**OpenQASM 3 / OpenPulse 的时序—反馈形式化语义与可验证编译**

其核心目标是构建一条：

**从实时量子程序到可执行控制序列的可验证编译链**。

覆盖的重点包括：

- OpenQASM 3 的 timing / control-flow 子集；
- OpenPulse 风格的 ports / frames / waveforms / play / capture；
- 动态电路中的 measurement -> classical if -> feedback 语义；
- 资源安全、时序一致性、反馈一致性等性质。

## 结构性空白（为什么值得做）

现有已验证量子编译工作大多集中在：

- 门级电路语义；
- 低层量子 IR 的语义保持优化；
- pass 级正确性验证；

但对 **OpenQASM 3 / OpenPulse / 动态电路** 这一层，仍缺少一个统一、可机检的：

- 实时语义；
- 时序/资源/反馈感知 IR；
- 端到端可验证 lowering/编译链。

也就是说，本项目要填补的不是“再做一个编译器”，而是：

> **把验证对象从“门级电路等价”推进到“时序/反馈/寄存器/校准”的统一语义层。**

## 论文目标与投稿定位

本项目的目标不是一般工程实现，而是以**CCF-A 口味**的研究原型为目标。

优先投稿方向：

- **POPL / PLDI**：如果强调编程语言、形式化语义、证明与 certified transpilation；
- **ICSE**：如果强调软件工程、可靠性、验证方法学、bug-finding；
- **ASPLOS**：如果后续更强调实时控制语义与体系结构/系统约束的结合。

当前优先定位：

> **POPL / PLDI / ICSE 方向的主稿**

更具体的项目叙事可以对齐如下论文风格：

- Verified Real-time Quantum Control Compilation
- Timing- and Feedback-Safe Quantum Compilation

## 预期核心贡献

论文和系统原型的贡献建议围绕三段式展开：

### 1. 形式化语义
给出一个受控子集的 OpenQASM 3 + OpenPulse 形式化语义，覆盖至少：

- delay / duration / timing intent；
- measurement 与 classical if；
- 后续可扩展到 play / capture / frame / port 资源语义。

### 2. 可验证编译链
设计一个显式表达时间、资源和反馈边的 RT-IR（或 RT-SQIR 风格 IR），并实现：

- OpenQASM 3 子集 -> RT-IR 的 lowering；
- 关键语义/约束保持的证明或自动验证。

### 3. 回归与安全属性验证
用 regression suite 和自动检查展示：

- 资源冲突自由；
- 寄存器/经典位访问合法；
- timing / alignment / delay intent 保持；
- feedback 因果顺序正确；
- 能发现实现与规范不一致的问题。

## MVP（当前这阶段最小要做成什么）

当前阶段不要追求完整 OpenQASM 3 编译器，而要完成一个**MVP v0**：

### 受控子集
先只支持一个极小子集，例如：

- `h q[i];`
- `x q[i];`
- `delay[Ndt] q[i];`
- `c[i] = measure q[j];`
- `if (c[i] == 1) { x q[j]; }`
- `if (c[i] == 1) { h q[j]; }`

### 一个 toy real-time IR
每个事件至少显式包含：

- `event_id`
- `kind`
- `start`
- `duration`
- `resource`
- `qubit`
- `creg`
- `condition`
- `payload`
- `depends_on`

### 一条最小 lowering 链
打通：

**OpenQASM 3 小样例 -> toy RT-IR -> timeline -> 检查器**

### 两类最基础检查
先做最小但有研究意义的检查：

1. **资源冲突检查**
   - 同一 resource 上不能有重叠事件。

2. **反馈因果检查**
   - branch 必须晚于其依赖的 measure 结束；
   - classical bit/readiness 必须满足因果顺序。

## 当前不做什么

为了避免 scope creep，当前版本明确**不做**：

- 不做完整 OpenQASM 3 支持；
- 不做完整 OpenPulse 物理精确语义；
- 不做工业级优化器；
- 不做大规模真实硬件验证；
- 不以噪声最优或系统延迟最优为首要目标。

这些可以放到后续版本，但当前项目的核心是：

> **语义、lowering、可验证性。**

## 方法学路线

建议走“先小后大”的路线：

### 第一阶段：快速原型
- Python + Qiskit + `qiskit_qasm3_import`
- 手写 toy lowering
- timeline / conflict / causality 检查
- pytest 回归测试

### 第二阶段：自动验证
- 引入 Z3，把 start/end/depends_on 等关系写成约束；
- 做 timing / causality / conflict 的可满足性检查；
- 把“正确性”从代码逻辑提升为显式约束验证。

### 第三阶段：更强形式化
根据资源选择其一：

- **Coq 路线**：借鉴 QWIRE / SQIR / VOQC；
- **SMT / 契约路线**：借鉴 CertiQ。

## 与已有工作的关系

本项目的相关工作锚点主要是：

- **VOQC / SQIR**：已验证 IR / 优化证明；
- **CertiQ**：面向真实 pass 的契约 + SMT 验证；
- **OpenQASM 3 / OpenPulse**：真实语言与 timing / pulse / calibration 语义来源；
- **qiskit_qasm3_import**：现实工具链入口。

本项目与这些工作的差异不是“也做一个 compiler”，而是：

> **把验证维度扩展到 timing、resource、feedback 与实时控制语义。**

## 对 Claude Code 的实现要求

你在实现代码时，请始终记住：

1. **这是研究原型，不是产品。**
2. **优先保证最小闭环可运行。**
3. **优先保持语义清晰，不要过度设计。**
4. **每一版都要能跑 demo 和测试。**
5. **后续扩展时，优先把当前 toy lowering 升级为 AST 驱动 lowering。**
6. **下一步最值得做的是：Z3 约束化 + AST lowering + 更清晰的 feedback 语义。**

## 当前开发目标（请优先完成）

当前请先完成：

- 一个小型项目 `openqasm-rtir`
- 一个受控子集的 OpenQASM 3 输入
- 一个 toy RT-IR
- 一个 lowering 模块
- 一个 timeline 输出器
- 两个检查器：`no_conflict` 与 `causality`
- 两个 example
- 一套 pytest 测试

完成标准不是“架构看起来很高级”，而是：

> **代码清晰、可跑、可测，并且清楚地体现这个研究问题是什么。**

## 后续迭代建议

如果 v0 跑通，后续按这个顺序升级：

1. **AST 驱动 lowering**
   - 不再依赖文本正则，而是基于 OpenQASM AST / importer 结果。

2. **Z3 约束检查**
   - 把时序、依赖、无冲突等条件变成 SMT 约束。

3. **更明确的 timing / feedback 语义**
   - 把 classical readiness、resource availability、backend timing context 说清楚。

4. **引入 toy pulse/play/capture**
   - 让项目更贴近 OpenPulse。

5. **回归测试集扩展**
   - 支持更多 timing / defcal / openpulse 样例。

## 一句话总结

这个项目的目的不是“做一个量子编译器”，而是：

> **做一个最小可验证编译原型，研究带 timing、pulse 和 feedback 的量子程序，如何被精确定义、可靠 lowering，并在关键时序/资源/反馈性质上被证明正确。**
