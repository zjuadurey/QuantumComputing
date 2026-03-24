# Qiskit 到脉冲层问题梳理

## 需要明确的问题

1. `Qiskit Target` 会不会自适应？
2. 从 Qiskit 开始，到脉冲语言，每一层是什么？
3. Qiskit 真的完全正确吗？会不会有“Qiskit 通过，但实际上无法运行”的代码？
4. Qiskit 等门级电路如何转换到你们的 IR？

下面整理的是刚才对这四个问题的回答原意。

## 1. `Qiskit Target` 会不会自适应？

会“随 backend 信息更新而变化”，但它不是那种会自己推理、自己修程序的自适应。

更准确地说：

- `Target` 是 backend 的机器能力与约束描述。
- Qiskit transpiler 会根据这个 `Target` 去适配电路。
- backend 的配置和性质会随校准更新，`backend.refresh()` 也会刷新当前 backend 的 `target`。
- IBM 文档还区分了静态 backend 信息和动态 backend 信息；动态属性通常会在大约每 24 小时校准后更新。

所以：

- `Target` 本身是“约束数据”
- “自适应”的是编译流程根据 `Target` 做 layout/routing/translation/scheduling
- 它不是一个会自动修复所有问题的智能层

相关文档：
- `Target` 说明：https://qiskit.qotlabs.org/api/qiskit/qiskit.transpiler.Target
- `IBMBackend.refresh()` / `target`：https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/ibm-backend
- backend 动态信息：https://docs.quantum.ibm.com/guides/get-qpu-information

## 2. 从 Qiskit 开始，到脉冲语言，每一层是什么？

按现在官方文档，比较稳妥的分层是：

1. 用户层  
   `QuantumCircuit` 或 OpenQASM 3 风格程序

2. Qiskit transpiler 内部 IR  
   `DAGCircuit`  
   官方明确说 transpiler 栈内部使用的是图式 IR，不是直接拿 `QuantumCircuit` 改来改去。

3. 面向具体 backend 的 ISA circuit  
   经过 `layout -> routing -> translation -> optimization -> scheduling` 之后，得到满足 backend `Target` 约束的电路

4. 更低层的 timing / schedule 信息  
   scheduling stage 会显式考虑 idle time 和 `delay`

5. 历史上的 pulse/program layer  
   以前是 `ScheduleBlock` / `qiskit.pulse` 这套  
   但现在官方已经写得很明确了：
   - IBM 处理器上的 pulse-level control 已移除
   - `qiskit.pulse` 在 Qiskit 2.0 已移除
   - 如果要做 pulse simulation，官方建议看 Qiskit Dynamics

而且 IBM 文档还明确说过一句很关键的话：  
IBM 设备“不接受完整的 pulse program”，pulse program 只是用来描述 gate 的子程序。

所以，**从现在的 Qiskit/IBM 官方栈看，并不存在一个仍然主流、官方支持的“门级电路 -> 真机可执行完整 pulse 程序”公开工作流。**

相关文档：
- transpiler 概览 / `DAGCircuit`：https://qiskit.qotlabs.org/docs/api/qiskit/transpiler
- transpiler stages：https://qiskit.qotlabs.org/docs/guides/transpiler-stages
- DAG guide：https://qiskit.qotlabs.org/docs/guides/DAG-representation
- pulse docs / removal notice：https://quantum.cloud.ibm.com/docs/en/guides/pulse

## 3. Qiskit 真的完全正确吗？会不会有“Qiskit 通过，但实际上无法运行”的代码？

会有，而且这是官方承认的现实问题。

最直接的官方证据是 IBM 对 `non-ISA circuits` 的说明：  
从 Runtime 的要求看，**一个“合法的 Qiskit 电路”并不等于“对某个 backend 可执行”**。如果电路不符合 backend 的 `Target`，Runtime 会直接拒绝。

IBM 官方博客明确说：
- 所有提交到 backend 的 circuits 都必须符合该 backend 的 `Target`
- 这些约束包括 native basis gates、connectivity，以及在相关情况下的 pulse / timing 规格
- 引入这个要求就是为了过滤掉设备根本无法执行的作业

所以至少有两层“合法”：

- Python/Qiskit API 语法上合法
- backend ISA / timing / resource 约束上合法

前者不推出后者。

另外，在 pulse 层这个 gap 更大：  
即使你以前能在 Qiskit 里构造出 pulse schedule 对象，也不代表今天还能把它直接发到 IBM 真机执行。官方文档现在直接说 IBM QPU 的 pulse-level control 已移除。

所以你们工作的价值可以建立在一个真实存在的 gap 上，但要说准：

不是“Qiskit 全错了”，而是：

- 源程序层面的合法性，不等于硬件层面的可执行性
- 即便门级编译通过，也仍然需要对更低层的 timing/resource/feedback obligations 做验证
- 这正是你们 contract 的切入点

相关文档：
- ISA circuits 说明：https://www.ibm.com/quantum/blog/isa-circuits
- IBM error catalog：https://quantum.cloud.ibm.com/docs/errors
- pulse removal：https://quantum.cloud.ibm.com/docs/en/guides/pulse

## 4. Qiskit 等门级电路如何转换到你们的 IR？

这里要非常谨慎：**没有一个现成的、官方标准的、通用的“Qiskit gate circuit -> 你们这个 pulse IR”直接映射。**

比较现实的路线是：

1. 先把 `QuantumCircuit` transpile 成 backend-specific ISA circuit  
   这一步由 Qiskit 完成

2. 从 ISA circuit 抽取你们 IR 需要的控制结构  
   例如：
   - `measure` / mid-circuit measure -> `Acquire`
   - classical condition / `if_test` -> `IfBit`
   - 显式 `delay` -> `Delay`

3. 对门操作决定怎么 lower  
   这里分两种：
   - 如果只有门级信息，没有脉冲校准，就只能 lower 到“抽象控制事件”，拿不到真实 `Play/ShiftPhase` 细节
   - 如果有校准/脉冲子程序模型，才有机会继续展开成 `Play`、`ShiftPhase`、frame/port 级事件

4. 再用 backend 的 timing/resource 信息补全  
   比如 duration、alignment、共享资源约束等

所以实话是：

- 从门级电路到你们 IR，可以做
- 但它通常不是“直接转换”
- 而是“先变成 ISA circuit，再结合 backend/native implementation 做自定义 lowering”

更关键的是，**在今天的 IBM 官方栈里，完整 pulse execution 已经不是主流公开接口**。这意味着：

- 你们的 IR 更像一个“验证用的统一低层控制 IR”
- 它可以从门级电路、OpenQASM 3、或其他 pulse DSL lowered 过来
- 但这一步需要你们自己定义映射规则，而不能指望 Qiskit 官方替你们完成

## 对论文最有用的一句话

可以这样讲：

“Qiskit 中一个语法合法、甚至电路级别看似合理的程序，并不自动保证其满足具体后端的低层执行约束；官方引入 ISA-circuit 要求本身就说明了这种 source-level legality 与 hardware-facing executability 之间的 gap 真实存在。我们的工作正是把这部分 gap 形式化为一个可检查的 lowering contract。”

## v0.5 的落地决定：Qiskit Dynamics 的角色

基于后续讨论，`Qiskit Dynamics` 在本项目里的角色已经收敛得比较明确：

- 它不是主验证器
- 它不定义我们的 source semantics
- 它不替代 FullContract

它在 v0.5 里承担的是：

- 一个社区认可的、外部独立的 supporting-evidence 组件
- 用来回答 reviewer/读者的自然问题：
  “如果你们说 lowering 有问题，外部可信系统能不能也观察到差异？”

所以 v0.5 的逻辑是：

1. 我们自己的 pulse-level lowering / contract / checkers 是 **core idea**
2. `FullContract` 是 **主验证器**，负责定义“什么叫 lowering correctness”
3. `Qiskit Dynamics` 是 **主要外部实验平台**，负责提供社区认可的独立证据

当前最适合接 Qiskit Dynamics 的 fault family 是：

- `drop_phase`

因为它最自然地对应到 phase-sensitive 的 pulse 控制差异，
而不会把项目过早拖入更大、更难控的 backend/resource/integration 泥潭。

这里要特别区分两个“主”：

- **主验证器**：谁来判定你们研究问题里的“对/错”
- **主实验平台**：谁来提供最有说服力的外部实验观测

对本项目来说，这两个角色不是同一个东西：

- `FullContract` 是主验证器
- `Qiskit Dynamics` 是主实验平台

这并不矛盾。更准确地说：

- 在**理论/方法**层面，你们自己的 contract 拥有定义权
- 在**实验/证据**层面，`Qiskit Dynamics` 站在前台，作为社区认可的独立见证组件
