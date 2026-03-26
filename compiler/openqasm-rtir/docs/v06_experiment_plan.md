# v0.6 实验章节规划

这份文档整理了当前关于实验设计、指标选择、真实输入来源和论文 `Evaluation`
章节结构的讨论结果。目标不是给出最终投稿文字，而是形成一份稳定的实验写作底稿。

---

## 1. 当前实验主线

当前项目的实验链路已经收敛为：

1. **真实 Qiskit pulse 代码作为前端输入**
   - 输入对象是 `ScheduleBlock`
   - 当前只支持一个受限但真实存在的子集：
     - `play`
     - `delay`
     - `shift_phase`
     - `acquire`

2. **Qiskit pulse -> core IR**
   - 通过 `pulse_frontends/qiskit_pulse.py`
   - 翻译为：
     - `Play`
     - `Delay`
     - `ShiftPhase`
     - `Acquire`
     - 以及对应 `Config`

3. **core IR -> FullContract**
   - `FullContract` 是主验证器
   - 输出：
     - Boolean correctness verdicts
     - structured diagnostics

4. **correct / faulty lowered schedules -> Qiskit Dynamics**
   - `Qiskit Dynamics` 不是 verifier
   - 它是主要外部实验平台
   - 负责提供 independent external witness

5. **Qiskit Dynamics -> fidelity**
   - 提供 external observable deviation

因此，最准确的说法不是简单的：

> Qiskit pulse -> our IR -> Qiskit Dynamics

而是：

> Qiskit pulse -> our IR -> FullContract, and then
> lowered schedules -> Qiskit Dynamics witness.

---

## 2. 角色分工

实验和方法中必须持续保持下面的角色区分：

- `core IR`
  - verification-oriented pulse core
  - 不是对 Qiskit Pulse 的再实现

- `FullContract`
  - primary verifier
  - 定义什么叫 lowering correctness

- `Qiskit Dynamics`
  - primary external experimental witness
  - 不定义语义
  - 不替代 contract

- `Qiskit pulse ScheduleBlock`
  - realistic frontend entry
  - 证明系统不是只吃手写 toy IR

一句最稳的表述：

> We use real Qiskit pulse code as a frontend entry, translate it into our
> verification-oriented core IR, verify lowering correctness with FullContract,
> and then project lowered schedules into Qiskit Dynamics as an external
> witness of observable deviation.

---

## 3. 为什么实验指标不能只找一个“分数”

形式化验证的核心输出天然是：

- pass / fail
- proved / violated
- sound / unsound

所以这类工作不适合硬凑一个统一 accuracy-style 指标。

更自然的实验证据应当是组合型的：

1. **correctness verdict**
   - Boolean
   - 表示是否满足 contract

2. **fault coverage / discriminative power**
   - 不同 checker 能抓到哪些 fault family

3. **diagnostic magnitude**
   - phase drift
   - time drift
   - overlap duration
   - feedback earliness

4. **external corroboration**
   - Qiskit Dynamics fidelity

5. **real-entry applicability**
   - 真实 Qiskit pulse 代码能否进入系统
   - 当前支持哪些构件
   - 有多少真实样例已接入

因此，实验章节不应该寻找一个万能指标，而应准备一组互补证据。

---

## 4. 建议的实验问题（RQs）

### RQ1: Can real-entry Qiskit pulse programs be ingested by the framework?

回答目标：

- 系统不是只吃手写 IR
- 存在真实来源的 Qiskit pulse 代码
- 它们能翻译成当前 core IR

推荐报告：

- 支持的 Qiskit pulse subset
- 接入的真实样例列表
- 每个样例翻译出的 IR 构件

### RQ2: Which contract components detect which lowering faults?

回答目标：

- `FullContract` 不是空定义
- fault families 与 checker 之间存在清晰关系

推荐报告：

- `WF`
- `PortExcl`
- `FeedbackCausal_sched`
- `FrameConsist`

即当前已有的 bug matrix。

### RQ3: How large are the detected contract violations?

回答目标：

- 系统不只是 yes/no
- diagnostics 能量化错误幅度

推荐报告：

- phase drift
- time drift
- max overlap dt
- total overlap dt
- earliness dt

### RQ4: Are contract-detected faults also externally visible?

回答目标：

- 避免 define-detect-declare 自证循环
- 独立组件也能看见这些 fault 的 effect

推荐报告：

- witness type
- fidelity

---

## 5. 推荐的实验章节结构

最推荐的整体版式：

1. **Setup / Pipeline**
   - 一小段解释实验链路
   - 配一张总图

2. **RQ1: real-entry ingestion**
   - 真实 Qiskit pulse 入口

3. **RQ2: contract detection**
   - bug matrix

4. **RQ3: diagnostic magnitude**
   - drift / overlap / earliness

5. **RQ4: external corroboration**
   - fidelity

6. **case study**
   - 一个最直观的 schedule-level figure

推荐图表组合：

- Figure 1: evidence-structure figure
- Table 1: Qiskit frontend subset / real-entry examples
- Table 2: fault-family coverage + diagnostics + fidelity
- Figure 2: `ignore_shared_port` 或 `early_feedback` case study

---

## 6. Case study 应该长什么样

最合适的是挑一个 schedule 层面最直观的例子，比如：

- `ignore_shared_port`
- 或 `early_feedback`

一张好的 case-study figure 应该同时包含：

- correct schedule
- faulty schedule
- 关键冲突位置（overlap 或 earliness）
- checker verdict / diagnostic magnitude
- external fidelity

目的不是再讲一遍全套理论，而是让读者一眼看到：

> the same lowering bug is visible in the schedule, in the contract,
> and in the external witness.

---

## 7. 关于真实 Qiskit 输入的表述

当前已经确认：

- Qiskit Pulse 虽然 deprecated / removed as a public user-facing API path
- 但真实的 `ScheduleBlock` pulse code 确实存在
- 而且其中一部分可以映射到当前 core IR

当前仓库中的真实入口样例位于：

- `qiskit_examples/single_drive_with_virtual_z.py`
- `qiskit_examples/drive_and_acquire.py`
- `qiskit_examples/two_drive_channels.py`

这些文件已经带来源注释。

实验里最稳的 claim 是：

> Our framework already admits a realistic frontend entry via a restricted
> subset of real Qiskit pulse `ScheduleBlock`s, rather than only hand-written
> core IR programs.

不应该写成：

> We support Qiskit Pulse in general.

也不应该写成：

> We ingest arbitrary real-world pulse programs.

---

## 8. 当前最重要的边界

实验中必须持续强调下面这些边界：

1. 当前 Qiskit front-end 支持的是受限子集
   - `play`
   - `delay`
   - `shift_phase`
   - `acquire`

2. 当前不支持：
   - `shift_frequency`
   - 更复杂的 alignment / barrier 语义
   - pulse-level `IfBit` 对等输入

3. `Qiskit Dynamics` 是 witness，不是 verifier

4. 论文关注的是 lowering correctness，不是 physics-level correctness

---

## 9. 为什么工作仍然有意义

一个已经明确的重要定位是：

> Pulse-level control remains important in the underlying hardware-control
> stack, even though publicly exposed user-facing pulse APIs such as Qiskit
> Pulse have been deprecated or removed.

因此，实验章节不应把意义建立在“Qiskit Pulse 是今天主流用户接口”上，
而应建立在：

- pulse-level control semantics 仍然存在于 hardware-facing stack 中
- source-to-schedule correctness 问题依然真实存在
- 现有公开工具链没有给出 lowering contract

---

## 10. 接下来最值得补强的实验点

按优先级排序：

1. **real-entry example table**
   - 把真实 Qiskit 输入系统化列出来

2. **one strong case-study figure**
   - 推荐 `ignore_shared_port`

3. **把 Qiskit frontend 入口并入主实验脚本**
   - 让实验章节不只是说“我们能 ingest”
   - 而是能展示实际结果

4. **后续可选增强**
   - 更多真实来源样例
   - 更强的 alignment/barrier 语义
   - eventually: dynamic frequency support

---

## 11. 一句话版本

实验章节的核心逻辑应当是：

> 先证明真实来源输入能进来，再证明 FullContract 能抓到 lowering 错误，
> 然后证明这些错误在外部公认组件中也会表现出可观察后果。

