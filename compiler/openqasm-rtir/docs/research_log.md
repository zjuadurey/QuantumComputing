# openqasm-rtir 研究日志

This file is a historical log of how the project evolved.
It is useful for reconstructing decisions and version history, but it is
not the authoritative current spec. For the current frozen design, read
`docs/v04_fullcontract_spec.md`; for current usage and module layout, read
`docs/v04_guide.md`.

Minimal context-recovery order across machines:
1. `docs/v04_fullcontract_spec.md`
2. `docs/v04_guide.md`
3. This file
4. `PROJECT_INTENT.md`

---

## 2026-03-25 — v0.5 External supporting evidence + robustness fixes

### 背景

v0.4 已经形成了一个完整的 lowering-correctness pipeline，但仍然存在两个问题：

1. 证据结构偏“自证循环”：语义、checker、fault family 都由我们自己给出；
2. 代码里确实存在一个 reconstruction 层面的 order-sensitivity 缺陷，`early_feedback`
   这个 fault family 也过于依赖特定 fixture 结构。

因此 v0.5 的目标不是替换 FullContract，而是：

- 保持 FullContract 作为主验证器；
- 增加一个外部、社区认可的 supporting-evidence 锚点；
- 同时修掉暴露出来的实现薄弱点。

### 完成的工作

**pulse_lowering/reconstruct.py** — 修复 phase 重建顺序敏感 bug：
- v0.4 中 `phase[f]` 取决于 event list 最后一个元素，而不是时间上最后执行的事件
- v0.5 改成按 `(end, start, event_id)` 选取每个 frame 的最终 phase
- 这样 reconstruction 不再依赖输入 schedule 的列表顺序

**pulse_lowering/buggy_variants.py** — 强化 `lower_buggy_early_feedback`：
- v0.4 只会交换一个相邻的 `(Delay, IfBit)` 对
- v0.5 改成把 `IfBit` 提前越过整个紧邻的 `Delay` block
- 仍然保持“只破坏 causality，不破坏总 time/phase”这个 fault family 设计目标

**pulse_external/qiskit_dynamics.py** — [NEW] 第三方 supporting-evidence 层：
- `simulate_single_frame_schedule(...)`
- `simulate_schedule(...)`
- `compare_single_frame_lowerings(...)`
- `compare_schedule_lowerings(...)`
- 用 `Qiskit Dynamics` 将 selected `PulseEvent` schedules 投影到一个有效单量子比特 witness
- 当前已覆盖：
  - `drop_phase`（phase-sensitive）
  - `ignore_shared_port`（shared-port overlap）
  - `reorder_ports`（flattened schedule）
  - `early_feedback`（timing-sensitive with static drift）

**tests/test_v05_external_corroboration.py** — [NEW] v0.5 测试：
- reconstruction order-insensitivity regression
- stronger `early_feedback` regression
- Qiskit Dynamics identity sanity check
- Qiskit Dynamics witnesses for:
  - `drop_phase`
  - `ignore_shared_port`
  - `reorder_ports`
  - `early_feedback`

### 设计决策

1. 第三方组件的角色是 `corroboration`，不是 `oracle`
2. 先选最自然的 single-frame phase case 做第一条 Qiskit Dynamics witness，再扩到 shared-port 和 timing-sensitive witness
3. `ignore_shared_port` / `reorder_ports` 不强行做 backend API 绑定，而是先在外部 pulse simulator 里展示“同一 lowering bug 会产生独立可见偏差”
4. `early_feedback` 通过对受影响 frame 引入静态漂移来构造 timing-sensitive 外部见证，而不是试图让 Qiskit Dynamics 承担 classical control semantics

### 项目状态

| Item | Status |
|------|--------|
| v0.4 FullContract | ✅ |
| reconstruction order bug | ✅ fixed |
| stronger early_feedback fault | ✅ |
| external supporting evidence layer | ✅ initial Qiskit Dynamics integration |
| v0.5 tests | ✅ |

### 新的证据结构

```
Primary evidence:
  source semantics + lowering contract + independent checkers

Secondary evidence:
  Qiskit Dynamics supporting witness on selected fault families
```

### 结论

v0.5 之后，这个项目的叙事不再是纯粹的“内部自证”：

- FullContract 仍然定义 lowering correctness；
- Qiskit Dynamics 提供独立、社区认可的外部观察窗口；
- 对四类核心 fault（drop_phase / ignore_shared_port / reorder_ports / early_feedback），可以同时给出 contract-level detection 和 external observable deviation。

---

## 2026-03-21 — v0.3 Pulse lowering + correspondence verification

### 背景

v0.2 的 checker 只检查 oracle 自己的输出，FrameConsist 对 oracle 平凡成立（Codex 审查发现）。
需要一个独立编译路径，让 checker 真正验证"编译器输出是否保持源程序语义"。

### 完成的工作

**pulse_lowering/schedule.py** — PulseEvent 数据结构：
- 显式 start/end/port/frame/phase_before/phase_after
- `conditional_on: frozenset[str]` — 记录所有祖先 IfBit 的 cbit 依赖（支持嵌套）

**pulse_lowering/lower_to_schedule.py** — 正确 lowering：
- PulseStmt → list[PulseEvent]，顺序调度，独立跟踪 time/phase
- 不 import ref_semantics
- `active_cbits` 累积集合：进入 IfBit 时 union，退出时恢复，嵌套正确传播

**pulse_lowering/reconstruct.py** — schedule → FrameState 桥接：
- 从 PulseEvent list 重建 FrameState，供 checker 使用
- 独立于 oracle 和 lowering 逻辑

**pulse_lowering/buggy_variants.py** — 4 个有 bug 的 lowering：
- `lower_buggy_drop_phase`: 丢掉 ShiftPhase → FrameConsist 抓到
- `lower_buggy_extra_delay`: 插入多余 delay → FrameConsist 抓到 (time drift)
- `lower_buggy_reorder_ports`: 所有事件压到 t=0 → PortExcl 抓到
- `lower_buggy_early_feedback`: 重排 (Delay, IfBit) → FeedbackCausal 抓到

**pulse_checks/feedback_causality.py** — compiled mode 重设计：
- 接收 `compiled_events: list[PulseEvent]`（不再是 FrameState）
- 从 acquire events 构建 cbit_ready
- 对每个 conditional event 逐一检查 conditional_on 中所有 cbit

**tests/test_lowering_pulse.py** — 13 个端到端测试：
- 3 个正确 lowering 测试（all checks PASS + oracle 一致性）
- 4 个 buggy lowering 测试（target FAILS + ALL non-targets PASS）
- 4 个 schedule 结构测试（event count, explicit times, phase snapshots, conditional tagging）
- 2 个嵌套 IfBit 测试（正确累积 + 外层 cbit 未 ready 被抓到）

### 设计决策

1. Lowering 独立于 ref_semantics（两条独立编译路径）
2. reconstruct_state 从 event list 读数据，不知道 lowering 内部逻辑
3. Buggy variants 通过修改输入程序实现（重排/删除/插入），而非篡改 state
4. `lower_buggy_early_feedback` 用重排（不删 Delay），保证只破坏 causality 不破坏 FrameConsist（bug specificity）
5. `conditional_on` 用 `frozenset[str]` 而非 `str | None`，正确处理嵌套 IfBit
6. Compiled mode FeedbackCausal 在 schedule 级别逐 event 检查，不依赖 final FrameState time

### Codex 审查历史

- **R1**: FeedbackCausal 未走 lowering 链路 + specificity 未测试 → 添加 compiled mode + non-target 断言
- **R2**: early_feedback 删 Delay 同时破坏 FrameConsist + compiled mode 用 final time 不精确 → 改为重排 + 改用 events
- **R3**: nested IfBit 丢失外层条件 → frozenset 累积 + 2 个嵌套测试

### 项目状态

| Item | Status |
|------|--------|
| v0.1 gate-level MVP | ✅ (10 tests) |
| v0.2 pulse-level prototype | ✅ (14 tests) |
| v0.3 pulse lowering + correspondence | ✅ (13 tests) |
| 全部测试 | ✅ 37/37 passing |
| 论文 | ❌ 未开始 |

### 当前完整流水线

```
Source program (PulseStmt)
    ├──→ ref_semantics.run()        → oracle FrameState (ground truth)
    └──→ lower_to_schedule()        → PulseEvent list (with conditional_on)
            └──→ reconstruct_state() → compiled FrameState
                    └──→ checkers 对比 source 语义
                         (FeedbackCausal 直接检查 events)
```

---

## 2026-03-21 — v0.2 Pulse-level 原型实现

### 背景

ASE 2026 投稿截止 03-26，需在 5 天内完成 pulse-level 原型 + 论文。
PLanQC 2026 已有相关工作（Wu et al., "A Pulse-Level DSL for Real-Time Quantum Control with Hardware Compilation and Emulation"），
做 DSL + compilation + emulation，但不做形式化验证——我们的工作填补这个 gap。

分工：Claude 编码，ChatGPT 审查与方向把控。

### 完成的工作

**pulse_ir/ir.py** — 数据类型，严格对应 formal_definitions_v0.md §1.2：
- `Waveform(name, duration)` — 不透明波形信封
- `PulseStmt` 联合类型：`Play | Acquire | ShiftPhase | Delay | IfBit`
- `Config` — 静态硬件描述（frames, ports, port_of, init_freq, init_phase）
- `FrameState` — 可变执行状态（time, phase, cbit, cbit_ready, occupancy）

**pulse_ir/ref_semantics.py** — 参考语义（独立 oracle）：
- `step(state, stmt, config)` — 单步执行，返回新状态（不可变）
- `run(program, config)` — 顺序执行完整程序
- IfBit 在 oracle 中无条件执行 body（保守 trace）

**pulse_checks/** — 三个独立检查器（不复用 ref_semantics 代码路径）：
- `port_exclusivity.py` — 检查 occupancy 区间无重叠
- `feedback_causality.py` — 独立跟踪 frame_time 和 cbit_ready，检查 IfBit 时序
- `frame_consistency.py` — 从 AST 独立计算 expected phase，与 state 对比

**pulse_examples/** — 6 个示例（3 正例 + 3 反例）：
- `correct_single_play.py` — 单 frame 单 play
- `correct_measure_feedback.py` — acquire + delay + IfBit（因果正确）
- `correct_multi_frame.py` — 双 frame 不同 port + ShiftPhase
- `violation_port_conflict.py` — 双 frame 共享 port，play 重叠
- `violation_causality.py` — IfBit 在 acquire 完成前触发
- `violation_phase.py` — 模拟编译器 bug 导致 phase 偏移

**tests/test_pulse.py** — 13 个测试全部通过：
- 4 个 step 单元测试（play/acquire/shift_phase/delay）
- 3 个正例集成测试（三项检查全 PASS）
- 3 个反例集成测试（各对应检查 FAIL）
- 3 个边界测试（空程序、多次 shift 累加、IfBit oracle 行为）

### 设计决策

1. **Oracle vs Checker 分离**：ref_semantics.py 是 oracle，三个 checker 独立实现时序/相位计算，不 import ref_semantics
2. **IfBit oracle 行为**：oracle 无条件执行 body，产生保守 trace；checker 独立检查因果约束
3. **Phase corruption 测试策略**：violation_phase 通过 post-hoc 篡改 oracle 输出模拟编译器 bug，而非构造"错误的 ref_semantics"
4. **Config 用 frozen dataclass**：初始化后不可变，assert 校验所有 frame 映射完整

### 项目状态

| Item | Status |
|------|--------|
| v0.1 gate-level MVP | ✅ (10 tests) |
| v0.2 pulse-level prototype | ✅ (13 tests) |
| 全部测试 | ✅ 23/23 passing |
| 形式化定义 | ✅ formal_definitions_v0.md |
| 论文 | ❌ 未开始 |

---

## 2026-03-13 — v0.1 MVP: 最小闭环打通

### 目标

建立最小可验证编译原型：OpenQASM 3 小子集 → real-time IR → timeline → 正确性检查。
不做大而全编译器，先把链路跑通。

### 完成的工作

1. **项目结构搭建** — 6 个 Python 模块 + 2 个 qasm 示例 + 10 个测试
2. **parser_bridge** — 通过 `qiskit_qasm3_import.parse()` 做语法校验，确认 OpenQASM 3 入口通路
3. **RTEvent IR 定义** — dataclass，字段：event_id, kind, start, duration, resource, qubit, creg, condition, payload, depends_on；end 作为 property
4. **Regex-based toy lowering** — 逐行正则匹配受控子集，维护 qubit_ready / resource_ready / classical_ready / last_writer 四个跟踪 map，贪心调度
5. **Timeline 输出** — 格式化表格打印
6. **两个检查器** —
   - `no_conflict`: 同一 resource 上的事件按 start 排序后检查区间重叠
   - `causality`: 每个 depends_on 依赖是否在当前事件 start 前结束
7. **pytest 10/10 通过**

### 关键发现与决策

- **qiskit 对 bit 条件的限制**：`qiskit_qasm3_import.parse()` 不接受 `if (c[0] == 1)`，要求 `if (c[0])` 或 `if (c[0] == true)`。这是 qiskit 的 parser 行为，不是 OpenQASM 3 规范的限制。regex lowering 已做兼容，同时支持两种写法。
- **资源模型**：当前用 `drive_q{i}` 和 `measure_q{i}` 两类资源。实际硬件有更细粒度的资源（frame、port、channel），但 v0.1 先抽象为两类。
- **调度策略**：纯贪心、顺序调度。没有并行 qubit 的场景（v0.1 只有 1 qubit），后续加多 qubit 时需要处理并行调度。
- **branch 的因果语义**：branch 的 start 取 max(qubit_ready, drive_ready, classical_ready)，并显式记录 depends_on 指向产生该 classical bit 的 measure 事件。这是本项目的核心语义贡献点。

### 运行结果

**simple_delay.qasm**: h(0..10) → delay(10..30) → measure(30..60) → branch(60..70, dep=measure)

**measure_if.qasm**: x(0..10) → measure(10..40) → branch(40..50, dep=measure)

两个示例的 no_conflict 和 causality 检查均 PASS。

### 时间线语义验证

以 simple_delay.qasm 为例手动验证：
- h 在 drive_q0 上 [0,10)
- delay 在 drive_q0 上 [10,30) — 紧接 h 之后，无冲突
- measure 在 measure_q0 上 [30,60) — qubit ready=30，与 drive_q0 无冲突
- branch 在 drive_q0 上 [60,70) — 必须等 classical_ready=60（measure 结束），drive_q0 free=30，取 max=60
- depends_on: branch → measure（event 2），measure.end=60 ≤ branch.start=60 ✓

### 下一步方向（v0.2 候选）

1. **AST-driven lowering** — 用 QuantumCircuit.data 遍历 instructions 替代正则，更鲁棒
2. **Z3 约束检查** — 将 start/end/depends_on 编码为 SMT 公式，做形式化可满足性验证
3. **多 qubit 扩展** — 加 cx，引入 multi-resource locking 和并行 timeline
4. **feedback 延迟建模** — 加入 classical processing latency（measure → classical ready 之间的传播延迟）
5. **toy pulse 语义** — 把 gate 展开为 play(frame, waveform, duration)，引入 frame/port 资源

---

## 2026-03-13 — Baseline 设计：三层框架

### 问题

v0.1 最初的 baseline 思路是"拿 Qiskit/pytket 跑，展示它们静默通过而我能报错"。
这对 SE/ICSE 风格 bug-finding 叙事够用，但对 PLDI/POPL/FM 不够。
审稿人会追问：(1) 和已有形式化工作比新在哪 (2) 你怎么知道你报的是对的。

### 决策：三层 baseline

**L1 工程 baseline — Qiskit / pytket**
- 作用：衡量现实工具链对 timing/resource/feedback 违例的暴露能力
- 不是为了证明"比它们强"，而是回答"用户今天交给主流工具会怎样"
- Qiskit 有 ASAP/ALAP scheduling pass + dynamic circuit 支持
- pytket 有 conditional gate 支持 + 编译时保留 conditional data
- 它们做调度但不做独立的形式化时序/因果验证

**L2 形式化 baseline — Giallar / VOQC**
- 作用：restricted-scope 的形式化对照
- 说明"已有验证到哪里为止"（门级等价、pass 正确性），"我把验证边界推到哪里"（timing/resource/feedback）
- 不是 end-to-end 竞品，而是能力维度对比

**L3 Oracle baseline — Z3/SMT 可执行语义**
- 作用：correctness ground truth
- 对小规模程序用 SMT 穷举/约束求解得到"真值"
- 用于评估 checker 的 soundness / precision / false positive / false negative
- 这是审稿人最关心的：你怎么知道你报的对

### 四类评测 case

| 类型 | 含义 |
|------|------|
| 时序违例 | delay/duration/alignment 造成先后关系错误 |
| 资源违例 | port/frame/waveform 使用冲突 |
| 反馈违例 | 测量结果尚不可用就触发条件控制 |
| lowering 违例 | lowering 引入重排导致可观察行为变化 |

### 实现启示

Z3 oracle 在项目中有双重角色：
- 作为 v0.2 的核心功能（约束化检查）
- 同时作为 baseline L3 的 oracle

这意味着 v0.2 的 Z3 工作不只是"升级 checker"，而是同时在构建评测基础设施。
应优先实现。

---

## 2026-03-19 — 方向决策：gate warmup + pulse core

### 背景

经过多轮讨论（含与 ChatGPT 交叉验证），确定论文策略和技术路线。

### 关键决策

**1. 论文策略：gate 层 warmup + pulse 层核心贡献**
- Gate-level（v0.1 已有）只做方法论演示（≤3页），展示 no_conflict + causality
- Pulse-level 是核心贡献：对 OpenPulse 语义的形式化定义 + 验证框架
- 不做"只 gate"也不做"gate+pulse 完整统一框架"，取中间路线

**2. 锚定 OpenQASM/OpenPulse 规范层，不锚定厂商 API**
- IBM 已弃用 qiskit.pulse（SDK v1.3 起弃用，v2.x 移除）
- 但 OpenQASM 规范仍保留 OpenPulse grammar
- 研究对象是语言/IR 语义，不是某个 SDK

**3. Pulse 核心对象（最小子集）**

```
PulseStmt ::= Play(frame, waveform, duration)
            | Acquire(frame, duration, creg)
            | ShiftPhase(frame, angle)
            | Delay(duration, frame)
```

资源模型：port + frame（frame 绑定 port，携带 time/freq/phase）

暂不做：SetFreq、defcal、Barrier（留给后续扩展）

**4. 三条 pulse-level 可验证性质**

| 性质 | 定义 |
|------|------|
| Port exclusivity | 同一 port 上 play/acquire 区间不重叠 |
| Feedback causality | acquire 完成后 classical result 才可用于条件动作 |
| Frame consistency | phase = initial + Σ(explicit_shifts) + 2π × freq × Σ(Δt) |

**5. 明确不做的事**
- Waveform-to-unitary correspondence（量子控制理论问题）
- 强版本 timing equivalence / timed bisimulation
- 复杂 defcal 绑定语义
- 大而全 pulse toolchain

### Baseline 框架（修正版）

| 位置 | 内容 |
|------|------|
| Related work | VOQC / Giallar / CertiQ capability table（不进 evaluation） |
| Evaluation | benchmark × independent oracle × checker × Qiskit/pytket |

关键修正：
- L2（Giallar/VOQC）从 evaluation 移回 related work——它们是能力维度对比，不是实验 baseline
- Oracle 必须独立于 checker（不能 checker 用 Z3、oracle 也用 Z3）
- 可行的 oracle 策略：小规模穷举枚举 + 手工证明
- timing_intent_preservation 需先收窄定义才能进 evaluation；弱版本（delay duration 保持）太 trivial，需要至少做到"时序间距约束被尊重"

### 方法论执行顺序

```
1. 抽象语法定义（gate subset + pulse subset）
2. 三条性质的数学表述
3. Reference semantics（独立 oracle）
4. Checker 实现
5. Benchmark 构造 + oracle 标注
6. Qiskit/pytket 工程对照
```

核心原则："定义是第一产物，checker 只是定义的一个实现"

### 前置条件

需要确认能否在不依赖 defcal 的前提下，把 port/frame/play/acquire/shift_phase 的语义讲清楚。
已整理 OpenPulse 语义摘要：`docs/openpulse_semantics_summary.md`

---

## 2026-03-19 — 项目本质定义

### 一句话定义

> 把一种已经被规范提出、但还没有被精确定义和可靠检查的程序对象，
> 变成一个可以被理解、被验证、被自动检查的研究对象。

### 展开

OpenPulse 规范说了"frame 有频率、有相位、play 会推进时间"，但它没有给出：

- **精确的状态转移规则**：执行一条 play 之后，state 的每个分量怎么变
- **可判定的正确性性质**：什么叫"这个 schedule 是对的"
- **独立于实现的参考语义**：不依赖任何编译器，纯粹从定义出发的"标准答案"

本项目补上这三样东西。

论文贡献不是"做了一个工具"，而是**把一个非形式化的规范对象变成了一个可验证的形式化对象**。工具只是定义的副产品。

### 三层意义

**第一层：把"会写"变成"说得清"。**
OpenPulse 已经给出了 play/acquire/shift_phase/delay，但规范存在不等于 correctness 已经被形式化。我们回答：什么叫端口冲突？什么叫反馈因果正确？什么叫 frame 相位一致？——给这类程序建立清楚的语义和判据。

**第二层：把"经验上没问题"变成"可以检查"。**
现有工具能生成/处理脉冲控制序列，但不给出严格保证。我们的工作是：不只是"编译出来了"，还要能判断"它在时序、资源、反馈上是不是对的"。为 pulse-level 量子程序提供 correctness checking 的基础。

**第三层：为后续更大的工作打地基。**
一旦这套小核立住，后续才有统一基础：更复杂的 pulse 编译、calibration/defcal 一致性检查、更强的 lowering correctness、更完整的 verified quantum control stack。当前不是终点，而是第一层基础设施。

### 一句话本质

> **为一类贴近硬件控制的量子程序，建立"什么叫正确"以及"如何自动判断正确"的基础。**
>
> 即：pulse-core 程序的 correctness model。
>
> 不是在证明某个厂商 API 好不好，不是在做完整编译器，也不是在做控制物理。

### 形式化定义文档

已产出第一版：`docs/formal_definitions_v0.md`

内容：pulse core 抽象语法、Config/State 定义、四条 step 规则、三条可验证性质（port exclusivity, feedback causality, frame consistency）。

---
