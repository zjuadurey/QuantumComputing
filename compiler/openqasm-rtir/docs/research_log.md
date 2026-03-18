# openqasm-rtir 研究日志

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
