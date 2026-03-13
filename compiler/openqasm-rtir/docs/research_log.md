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
