# v0.4 Guide — 代码做了什么、怎么用

## 一句话总结

v0.4 实现了 **FullContract**：对一个受 OpenPulse 启发的最小 pulse-control core calculus，
先检查源程序合法性（well-formedness），再验证 lowering 产出的 schedule 是否保持三个
正确性性质（port exclusivity, feedback causality, frame consistency）。

## 和 v0.3 的区别

| | v0.3 | v0.4 |
|---|---|---|
| 时间模型 | per-frame 独立时钟 | port-aware: `start = max(time[f], port_time[p])` |
| 共享 port | 会产生重叠（oracle 可违反 PortExcl） | 自动串行（oracle 保证 PortExcl） |
| phase 等待 | 不存在 | stall 期间 phase 继续演化 |
| 源程序检查 | 无 | WF precheck 拒绝 ill-formed 程序 |
| 统一入口 | 无 | `verify_lowering()` 返回 `VerificationReport` |
| 测试 | 27 pulse tests | 38 pulse tests + 10 gate = 48 total |
| buggy variants | 4 | 5（+IgnoreSharedPort） |

## 文件结构

```
pulse_ir/
├── ir.py                    ← FrameState 新增 port_time
└── ref_semantics.py         ← port-aware step rules

pulse_checks/
├── wellformedness.py        ← [NEW] WF(P,C) source precheck
├── port_exclusivity.py      ← 无变化
├── feedback_causality.py    ← schedule-level `check_schedule_causality(events)`
└── frame_consistency.py     ← port-aware source-vs-compiled correspondence

pulse_lowering/
├── lower_to_schedule.py     ← port-aware lowering
├── reconstruct.py           ← 重建 FrameState（含 port_time）
├── verify.py                ← [NEW] verify_lowering() 统一入口
├── buggy_variants.py        ← +lower_buggy_ignore_shared_port
└── schedule.py              ← 无变化

pulse_examples/
├── correct_shared_port.py   ← [NEW] 共享 port 正确示例
├── violation_port_conflict.py ← 改为 lowering bug 示例
├── violation_causality.py   ← 改为 ill-formed 示例（WF 拒绝）
└── ...（其余不变）

tests/
├── test_pulse.py            ← 24 tests（含 WF、shared port、verify_lowering）
└── test_lowering_pulse.py   ← 14 tests（含 IgnoreSharedPort）
```

## 怎么跑

```bash
# 跑全部测试（48 tests）
conda run -n qiskit_qasm_py312 --no-capture-output pytest -q tests/

# 只跑 pulse-level 测试
conda run -n qiskit_qasm_py312 --no-capture-output pytest -q tests/test_pulse.py tests/test_lowering_pulse.py

# 用 verify_lowering() API
conda run -n qiskit_qasm_py312 --no-capture-output python -c "
from pulse_examples.correct_measure_feedback import config, program
from pulse_lowering.verify import verify_lowering
report = verify_lowering(program, config)
print(f'WF: {report.well_formed}')
print(f'PortExcl: {report.port_exclusive}')
print(f'FeedbackCausal: {report.feedback_causal}')
print(f'FrameConsist: {report.frame_consistent}')
print(f'Overall: {report.overall_ok}')
"
```

## 核心 API

```python
from pulse_lowering.verify import verify_lowering, VerificationReport
from pulse_checks.wellformedness import check_wellformedness
from pulse_checks.feedback_causality import check_schedule_causality

report: VerificationReport = verify_lowering(program, config)
# report.well_formed       — 源程序合法性
# report.port_exclusive    — PortExcl on compiled schedule
# report.feedback_causal   — FeedbackCausal on compiled schedule (event-level)
# report.frame_consistent  — FrameConsist: compiled state vs source semantics
# report.overall_ok        — 全部通过
# report.oracle_state      — oracle 产出的 ground truth
# report.compiled_state    — lowering → reconstruct 产出的 compiled state
# report.events            — lowered schedule (list[PulseEvent])
# report.errors            — 所有错误信息

# 用 buggy lowering 测试
from pulse_lowering.buggy_variants import lower_buggy_ignore_shared_port
report = verify_lowering(program, config, lower=lower_buggy_ignore_shared_port)
assert not report.overall_ok
```

## API migration（v0.3 → v0.4）

- source-side feedback legality 不再通过 `check_feedback_causality(program, config)` 检查
- source-side legality 现在统一走 `check_wellformedness(program, config)`
- schedule-side feedback 正确性现在统一走 `check_schedule_causality(events)`

也就是说，v0.4 明确拆成两层：
- `wellformedness.py` 负责源程序合法性
- `feedback_causality.py` 只负责 lowered schedule 的 event-level causality

## 语义核心（v0.4 改动）

**Play(f, w):**
```
start = max(σ.time[f], σ.port_time[p])
end = start + dur(w)
σ'.time[f]      = end
σ'.phase[f]     = σ.phase[f] + 2π × freq × (end - σ.time[f])  # 含 stall
σ'.port_time[p] = end
σ'.occupancy[p] ∪= {(start, end)}
```

**WF(P, C):**
```
∀ IfBit(c, body) in P:
  1. c 已被 Acquire 定义
  2. t_use = body 的真实开始时刻（Play/Acquire 取 max(frame_time, port_time)）
  3. t_use ≥ cbit_ready[c]
  4. 嵌套 IfBit: 所有祖先依赖也必须满足
```

## 5 个 buggy variant

| Variant | Bug | Caught by |
|---------|-----|-----------|
| drop_phase | 丢掉 ShiftPhase | FrameConsist |
| extra_delay | 插入多余 delay | FrameConsist |
| reorder_ports | 全部扁平到 t=0 | PortExcl + FrameConsist |
| early_feedback | IfBit 提前执行 | FeedbackCausal |
| ignore_shared_port | 忽略 port_time | PortExcl + FrameConsist |

## 下一步

- 交 Codex review（更新 review_spec.md）
- 扩展 Pool B（15-20 curated programs）
- 实现 partial validators + ablation
- 跑实验填表
- 论文 §3-§5 同步更新
