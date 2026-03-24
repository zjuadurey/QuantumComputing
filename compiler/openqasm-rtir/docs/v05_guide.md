# v0.5 Guide — external corroboration on top of FullContract

This guide describes the runnable additions introduced in v0.5.
For the frozen v0.4 contract itself, still read:

1. `docs/v04_fullcontract_spec.md`
2. `docs/v04_guide.md`
3. this file

## 一句话总结

v0.5 不改变 FullContract 作为主验证器的定位；它增加了一层
**external supporting evidence**，当前使用 `Qiskit Dynamics`
作为**主要外部实验平台**，为 selected lowering faults 提供独立佐证。

## v0.5 相比 v0.4 的增量

| Area | v0.4 | v0.5 |
|---|---|---|
| 主验证器 | FullContract | FullContract（不变） |
| 第三方锚点 | 无 | `Qiskit Dynamics` 作为主要外部实验平台 |
| reconstruction | 对事件列表顺序敏感 | 修复为按最新执行位置重建 |
| `early_feedback` seeded fault | 只跨单个相邻 `Delay` | 跨整个紧邻 delay block |
| 新测试 | 无 external layer | 新增 v0.5 corroboration tests |

## 三个角色要分开

v0.5 之后，仓库里有三个不同角色：

1. **Core idea**
   - 你们自己的 pulse-level lowering / contract / checker pipeline

2. **主验证器**
   - `FullContract`
   - 决定本文里“什么算 lowering correctness”

3. **主要外部实验平台**
   - `Qiskit Dynamics`
   - 在实验部分承担主要的外部见证作用

所以：

- `Qiskit Dynamics` 可以是主实验平台
- 但它仍然不是主验证器
- 更不是 source semantics 的定义者

## 新增模块

```text
pulse_external/
├── __init__.py
└── qiskit_dynamics.py
```

核心 API：

```python
from pulse_external.qiskit_dynamics import (
    compare_schedule_lowerings,
    compare_single_frame_lowerings,
)

result = compare_single_frame_lowerings(
    program,
    config,
    frame="d0",
    lower_candidate=lower_buggy_drop_phase,
    drive_scale=0.02,
)
print(result.fidelity)
```

更通用的 schedule-level API：

```python
result = compare_schedule_lowerings(
    program,
    config,
    scope="shared-port",
    lower_candidate=lower_buggy_ignore_shared_port,
    drive_scale=0.02,
)
print(result.fidelity)
```

解释：

- `lower_to_schedule` 仍然是正确 lowering 的基线
- `lower_candidate` 是待对照的 faulty lowering
- 返回的 `fidelity` 越小，表示作为主实验平台的外部组件看到的偏差越明显

## 怎么跑

### 1. 跑 v0.5 相关测试

```bash
conda run -n qiskit_qasm_py312 --no-capture-output \
  pytest -q tests/test_lowering_pulse.py tests/test_v05_external_corroboration.py
```

### 2. 手动跑一个 `drop_phase` 外部佐证

```bash
conda run -n qiskit_qasm_py312 --no-capture-output python - <<'PY'
import math
from pulse_ir.ir import Config, Waveform, Play, ShiftPhase
from pulse_lowering.buggy_variants import lower_buggy_drop_phase
from pulse_external.qiskit_dynamics import compare_single_frame_lowerings

cfg = Config(
    frames=frozenset(["d0"]),
    ports=frozenset(["p0"]),
    port_of={"d0": "p0"},
    init_freq={"d0": 5.0e-3},
    init_phase={"d0": 0.0},
)
prog = [
    Play("d0", Waveform("g160", 160)),
    ShiftPhase("d0", math.pi / 2),
    Play("d0", Waveform("g160", 160)),
]
result = compare_single_frame_lowerings(
    prog,
    cfg,
    frame="d0",
    lower_candidate=lower_buggy_drop_phase,
    drive_scale=0.02,
)
print("fidelity:", result.fidelity)
PY
```

### 3. 手动跑一个 shared-port 外部佐证

```bash
conda run -n qiskit_qasm_py312 --no-capture-output python - <<'PY'
import math
from pulse_ir.ir import Config, Waveform, Play, ShiftPhase
from pulse_lowering.buggy_variants import lower_buggy_ignore_shared_port
from pulse_external.qiskit_dynamics import compare_schedule_lowerings

cfg = Config(
    frames=frozenset(["d0", "d1"]),
    ports=frozenset(["p0"]),
    port_of={"d0": "p0", "d1": "p0"},
    init_freq={"d0": 0.0, "d1": 0.0},
    init_phase={"d0": 0.0, "d1": 0.0},
)
prog = [
    Play("d0", Waveform("g160", 160)),
    ShiftPhase("d1", math.pi / 2),
    Play("d1", Waveform("g200", 200)),
]
result = compare_schedule_lowerings(
    prog,
    cfg,
    scope="shared-port",
    lower_candidate=lower_buggy_ignore_shared_port,
    drive_scale=0.02,
)
print("fidelity:", result.fidelity)
PY
```

## 当前支持范围

v0.5 不是一个“大而全”的 simulator integration。当前只做：

- single-frame / selected multi-frame witnesses
- play-event driven external simulation
- `drop_phase`, `ignore_shared_port`, `reorder_ports`, `early_feedback` 的 selected corroboration
- 不把外部 simulator 扩张成新的主验证器

这符合 v0.5 的设计目标：先加一个可信锚点，而不是让第三方组件吞掉论文主线。

## 推荐的论文表述

可以直接把 v0.5 的角色说成：

> FullContract remains the primary verifier of lowering correctness.
> Our pulse-level lowering contract and checker pipeline constitute the core idea.
> Qiskit Dynamics is used as the primary external experimental witness
> to show that selected contract-detected faults also induce externally
> observable deviations.

这句话很重要，因为它明确防止 reviewer 把外部 simulator 当成你工作的真正定义者。
