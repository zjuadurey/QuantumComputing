# IR Lowering 到低层脉冲表示的验证案例

本文档用一个具体例子说明，这篇论文中的形式化验证如何检查一个 pulse-level IR 的 lowering 是否正确。

## 1. 例子目标

我们从一个简化的源级脉冲程序出发：

```text
[
  Acquire(f_meas, 4, c0),
  IfBit(c0, Play(f_drive, x90[dur=6])),
  ShiftPhase(f_drive, π/2),
  Delay(2, f_drive)
]
```

它表达的意思是：

- 先在测量 frame `f_meas` 上做一次时长为 `4 dt` 的 `Acquire`，结果写入经典位 `c0`
- 如果 `c0` 可用，则在驱动 frame `f_drive` 上打一段时长为 `6 dt` 的 `x90` 脉冲
- 然后对 `f_drive` 做一次 `ShiftPhase(π/2)`
- 最后再延迟 `2 dt`

论文的验证目标不是“程序能不能 lower”，而是“lower 之后的显式 schedule 是否保持了源程序要求的时间、资源和反馈语义”。

## 2. 硬件配置

为了体现共享端口约束，设定：

```text
frames = {f_meas, f_drive}
ports  = {p0}

port_of(f_meas)  = p0
port_of(f_drive) = p0

init_phase(f_meas)  = 0
init_phase(f_drive) = 0

init_freq(f_meas)   = 0
init_freq(f_drive)  = 1/8
```

这里最关键的是 `f_meas` 和 `f_drive` 共享同一个物理端口 `p0`。因此，后续 `Play(f_drive, ...)` 即使 frame 本地时间还是 `0`，也必须等待 `Acquire` 释放端口后才能开始。

## 3. 从源 IR 到低层 PulseEvent

论文中的 lowering 会把源级 `PulseStmt` 编译为一个显式的 `PulseEvent` 列表。对这个例子，lowering 的输出可写成：

```text
[
  PulseEvent(
    event_id=0,
    kind="acquire",
    frame="f_meas",
    port="p0",
    start=0,
    end=4,
    phase_before=0,
    phase_after=0,
    cbit="c0",
    conditional_on=∅
  ),

  PulseEvent(
    event_id=1,
    kind="play",
    frame="f_drive",
    port="p0",
    start=4,
    end=10,
    phase_before=0,
    phase_after=5π/2,
    payload="x90",
    conditional_on={c0}
  ),

  PulseEvent(
    event_id=2,
    kind="shift_phase",
    frame="f_drive",
    port=None,
    start=10,
    end=10,
    phase_before=5π/2,
    phase_after=3π,
    payload="π/2",
    conditional_on=∅
  ),

  PulseEvent(
    event_id=3,
    kind="delay",
    frame="f_drive",
    port=None,
    start=10,
    end=12,
    phase_before=3π,
    phase_after=7π/2,
    conditional_on=∅
  )
]
```

这个 lowered 结果体现了论文里的三个关键设计：

- 控制结构 `IfBit(c0, ...)` 被摊平成普通事件，并通过 `conditional_on={c0}` 记录反馈依赖
- `Play` 的真实启动时间是 `max(time[f_drive], port_time[p0]) = max(0, 4) = 4`
- `phase_after` 按总推进时间更新，所以 `Play` 的相位推进包含了等待共享端口的 `4 dt` 和真正播放波形的 `6 dt`

## 4. 验证总流程

论文中的统一验证入口可以概括为五步：

1. 对源程序做 well-formedness 检查
2. 用 oracle 执行源语义，得到参考状态
3. 运行 lowering，得到显式 schedule
4. 从 schedule 重建 compiled state
5. 检查三个 contract 组件：`PortExcl`、`FeedbackCausal_sched`、`FrameConsist`

这五步并不是重复做同一件事，而是在不同层面验证 lowering 的正确性。

## 5. 第一步：源程序 well-formedness

在论文里，`IfBit` 的合法性首先由源级检查器 `WF(P, C)` 保证，而不是留到 schedule 阶段再处理。

对于本例：

- `c0` 先由 `Acquire(f_meas, 4, c0)` 定义，因此“条件位已定义”成立
- `Acquire` 占用 `p0` 的时间区间是 `[0, 4)`
- 当遇到 `IfBit(c0, Play(f_drive, x90))` 时，guarded body 的真实启动时间是

```text
t_use = max(time[f_drive], port_time[p0]) = max(0, 4) = 4
```

- 与此同时，`Acquire` 使得

```text
cbit_ready(c0) = 4
```

因此：

```text
t_use >= cbit_ready(c0)
4 >= 4
```

所以这个 `IfBit` 在源级别是合法的。

如果这里不成立，比如 lowering 前的程序写成“先 `IfBit` 再 `Acquire`”，那么验证会直接在 `WF` 阶段失败，后续不会再继续。

## 6. 第二步：oracle 计算参考语义

well-formedness 通过后，oracle 按论文中的源级语义独立执行程序，得到期望的最终状态。

### 6.1 `Acquire(f_meas, 4, c0)`

- `start = max(time[f_meas], port_time[p0]) = max(0, 0) = 0`
- `end = 4`
- 更新后：

```text
time[f_meas] = 4
port_time[p0] = 4
cbit_ready[c0] = 4
phase[f_meas] = 0
```

### 6.2 `IfBit(c0, Play(f_drive, x90[dur=6]))`

在论文的 v0.4 语义中，`IfBit` 采用 taken-branch contract semantics。也就是说，只要源程序已经通过 `WF`，oracle 就把 body 当作会执行的语义包络来计算时间和相位。

对 `Play`：

- `start = max(time[f_drive], port_time[p0]) = max(0, 4) = 4`
- `end = 10`
- 总推进时间是

```text
end - time[f_drive] = 10 - 0 = 10
```

- 所以相位更新为

```text
phase[f_drive] += 2π · (1/8) · 10 = 5π/2
```

- 更新后：

```text
time[f_drive] = 10
port_time[p0] = 10
phase[f_drive] = 5π/2
```

### 6.3 `ShiftPhase(f_drive, π/2)`

```text
phase[f_drive] = 5π/2 + π/2 = 3π
time[f_drive] = 10
```

### 6.4 `Delay(2, f_drive)`

```text
time[f_drive] = 12
phase[f_drive] = 3π + 2π · (1/8) · 2 = 7π/2
```

因此 oracle 给出的最终参考状态是：

```text
f_meas:  time = 4,  phase = 0
f_drive: time = 12, phase = 7π/2
port_time[p0] = 10
cbit_ready[c0] = 4
```

## 7. 第三步：从 lowered schedule 重建 compiled state

接下来，验证器不直接拿 lowering 内部维护的状态做判断，而是从输出的 `PulseEvent` 列表重新构建 compiled state。

对本例，重建出来的关键信息是：

- `f_meas.time = 4`
- `f_drive.time = 12`
- `f_meas.phase = 0`
- `f_drive.phase = 7π/2`
- `occupancy[p0] = [(0, 4), (4, 10)]`
- `port_time[p0] = 10`
- `cbit_ready[c0] = 4`

这一步的作用是把“显式 schedule”重新转成一个便于检查的状态表示，从而让后续 checker 只依赖 lowered 输出本身，而不依赖 lowering 的内部实现过程。

## 8. 第四步：检查三个 contract 组件

### 8.1 PortExcl

`PortExcl` 要求同一个端口上的任意两个占用区间不能重叠。

本例中：

```text
occupancy[p0] = [(0, 4), (4, 10)]
```

因为这两个区间只是首尾相接，没有交叠，所以：

```text
PortExcl = true
```

如果 buggy lowering 把第二个事件错误地下沉到 `start=0`，那么区间会变成 `[(0, 4), (0, 6)]`，这时 `PortExcl` 会立即报错。

### 8.2 FeedbackCausal_sched

这个性质检查 lowered schedule 中所有带 `conditional_on` 的事件，是否真的等到了对应测量结果可用。

在本例中，第二个事件满足：

```text
conditional_on = {c0}
start = 4
cbit_ready(c0) = 4
```

因此：

```text
start >= cbit_ready(c0)
4 >= 4
```

所以：

```text
FeedbackCausal_sched = true
```

如果 buggy lowering 错误地把这个条件 `play` 提前到 `t=0`，那么虽然事件表面上还写着 `conditional_on={c0}`，但它已经违反了 feedback causality，会被这个 checker 捕获。

### 8.3 FrameConsist

`FrameConsist` 是论文里最关键的 correspondence check。它不是问“schedule 自己内部看起来像不像对的”，而是问：

“由 schedule 重建出来的最终 frame 时间和相位，是否与源程序语义独立计算出的结果一致？”

对本例：

```text
compiled_state(f_drive) = (time=12, phase=7π/2)
expected_from_source(f_drive) = (time=12, phase=7π/2)
```

`f_meas` 也一致，因此：

```text
FrameConsist = true
```

这一步特别重要，因为有些 lowering bug 不会造成端口冲突，也不会破坏反馈因果，但仍然会造成最终时间或相位漂移。例如：

- 漏掉 `ShiftPhase`，会导致 phase mismatch
- 插入额外 `Delay`，会导致 time mismatch
- 忽略共享端口等待，会导致 `f_drive` 的最终时间从 `12` 错成 `8`

这些错误都主要依赖 `FrameConsist` 来捕获。

## 9. 本例的最终验证结论

对这个例子，验证报告应当得到：

```text
well_formed      = true
port_exclusive   = true
feedback_causal  = true
frame_consistent = true
overall_ok       = true
```

也就是说，这个 lowering 结果同时满足：

- 源程序本身是可接受的
- lowered schedule 没有破坏共享端口互斥
- lowered schedule 没有破坏测量反馈因果关系
- lowered schedule 的最终时间和相位与源语义一致

## 10. 这个例子说明了什么

这个案例恰好展示了论文验证边界的核心思想：

- `WF` 负责在 lowering 之前排除非法的反馈用法
- `FeedbackCausal_sched` 负责检查 lowering 后的条件事件是否仍然尊重经典依赖
- `PortExcl` 负责检查 lowering 是否正确维护了共享硬件资源约束
- `FrameConsist` 负责检查 lowering 是否保留了源语义要求的最终时间与相位结果

因此，论文中的“形式化验证”不是单一的 pass/fail 断言，而是一个分层 contract：

源程序先合法，lowered schedule 再分别满足资源安全、反馈因果和源到目标的一致性。

## 11. 一句话总结

对这个例子来说，形式化验证的含义可以概括为：

“我们先证明这个 `IfBit` 在源程序里用得合法，再检查 lowering 生成的显式 pulse schedule 没有抢占共享端口、没有提前使用测量结果，并且最终得到的 frame 时间和 phase 与源语义完全一致。”
