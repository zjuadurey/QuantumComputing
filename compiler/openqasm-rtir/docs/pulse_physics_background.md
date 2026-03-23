# Pulse-Level 物理背景速记

这份笔记不是量子控制理论教程，而是给当前 `openqasm-rtir` 原型补最相关的脉冲层背景。
目标很明确：帮助理解 `port / frame / time / phase / feedback` 在规范里是什么意思，
以及它们为什么会进入我们的语义、checker 和 lowering contract。

---

## 1. 这个项目真正需要懂什么

当前原型关心的不是“波形如何精确驱动物理哈密顿量”，而是下面四类 correctness object：

- 资源：哪条物理通道在什么时间被占用
- 时间：指令真实从什么时候开始，什么时候结束
- 相位：frame 的 phase 如何随显式操作和时间流逝演化
- 反馈：测量结果什么时候可被后续条件动作使用

所以你需要的背景，不是完整的微波控制理论，而是 OpenPulse / OpenQASM 中：

- `port` 是什么
- `frame` 是什么
- `Play / Acquire / ShiftPhase / Delay` 语义各代表什么
- 为什么等待会推进时间，且可能推进 phase
- measurement-feedback 为什么必须显式建模 readiness / latency

---

## 2. Port 是什么

`port` 可以理解为一条物理控制或读出通道的抽象。
在同一时刻，同一个 `port` 不能无约束地承载多段互相冲突的 pulse/capture 活动。

对当前代码来说，这直接对应：

- 同一 `port` 上的 `Play` / `Acquire` 需要串行化
- `port_time[p]` 表示该端口最早何时重新可用
- `PortExcl` 检查的是 lowering 后 schedule 是否违反了这种资源约束

如果两个 frame 共享同一 `port`，那么第二个操作即使在“自己的 frame 局部时间”已经准备好了，也必须等到 `port` 空闲。

---

## 3. Frame 是什么

`frame` 不是简单的标签，而是一个挂在某个 `port` 上的参考系对象。
它通常携带：

- 所属 `port`
- 频率 `freq`
- 相位 `phase`
- 当前时间位置 `time`

在你现在的模型里，frame 的核心状态正好是：

- `time[f]`
- `phase[f]`
- 以及通过 `config.port_of[f]` 间接绑定到某个 `port`

这也是为什么这个项目里 `frame consistency` 是核心性质之一：后续 pulse 的物理意义依赖于 frame phase 是否正确。

---

## 4. Play / Acquire / Delay / ShiftPhase 各表示什么

### `Play(frame, waveform)`

表示在某个 frame 上播放波形。
在语义上，它至少做三件事：

- 占用该 frame 对应的 `port`
- 推进该 frame 的时间
- 推进该 frame 的 phase

如果 `port` 忙，则 `Play` 的真实开始时间不是单纯的 `time[f]`，而是：

```text
start = max(time[f], port_time[p])
```

### `Acquire(frame, duration, cbit)`

表示在某个 frame 上进行采集/测量，并在结束后让经典位结果变得可用。
语义上和 `Play` 类似，也会：

- 占用 `port`
- 推进时间
- 推进 phase
- 设置 `cbit_ready[c]`

### `Delay(duration, frame)`

表示 silence。
它推进 frame 时间，但不占用 `port`。

这正是为什么你们后来决定：

- `Delay` 改变 `time[f]`
- `Delay` 不改变 `port_time[p]`

### `ShiftPhase(frame, angle)`

表示显式相位平移。
它不推进时间，不占用 `port`，但会直接更新 `phase[f]`。

---

## 5. 为什么等待也会推进 phase

这是你们 v0.4 改动里最关键的物理建模选择之一。

如果一个 `Play` 因为共享 `port` 而被迫等待，那么 frame 的真实时间仍然在流逝。
只要你把 `time[f]` 解释成真实 elapsed time，那么 phase 也必须在这段等待期间继续演化。

当前代码采用的语义是：

```text
start = max(time[f], port_time[p])
end = start + d
total_advance = end - old_time[f]
phase[f] += 2π * freq[f] * total_advance
```

这里的 `total_advance` 包含两部分：

- stall：等待端口空闲的时间
- operation duration：真正执行 pulse 的时间

这样做的好处是：

- `time[f]` 真正表示现实中的流逝时间
- `FrameConsist` 公式不需要改，只要 `time[f]` 的定义正确即可
- shared-port wait 不会把相位语义搞乱

---

## 6. 为什么 feedback 要显式建模

measurement-feedback 的难点不在“有个 if”，而在：

- 测量结果不是瞬时可用的
- 结果需要在某个时间点之后才能被条件动作使用
- 硬件和控制系统可能还有额外 latency

在你们的最小模型里，这被抽象成：

- `Acquire(..., cbit)` 在结束时设置 `cbit_ready[cbit]`
- 条件动作的开始时刻必须满足 `start >= cbit_ready[cbit]`

v0.4 的设计把 feedback 分成两层：

- source side：`wellformedness` 预检查源程序是否合法
- schedule side：`check_schedule_causality(events)` 检查 lowering 产出的显式事件表

这比让 oracle 自动等待 `cbit_ready` 更适合当前论文叙事，因为它保留了 feedback 作为非平凡 contract 的角色。

---

## 7. 这些物理概念如何映射到当前代码

### 资源层

- `Config.port_of`
- `FrameState.port_time`
- `FrameState.occupancy`
- `check_port_exclusivity`

### 时间层

- `FrameState.time`
- `ref_semantics.step()`
- `lower_to_schedule()`
- `reconstruct_state()`

### 相位层

- `FrameState.phase`
- `ShiftPhase`
- `check_frame_consistency`

### 反馈层

- `FrameState.cbit_ready`
- `check_wellformedness`
- `check_schedule_causality`
- `conditional_on` in `PulseEvent`

---

## 8. 和论文写作最相关的几句话

你后面在论文里最稳的说法不是“我们模拟了完整 pulse physics”，而是：

1. 我们对一个受 OpenPulse 启发的最小 source-level core calculus 建模。
2. 该模型显式捕获资源受限下的真实开始时间、frame phase 演化以及 measurement-feedback readiness。
3. 我们验证的不是 waveform-to-unitary 物理正确性，而是 lowering 是否保持 time/resource/feedback/frame 这些 contract。

这三句话能把“形式化验证工作”和“控制物理工作”干净地区分开。

---

## 9. 推荐资料

优先顺序建议是：先看规范，再看动态反馈文档，最后看相邻研究。

### 官方规范

- OpenQASM Live Specification  
  https://openqasm.com/

- OpenPulse Grammar / Language page  
  https://openqasm.com/versions/3.0/language/openpulse.html

重点看这些关键词：

- Ports
- Frames
- Waveforms
- Play instruction
- Capture / Acquire
- Timing
- Phase tracking

### 动态反馈与硬件限制

- IBM Quantum: Classical feedforward and control flow  
  https://docs.quantum.ibm.com/guides/classical-feedforward-and-control-flow

- IBM Quantum: Hardware considerations and limitations for classical feedforward and control flow  
  https://docs.quantum.ibm.com/guides/dynamic-circuits-considerations

这两份资料最适合补“为什么 `cbit_ready` / latency / 条件动作时序必须单独建模”。

### 相邻研究

- PLanQC 2026: A Pulse-Level DSL for Real-Time Quantum Control with Hardware Compilation and Emulation  
  https://popl26.sigplan.org/details/planqc-2026-papers/14/A-Pulse-Level-DSL-for-Real-Time-Quantum-Control-with-Hardware-Compilation-and-Emulati

这篇更适合看“pulse-level toolchain / DSL / compilation / emulation”的系统视角。
它不是你们同题 baseline，但很适合补研究定位。

---

## 10. 建议的阅读顺序

如果你时间有限，按这个顺序就够了：

1. 先看 OpenPulse 页面里的 `Ports` / `Frames` / `Play` / `Timing`
2. 再看 IBM dynamic circuits 文档里的 classical feedforward
3. 最后回来看你们自己的 `ref_semantics.py`、`wellformedness.py`、`verify.py`

读完后，你应该能回答这三个问题：

- 为什么 shared-port wait 必须进入 source semantics
- 为什么 stall 期间 phase 也要继续演化
- 为什么 feedback legality 不能只看最终状态，必须看事件开始时刻

这三个问题正好就是当前 `openqasm-rtir` 的物理背景核心。
