# OpenPulse 关键语义摘要

> 来源：OpenQASM 3.0 规范 — OpenPulse Grammar
> 状态：规范仍在积极开发中，未来可能变化
> 启用方式：`defcalgrammar "openpulse";`

---

## 三个核心抽象

### 1. Port — 硬件资源接口

- 对硬件输入/输出的抽象，粒度由厂商决定
- 一个 qubit 可对应多个 port，一个 port 可作用于多个 qubit
- 通过 `extern port ...` 引入，编译期与厂商定义链接
- Port 被视为双向资源（可驱动也可观测）
- **形式化要点**：port 是资源互斥检查的基本单元

### 2. Frame — 时序/相位/频率的统一载体（最关键）

**组成**：

| 字段 | 含义 |
|------|------|
| port | 绑定的硬件端口 |
| frequency | 载波频率（可修改） |
| phase | 当前相位（可修改，且随时间自动累积） |
| time | 隐式维护的本地时钟 |

**初始化**：
- `newframe(port, frequency, phase)`
- 在 `cal` 中初始化：time 从全局 0 开始
- 在 `defcal` 中初始化：time 从该校准块被调度到的起点开始

**状态修改指令**（全部零时长，瞬时操作）：

| 指令 | 语义 |
|------|------|
| `set_phase(frame, θ)` | 将 phase 设为 θ |
| `shift_phase(frame, Δθ)` | phase += Δθ |
| `set_frequency(frame, f)` | 将 frequency 设为 f |
| `shift_frequency(frame, Δf)` | frequency += Δf |
| `get_phase(frame)` | 读取当前 phase |
| `get_frequency(frame)` | 读取当前 frequency |

**时间推进规则**（只有以下四种操作能推进 frame.time）：

| 操作 | 时间推进量 |
|------|-----------|
| `delay[d] frame` | frame.time += d |
| `play(frame, waveform)` | frame.time += waveform.duration |
| `capture(frame, ...)` | frame.time += capture_duration |
| `barrier(frame_set)` | 所有 frame 对齐到 max(frame.time) |

**相位自动累积**：
- 当 time 被推进 Δt 时，phase 隐式增加 `2π × frequency × Δt`
- 这是旋转坐标系的语义基础（virtual Z gate 等技巧依赖此机制）

**形式化要点**：frame 是项目最核心的形式化对象——它同时携带时间、频率、相位三个可变状态，且状态之间有耦合（时间推进会改变相位）。

### 3. Waveform — 时变包络

两种表示：
- **显式采样**：复数数组 `[c0, c1, c2, ...]`
- **参数化模板**：厂商提供的 extern 函数，如 `gaussian(duration, amp, sigma)`

组合操作（DSP 风格）：
- `mix(w1, w2)` — 逐点乘
- `sum(w1, w2)` — 逐点加
- `phase_shift(w, θ)` — 全局旋转
- `scale(w, a)` — 缩放

**形式化要点**：waveform 本身对验证来说主要提供 duration 信息；波形内容（形状/参数）与 unitary correspondence 相关，但那不是当前验证目标。

---

## 核心执行指令

### play(frame, waveform)

- 只能出现在 `defcal` 中
- 语义：用 frame 当前的 time/frequency/phase 调度波形
- 副作用：frame.time += waveform.duration; frame.phase 隐式累积
- 约束：波形时长必须能被对应 port 的采样率整除，否则编译报错
- **资源占用**：占用 frame 绑定的 port，持续 waveform.duration

### capture(frame, ...)

- 厂商定义的 extern 函数，最少需要一个 frame 参数
- 可选参数：duration, filter kernel 等
- 返回值：raw IQ / 处理后的值 / 判决后的 bit
- 副作用：frame.time += capture_duration; frame.phase 隐式累积
- **资源占用**：占用 frame 绑定的 port，持续 capture_duration

### delay[d] target

- 将 target（frame 或 qubit）的时间推进 d
- 不产生物理操作，但占用时间槽
- phase 会因时间推进而隐式累积

### barrier(targets)

- 将一组 frame/qubit 对齐到其中的最晚时刻
- `defcal` 入口对涉及的 frame 有隐式 barrier

---

## 关键约束与规则

### Frame collision（资源互斥）

> 如果同一个 frame 在两个并行的 cal/defcal 中被同时引用或调度，这是**编译期错误**。

这是规范层面已经表达的资源互斥语义——直接对应 `no_conflict` 检查。

### defcal 入口隐式 barrier

进入 defcal 时，对其中涉及的所有 frame 执行隐式 barrier，保证校准块入口的时序对齐。

### 零时长操作

set_phase / shift_phase / set_frequency / shift_frequency 是瞬时的（duration = 0）。如果硬件不支持瞬时修改，编译器应报错。

---

## 对形式化验证项目的直接映射

| OpenPulse 语义要素 | 可提取的可验证性质 |
|--------------------|--------------------|
| Port 资源占用（play/capture 期间） | **Port exclusivity** — 同一 port 上的 play/capture 不重叠 |
| Frame collision 规则 | **Frame exclusivity** — 同一 frame 不被并行 cal/defcal 同时使用 |
| Frame.time 推进规则 | **Timing consistency** — delay/play/capture 的时间推进与 barrier 对齐语义正确 |
| Phase 隐式累积 | **Frame-tracking consistency** — phase = initial + explicit_shifts + 2π × freq × elapsed_time |
| capture → classical result → if | **Feedback causality** — capture 完成后 classical result 才可用 |
| defcal 隐式 barrier | **Alignment correctness** — 入口处所有相关 frame 时间对齐 |

---

## 当前不需要形式化的部分

- Waveform 形状与目标酉变换的对应关系（量子控制理论问题）
- 具体硬件采样率 / 硬件属性模型（厂商相关）
- 通配 qubit 到脉冲资源的映射（规范自身的未决问题）

---

## 规范自身的未决问题（原文提及）

1. 如何把通配 qubit 映射到任意脉冲级资源
2. Frame/port 作为资源的时序语义是否已足够清晰
3. 硬件属性未来如何表示

这些开放点本身就是研究机会。
