# Shared Decoder Service in a Single FTQC Runtime

## 1. 这个 idea 现在到底在研究什么

这个 idea 研究的不是“如何设计一个更好的 decoder scheduler”，也不是“如何做一个更大的 FTQC cloud / OS 平台”。

它研究的是：

**在单个 FTQC runtime 内，如何治理一个 shared decoder service。**

这里的 decoder 不应只被看成一个等待调度的资源池，而应被看成 runtime 内部的一个共享系统服务。不同 decode flows 对这个服务的需求不同，因此系统需要在固定 decoder budget 下，对不同 flows 提供不同程度的服务承诺。

这个问题的核心不是“下一次该先服务谁”，而是：

- 哪些请求必须保住
- 哪些请求可以延后
- 哪些请求在过载时可以被限流、合并或降级处理
- 系统如何让退化行为可预测，而不是全局失控

---

## 2. 问题的核心抽象

我们当前认定，这个问题最重要的三个核心概念是：

### 2.1 Service Class

不同 decode flows 并不是同质的。

在 runtime 内，至少存在不同延迟敏感度、不同重要性、不同可退化程度的 decode flows。  
因此系统不应只看到“一堆 pending decode requests”，而应先把这些 requests 放入不同的 **service classes** 中。

例如，一个非常粗糙但有用的抽象是：

- **critical class**：必须低延迟、低 miss，不能轻易牺牲
- **latency-sensitive class**：希望维持较低延迟，但必要时可有限退化
- **best-effort class**：允许较高延迟，允许在过载时承受大部分冲击

这里的重点不是类名，而是：  
**系统必须显式区分不同服务对象，而不能把所有 decode requests 当成统一队列。**

### 2.2 SLO

如果 decoder 是一个 shared service，那么系统需要明确“服务承诺”。

在这个问题里，SLO 不需要一开始设计得很复杂，但至少应体现：

- 某类 flow 的延迟目标
- 某类 flow 可接受的 miss rate
- 过载时某类 flow 允许退化到什么程度

也就是说，SLO 的作用不是让系统“更聪明”，而是让系统知道：

**在资源不够时，什么不能丢，什么可以退，退到哪里为止。**

### 2.3 Overload Contract

过载不是异常情况，而是系统必须显式处理的正常运行状态之一。

因此系统需要定义过载时的 contract，例如：

- critical 类继续优先受保护
- latency-sensitive 类进入有限退化模式
- best-effort 类可能被限流、延后、合并，甚至拒绝进入

换句话说，这个问题不是单纯“在过载下还能不能调得更好”，而是：

**系统能否在过载时保持有边界、可解释、可预测的退化行为。**

---

## 3. 哪些东西是核心，哪些只是手段

### 3.1 核心

这个 idea 的核心是下面这些问题定义层的对象：

- service classes
- SLO
- overload contract
- class-level isolation
- predictable degradation

### 3.2 手段

下面这些不是核心问题本身，而只是可能的实现手段：

- scheduling
- reservation
- admission control
- throttling
- isolation mechanisms
- graceful degradation mechanisms

这点非常重要。

如果整个系统最后只剩下“某种更好的优先级排序策略”，那就已经退化成了一个 **scheduler benchmark**；  
如果整个系统不断向外扩展到 magic-state、placement、跨节点编排、quantum cloud resource management 等更大范围，那就已经偏离了当前问题层级。

当前阶段必须始终把讨论拉回：

**single FTQC runtime 内的 SLO-aware shared decoder service**

---

## 4. 这个 idea 目前不研究什么

为了防止问题摊得过大，当前明确排除以下方向：

### 4.1 不研究“decoder scheduling in isolation”

也就是说，我们不把问题定义成：

- 给定一堆 decode tasks，设计一个更好的排序策略
- 比较 RR / EDF / priority / heuristic score 谁更优
- 只关注平均延迟、最大 backlog、吞吐量

这些东西可以作为实现组件或 baseline，但不是问题本体。

### 4.2 不研究更大的 FTQC cloud / OS 平台

当前不讨论：

- 多程序共享整个 FTQC 平台
- 跨节点资源编排
- magic-state factory、placement、routing、decoder 的统一全局管理
- 一个完整 quantum OS 的设计

这些范围太大，会使问题边界迅速失控。

### 4.3 不在第一阶段研究真实完整 runtime

当前也不打算第一步就实现一个完整运行时系统。  
第一阶段的目标是做一个 **最小可运行、可观测、可迭代的代码原型**，先把问题压缩到最小闭环。

---

## 5. 当前最合理的代码推进方式

## 5.1 先不要从重型系统 artifact 起步

虽然现有如 `decoder-resources` 这样的仓库能提供 compiled IR 驱动的资源与调度模拟，但它依赖较重，起步成本高，而且会过早把工作带入“大型 trace-driven artifact 复现”。

当前阶段更适合先做一个 **轻量 decoder lab**，目的是先把公开 QEC 工具链跑通，拿到真实 decoder backend 的实验底座。

### 5.2 第一轮先做 lightweight decoder lab

第一轮工作的目标不是 service governance，而是先建立最小实验底座：

- Stim 生成 QEC 电路 / detector error model
- PyMatching 作为现成 decoder backend
- sinter 做 sweep 与结果落盘
- 可选加入第二个现成 decoder backend（如 stimbposd）

第一轮只回答下面这些问题：

- 公开工具链能否稳定跑通
- decoder backend 的正确率和延迟如何测量
- 最小实验脚本结构应该长什么样
- 哪些结果字段值得保留，便于后续升级成 shared service 问题

### 5.3 第一轮不做什么

第一轮明确不做：

- decoder pool simulator
- service class implementation
- overload controller
- SLO enforcement
- admission / throttling / reservation logic
- 重型 trace-driven runtime prototype

因为这些东西一旦一起做，问题会立刻重新摊大。

---

## 6. 未来系统原型的大致方向

在第一轮轻量 decoder lab 跑通后，后续系统原型可以在其外面只包一层很薄的治理逻辑，而不是重写整个系统。

后续最可能新增的层次是：

### 6.1 Request Replay Layer

把 decoder backend 从“单次离线调用”变成“请求流驱动”的形式：

- request arrival
- request completion
- backend contention
- queueing and waiting

### 6.2 Class Tagging Layer

开始引入最小的 service classes：

- critical
- latency-sensitive
- best-effort

### 6.3 Per-Class Metrics Layer

开始按类记录：

- latency
- miss rate
- tail behavior
- overload spillover
- degradation activation

### 6.4 Thin Governance Layer

在确认 workload 和指标稳定后，再尝试非常薄的治理机制，例如：

- reserved capacity
- guarded admission
- request coalescing
- bounded degradation

注意：  
这里的扩展顺序必须始终坚持“先薄包一层，再看是否真的需要继续加”，而不是一开始就发明一套完整 runtime 架构。

---

## 7. 当前阶段的判断标准

当前阶段判断一个设计是否合理，不看它“概念上是否完整”，而看它是否满足以下条件：

- 是否依赖公开、现成的工具链
- 是否能在 3–5 天内形成最小闭环
- 是否有自然 baseline
- 是否有清晰、可重复的输出结果
- 是否避免把问题重新摊大
- 是否保持在 **single FTQC runtime 内 shared decoder service** 这个层级上

如果一个设计需要同时自定义：

- workload
- decoder model
- runtime semantics
- scheduler
- metrics
- failure model

那基本就说明它已经偏大了。

---

## 8. 当前阶段的工作原则

当前阶段遵循三条原则：

### 8.1 先复现，再理解，再薄改

不要直接发明新系统。  
先把现成工具链跑起来，再看哪些地方真的需要自己写。

### 8.2 先搭底座，不先做治理

先有真实 decoder backend 与实验数据，再讨论 shared service 的治理层。  
否则问题会停留在抽象空转。

### 8.3 始终防止滑回两个错误方向

第一个错误方向是：

**退化成纯 scheduler benchmark**

第二个错误方向是：

**膨胀成 FTQC cloud / OS 级的大系统问题**

当前所有代码工作都应该围绕这两条红线来控制范围。

---

## 9. 一句话总结

这个 idea 的本质是：

**在单个 FTQC runtime 内，把 decoder 视为一个 shared service，在固定 decoder budget 下，为不同 service classes 提供不同程度的服务承诺，并在 overload 下实现可预测、有边界的退化。**

但在代码推进上，当前阶段不直接实现这一整套系统。  
当前更合理的策略是：

**先搭建一个轻量、真实、公开工具驱动的 decoder lab，作为后续 shared decoder service 原型的底座。**