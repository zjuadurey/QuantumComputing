# Round 1 Code Plan: Lightweight Decoder Lab

## 1. 本轮工作的定位

这一轮工作的目标，不是实现一个 shared decoder service，也不是实现一个 FTQC runtime 内的治理系统。

这一轮只做一件事：

**先搭建一个轻量、可运行、可重复的 decoder lab。**

这个 lab 的作用是为后续研究提供一个真实代码底座，使我们在继续推进 shared decoder service 之前，先拥有：

- 已跑通的公开 QEC 工具链
- 已验证可工作的 decoder backend
- 可重复的正确率与时延测量脚本
- 可落盘、可分析的实验结果

换句话说，这一轮是“先把实验底座立起来”，而不是“先把研究系统搭完整”。

---

## 2. 本轮工作的核心目标

本轮只追求以下四个结果：

### 2.1 跑通公开工具链

建立最小工作流：

**Stim circuit generation → detector data / detector error model → decoder backend → logical error stats + latency stats**

### 2.2 得到最小可重复实验脚本

脚本应能在本地独立运行，并输出明确结果，而不是依赖复杂的 notebook 手工操作。

### 2.3 保留后续扩展所需的数据出口

虽然本轮不研究 service governance，但输出结果应尽量保留未来会用到的基础字段，例如：

- decoder backend
- circuit parameters
- batch size
- decode time
- logical error count
- logical error rate

### 2.4 控制问题规模

本轮最重要的约束不是“功能尽量全”，而是：

**不要让问题重新膨胀。**

---

## 3. 本轮明确不做什么

为了避免再次陷入“自己定义太多东西”的情况，本轮明确不做以下内容：

### 3.1 不做 shared decoder service 原型

本轮不实现：

- service classes
- SLO
- overload detection
- overload contract
- graceful degradation
- class isolation
- admission control
- reservation / throttling

这些内容属于后续阶段，而不是第一轮底座搭建工作。

### 3.2 不做 decoder pool simulator

本轮不实现：

- 多 decoder worker 竞争
- request queueing simulator
- decoder contention model
- event-driven runtime simulator
- decoder resource governance framework

### 3.3 不做重型 artifact 复现

本轮不使用以下重型路径作为主线：

- decoder-resources
- liblsqecc
- compiled IR trace generation
- Docker-heavy artifact workflow

这些东西后续如果需要 realism，再考虑引入。

### 3.4 不自己写 decoder

本轮不实现任何 decoder 算法本身。  
decoder backend 只使用现成工具。

---

## 4. 本轮的工具选择

本轮使用的工具栈应尽量轻量、公开、成熟。

### 4.1 主工具链

- **Stim**  
  用于生成 QEC 电路、detector error model、以及采样 detector data

- **PyMatching**  
  用作第一默认 decoder backend

- **sinter**  
  用于进行小规模 sweep、Monte Carlo 收集、以及结果落盘

### 4.2 可选工具

- **stimbposd**  
  如果加入第二个现成 decoder backend 的成本很低，则可作为可选补充

- **pandas / matplotlib**  
  仅用于结果整理和简单可视化

### 4.3 当前不使用的工具

- decoder-resources
- liblsqecc
- 自定义 runtime simulator
- 自定义 scheduling framework

---

## 5. 本轮代码仓库结构

建议第一轮代码结构尽量小：

```text
qec_decoder_lab/
  README.md
  requirements.txt
  scripts/
    00_smoke_test.py
    01_sinter_sweep.py
    02_measure_latency.py
    03_compare_backends.py      # optional
  data/
    raw/
    results/
  notebooks/
    inspect_results.ipynb       # optional