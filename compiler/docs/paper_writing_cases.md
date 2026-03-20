# CCF-A 论文写作参考案例

这份笔记整理了几篇适合当前项目参考的 CCF-A 论文，重点不是“方法完全相同”，而是学习它们的：

- 行文结构
- 问题 framing
- 实验展示方式
- 图表组织方式
- 如何把“形式化定义 + 原型 + 验证结果”写成论文

当前项目更接近：

- `formal semantics`
- `reference oracle`
- `lowering / correspondence checking`
- `property-specific checkers`

所以最适合参考的，不是纯系统论文，而是“验证 / translation validation / semantics + implementation”风格的 PLDI / POPL / ASPLOS 论文。

## 最推荐先读的 3 篇

如果时间有限，先读这三篇：

1. **Alive2: Bounded Translation Validation for LLVM**  
   会议：PLDI 2021  
   主页：https://www.microsoft.com/en-us/research/publication/alive2-bounded-translation-validation-for-llvm/  
   论文页：https://www.pldi21.org/poster_pldi.30.html  
   作用：学习“translation validation / bug-finding / correctness pipeline”如何写。

2. **Giallar: Push-Button Verification for the Qiskit Quantum Compiler**  
   会议：PLDI 2022  
   主页：https://research.ibm.com/publications/giallar-push-button-verification-for-the-qiskit-quantum-compiler  
   公开摘要页：https://par.nsf.gov/biblio/10338372  
   作用：学习“量子编译验证论文”的定位、实验包装和 related work 写法。

3. **Towards Trustworthy Automated Program Verifiers: Formally Validating Translations into an Intermediate Verification Language**  
   会议：PLDI 2024  
   PDF：https://pm.inf.ethz.ch/publications/ParthasarathyDardinierBonneauMuellerSummers24.pdf  
   会议页：https://pldi24.sigplan.org/details/pldi-2024-papers/62/Towards-Trustworthy-Automated-Program-Verifiers-Formally-Validating-Translations-int  
   作用：学习“source semantics -> translation/lowering -> independent validation”这条线怎么写严谨。

---

## 详细案例

### 1. Alive2: Bounded Translation Validation for LLVM

- 会议：PLDI 2021
- 主页：https://www.microsoft.com/en-us/research/publication/alive2-bounded-translation-validation-for-llvm/
- 会议页：https://www.pldi21.org/poster_pldi.30.html

为什么值得看：

- 这是很经典的 `translation validation` 叙事。
- 论文把“为什么需要独立验证编译结果”讲得非常清楚。
- 实验部分非常适合模仿：覆盖范围、发现 bug 数量、开销、真实工具链接入。

你最该学的点：

- **引言怎么写**
  先讲 compiler correctness 的现实问题，再讲现有验证为什么不够，再引出 translation validation 的定位。
- **贡献怎么列**
  贡献非常集中，不发散。
- **实验怎么展示**
  常见结构是：
  - 覆盖了哪些 benchmark / tests
  - 找到了多少 bug
  - 运行时间和开销
  - 误报/漏报边界

对你这篇最有帮助的地方：

- 你现在也有两条路径：
  - source program -> oracle
  - source program -> lowering -> compiled schedule
- 这和 Alive2 的“独立验证优化后结果”在论文叙事上非常接近。

建议重点看：

- Abstract
- Introduction
- Evaluation 第一页
- 表格和 bug summary 的组织方式

---

### 2. Giallar: Push-Button Verification for the Qiskit Quantum Compiler

- 会议：PLDI 2022
- 主页：https://research.ibm.com/publications/giallar-push-button-verification-for-the-qiskit-quantum-compiler
- 摘要/公开条目：https://par.nsf.gov/biblio/10338372
- DOI 条目：https://www.osti.gov/biblio/1986256

为什么值得看：

- 这是最接近“量子编译验证”主题的 CCF-A 案例。
- 虽然它主要是 gate/circuit-level pass verification，不是 pulse-level timing semantics，但非常适合学习论文的包装和 positioning。

你最该学的点：

- **怎么写量子相关论文的 related work**
  把量子语言/编译器/验证工具放在一张清晰的地图里。
- **怎么定义 evaluation**
  用 `#verified passes / #versions / #detected bugs / runtime overhead` 这类指标，把形式化工作写得不空。
- **怎么把 prototype 说成系统**
  不是硬吹“大而全”，而是精确说明 scope 与能力边界。

对你这篇最有帮助的地方：

- 它能帮你写清楚：
  - 你的工作和量子 DSL / compiler 工程工作不同
  - 你的工作和 gate-level compiler verification 也不同
  - 你的贡献在 pulse/timing/feedback correctness model

建议重点看：

- Introduction
- Related Work
- Evaluation
- Conclusion 里对 scope 的描述

---

### 3. Towards Trustworthy Automated Program Verifiers: Formally Validating Translations into an Intermediate Verification Language

- 会议：PLDI 2024
- PDF：https://pm.inf.ethz.ch/publications/ParthasarathyDardinierBonneauMuellerSummers24.pdf
- 会议页：https://pldi24.sigplan.org/details/pldi-2024-papers/62/Towards-Trustworthy-Automated-Program-Verifiers-Formally-Validating-Translations-int
- 作者论文页：https://www.cs.ubc.ca/~alexsumm/papers/

为什么值得看：

- 这篇和你当前项目在“论文逻辑”上非常接近。
- 它关注的是 translation/front-end correctness，但最值得学的是：
  - formal semantics
  - translation path
  - independent validation
  - trust boundary / trusted base

你最该学的点：

- **如何写 lowering / translation correctness**
  尤其是“不是验证整个世界，只验证这条 translation path 的 soundness”。
- **如何写 trust model**
  哪部分是 trusted，哪部分是 validated，哪部分在 scope 外。
- **如何写 formal section 而不失去读者**
  它的 overview 和 formal 部分连接得比较自然。

对你这篇最有帮助的地方：

- 你现在的核心也正是：
  - source semantics
  - minimal lowering
  - correspondence verification

建议重点看：

- Overview
- Formal model / simulation argument
- Evaluation 里如何证明“现有实现确实被这条验证链覆盖”

---

### 4. Islaris: Verification of Machine Code Against Authoritative ISA Semantics

- 会议：PLDI 2022
- 官方条目：https://research-explorer.ista.ac.at/record/17502
- 开放条目：https://pure.mpg.de/pubman/item/item_3392162_2
- DOI：https://doi.org/10.1145/3519939.3523434

为什么值得看：

- 这篇特别适合学习“authoritative semantics / reference semantics”怎么写。
- 你的项目里也有 oracle / reference semantics，这篇会帮助你把“语义来源”和“权威基准”的话说得更像论文。

你最该学的点：

- **如何交代 semantic source of truth**
- **如何写 authoritative/reference 语义与实现验证之间的关系**
- **如何把形式化对象和实际系统对象连接起来**

对你这篇最有帮助的地方：

- 你可以学它怎么写：
  - 为什么 reference semantics 是可信基准
  - 为什么仍然需要另一条独立实现路径

建议重点看：

- Introduction
- Problem statement
- Threat model / trust model

---

### 5. OpenQASM 3: A Broader and Deeper Quantum Assembly Language

- 发表：ACM TQC 2022
- IBM 主页：https://research.ibm.com/publications/openqasm-3-a-broader-and-deeper-quantum-assembly-language

为什么值得看：

- 这篇不属于“验证论文主模板”，但很适合作为**量子语言 / pulse / timing / feedback 背景写法**的参考。
- 你的论文不是 DSL 论文，但需要把 `real-time classical control / timing / pulse` 讲清楚，这篇的 framing 很有帮助。

你最该学的点：

- **如何介绍 pulse / timing / classical feedback 背景**
- **如何给读者建立语义直觉**
- **如何写 running example**

对你这篇最有帮助的地方：

- 不是拿来抄 evaluation。
- 是拿来抄背景、overview 和语言层次介绍。

---

## 你这篇论文最适合模仿的结构

综合上面几篇，我建议你的论文骨架尽量接近下面这个顺序：

1. **Introduction**
- 问题：pulse-level quantum programs 缺少清晰 correctness model
- 难点：timing / resource / feedback / phase correspondence
- 方法：core subset + reference semantics + 3 properties + lowering + checkers
- 贡献：控制在 3 条

2. **Overview / Running Example**
- 用一个最小 feedback 程序
- 画出：
  - source program
  - oracle path
  - lowering path
  - compiled schedule
  - checker 在哪里检查什么

3. **Formal Model**
- abstract syntax
- config / state
- step rules
- 3 个 correctness properties

4. **Prototype / Verification Pipeline**
- reference semantics oracle
- lowering
- reconstruct
- independent checkers

5. **Evaluation**
- RQ1: 能否抓到目标 bug
- RQ2: 能否区分不同 bug 类别
- RQ3: 开销是否可接受
- RQ4: 在程序规模增大时表现如何

6. **Related Work**
- quantum DSL / compiler systems
- gate-level verified compilation
- translation validation / correspondence checking

7. **Conclusion**

---

## 你这篇论文适合出现的图和表

### 最重要的 4 张图

1. **System Overview**
- `PulseStmt -> ref_semantics`
- `PulseStmt -> lowering -> PulseEvent -> reconstruct -> checkers`

2. **Running Example Timeline**
- `Play / Acquire / Delay / conditional Play` 的时间线

3. **Bug Example Figure**
- 一个 buggy lowering 如何导致：
  - port conflict
  - early feedback
  - phase/time drift

4. **Evaluation Scaling Plot**
- 横轴：program size / number of frames / number of IfBit
- 纵轴：oracle time / lowering time / checker time

### 最重要的 3 张表

1. **Core Syntax / Semantics Summary**
- statement
- timing effect
- phase effect
- port occupancy
- classical effect

2. **Bug Specificity Matrix**
- 行：correct / drop_phase / extra_delay / reorder_ports / early_feedback
- 列：PortExcl / FeedbackCausal / FrameConsist
- 单元格：PASS / FAIL

3. **Runtime Table**
- benchmark/program
- #stmts
- #frames
- oracle time
- lowering time
- checker time

---

## 建议的阅读顺序

### 第 1 轮：只学结构

按这个顺序读：

1. Alive2
2. Giallar
3. PLDI 2024 那篇 translation validation

只看：

- Abstract
- Introduction
- Figure 1 / Overview
- Evaluation section 的第一个小节

目标：

- 学它们怎么讲故事
- 学它们怎么摆图和表

### 第 2 轮：带着你的题目去读

再回看：

- Islaris
- OpenQASM 3

目标：

- 学语义和背景怎么写
- 学“authoritative/reference semantics”怎么交代

---

## 对当前项目最实用的提醒

你现在不要试图把这些论文都“读懂到方法细节”。

更有效的方式是：

- 先把每篇的 **目录结构**、**图表样式**、**evaluation 组织方法**学过来
- 再把你自己的材料塞进类似的结构

换句话说，你现在读这些论文的主要目标不是：

- 学他们怎么证明

而是：

- 学他们怎么写

---

## 快速结论

如果只保留一句建议：

- **Alive2 学叙事和实验，Giallar 学量子验证论文怎么包装，PLDI 2024 那篇学 lowering/correspondence 怎么写严谨。**

如果只保留一个行动建议：

- **先照这三篇的风格，搭出你自己的论文大纲和图表清单，再继续补实验。**
