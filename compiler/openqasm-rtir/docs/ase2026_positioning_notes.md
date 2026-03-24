# ASE 2026 定位备忘

## 贡献应该怎么写

更准确的表述，不是“解决了量子电路到底层物理状态的正确性验证”，而是：

“提出了一个面向 pulse-level lowering 的形式化正确性契约，用来验证源级脉冲控制程序被编译成显式硬件时序后，是否仍然保持时间、资源和反馈语义。”

第一个说法太大，也容易被审稿人追问：
你们并没有建模量子态演化、哈密顿量、校准误差、控制波形对物理态的真实作用，所以很难说是在验证“到底层物理状态”的正确性。你们现在验证的是 compilation/lowering correctness，不是 physics-level correctness。

第二个方向更接近你们的真实贡献，但我会把“新的量子模拟语言”改成更准的说法：
不是“模拟语言”，而是“新的 pulse DSL / OpenQASM/OpenPulse 风格脉冲语言或中间表示”。

我建议把贡献写成下面这几个点。

### 推荐的总贡献表述

我们的工作为 pulse-level quantum programs 提供了一个可执行、可检查的 lowering correctness framework，使得编译器可以在显式时序层面验证其是否保留了源程序要求的反馈因果、共享资源约束以及 frame-level time/phase 行为。

### 可以展开成 3-4 条贡献

1. 一个最小但能表达关键硬件约束的 pulse-control core formalization。  
它覆盖了 `Play`、`Acquire`、`ShiftPhase`、`Delay` 和 `IfBit`，并显式建模 shared-port serialization、classical feedback 和 phase evolution。

2. 区分 source admissibility 和 compiled-schedule correctness。  
也就是把“源程序是否合法”与“lowered schedule 是否正确”拆开，这一点其实很重要，也很像你们工作的理论贡献。

3. 一个 lowering contract with 3 properties。  
它不是笼统地说“编译正确”，而是拆成了 `WF`、`PortExcl`、`FeedbackCausal_sched` 和 `FrameConsist` 这几个可独立检查的性质。

4. 一个 executable prototype with independent oracle/checkers and seeded faults showing detection。  
这说明你们的合同不是纯定义，而是真的能用于验证 pulse compiler / DSL lowering pass。

### 如果想强调应用场景

可以这样说：
这项工作可作为新 pulse DSL、OpenQASM/OpenPulse 风格前端、或量子控制编译器后端的验证边界，用来检查从源级脉冲程序到显式硬件时序表示的 lowering 是否正确。

### 我不建议的表述

- “验证量子电路到底层物理状态的正确性”
- “解决了量子硬件语义正确性问题”
- “证明量子程序在物理层面正确执行”

这些都比你们当前实际完成的内容大很多。

### 一句话版本

如果只保留一句，我会推荐这句：

“我们提出了一个面向 pulse-level quantum programs 的形式化 lowering contract，用于验证源级脉冲控制程序在编译为显式硬件时序后，是否保持反馈因果、共享资源约束以及 frame-level 时间与相位语义。”

## ASE 2026 是否适合

适合，但要看你们怎么讲。我的判断是：

“题目方向适合 ASE 2026 的 `Formal Aspects of Software Engineering` area；当前贡献有希望，但作为 `Research Paper` 还偏悬，关键在于你们能不能把它讲成一个自动化软件工程问题，而不只是量子领域里的一个小型形式化原型。”

ASE 2026 官方 Research Track 明确把 `Formal methods and model checking`、`Programming languages`、`Domain-specific or specification languages`、`Software validation and verification` 列为研究方向，而且 Technical Research Papers 强调“automating software development or automating support to users engaged in software development activities”以及 significance/novelty/soundness/verifiability。[ASE 2026 CFP](https://conf.researchr.org/track/ase-2026/ase-2026-research-track) 里还写了 10+2 页限制、强制 data-availability statement，`Paper Submission` 截止是 `Thu 26 Mar 2026 AoE`，按中国时区大约是 `2026-03-27 19:59`。ASE 最近也确实收 formal verification/PL 方向的论文，比如 2025 有 formal methods sessions 和 compiler/verification 类工作。[ASE 2025 papers](https://conf.researchr.org/track/ase-2025/ase-2025-papers)

你们这篇最合适的定位，不是“验证量子电路到底层物理状态”，而是：
“为 pulse-level DSL / IR 的 lowering 提出一个可执行的形式化 correctness contract，并实现自动化验证管线来检查 lowering 是否保持反馈因果、共享资源约束和 frame-level time/phase semantics。”
这个表述和 ASE 更对口，因为它强调：

1. `DSL/IR`
2. `compiler/lowering correctness`
3. `automated verification`
4. `tool-supported checking`

但“贡献是否足够”我会诚实地说：目前是“有潜力，但需要再补强一截”。最可能被 ASE reviewer 质疑的不是形式化本身，而是：

1. `Significance`：为什么这不只是量子领域里的一个小 DSL case study？
2. `Novelty`：相对已有 compiler verification / DSL verification / schedule validation，新的点到底在哪里？
3. `Evidence`：5 个 seeded fault families 和 curated examples 可能显得偏小，像 prototype validation，不太像 ASE 喜欢的强实证。

如果要投 ASE Research，我建议把贡献主轴收敛成这三条：

1. 一个把 source admissibility 和 compiled-output correctness 分离的 pulse-level lowering contract。
2. 一个自动化、独立实现的验证管线，而不是只给出语义定义。
3. 一组能系统区分 fault families 的检查器，说明不同 lowering bug 如何被不同 contract components 捕获。

同时最好再补至少两类增强：

1. 更强的实验：更多程序、更多 mutation/fault injection、最好接到真实或半真实的 pulse workloads。
2. 更强的 SE framing：把问题说成“新兴 DSL/compiler backend 的 lowering verification”，而不是“量子物理”。

如果来不及把实验做厚，我会把判断调成：
“作为 ASE 2026 Research Paper 偏冒险，但不是没机会；如果能显著补实验和 framing，可以冲。”

“如果到截稿前证据还是主要是 curated examples + seeded bugs，那更像一个不错的 formal methods / tool paper 雏形，而不是很稳的 ASE research hit。”

一句话结论：
能投，scope 是对的；但要把它包装成“自动化验证 DSL lowering 的软件工程贡献”，并补强实证，否则 reviewer 很容易觉得 contribution 窄、evaluation 轻。

## v0.5 证据结构建议

如果进入 v0.5，我建议把证据结构明确写成三层：

1. **Formal evidence**
   - core calculus
   - source semantics
   - lowering contract
   - independent checkers

2. **Internal empirical evidence**
   - seeded fault families
   - end-to-end tests
   - checker discrimination

3. **External corroboration**
   - `Qiskit Dynamics` 作为 pulse-level supporting evidence
   - 后续可补 backend/resource/timing-facing 对照

这里最关键的 framing 是：

- 你们自己的 pulse-level lowering / contract / checker pipeline 是 **core idea**
- `FullContract` 是 **主验证器**
- `Qiskit Dynamics` 是 **主要外部实验平台**

也就是说，要明确区分两个维度：

1. **谁定义什么叫正确**
2. **谁最能作为实验上的外部见证**

对本文来说：

- 在“定义 correctness”这个维度上，主角是 `FullContract`
- 在“提供主要外部实验说服力”这个维度上，主角可以是 `Qiskit Dynamics`

所以 `Qiskit Dynamics` 可以是主实验里的 centerpiece，
但仍然不能被写成主验证器或语义定义者。

更具体地说，推荐在论文里写成：

> FullContract remains the primary verifier of lowering correctness.
> Our pulse-level lowering contract and checker pipeline constitute the core idea.
> We use Qiskit Dynamics as the primary external experimental witness,
> providing independent, community-recognized evidence that selected
> contract-detected faults also induce observable deviations.

同时建议在 scope / threats 里补一句：

> Qiskit Dynamics is used for corroboration, not as the semantic oracle.

这会比“我们用 Qiskit Dynamics 验证我们的方法是对的”稳得多。

来源：
- ASE 2026 Research Track CFP: https://conf.researchr.org/track/ase-2026/ase-2026-research-track
- ASE 2025 papers/program: https://conf.researchr.org/track/ase-2025/ase-2025-papers
