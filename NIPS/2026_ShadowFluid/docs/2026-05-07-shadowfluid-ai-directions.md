# 2026-05-07 ShadowFluid 的 AI 化方向收敛

## 核心原则

当前实验已经说明一件事：

- `ShadowFluid` 的主干演化与任务对齐能力非常强。
- 直接让 AI 去“再学一个近似演化器”或“再学一套新字典动力学”，很容易变成一个更脆弱的 surrogate。

因此后续 AI 化的总原则应明确为：

**AI 用来弥补 `ShadowFluid` 的短板，不用来替代 `ShadowFluid` 已经很强的部分。**

这意味着：

- 不优先继续做 end-to-end learned dictionary dynamics
- 不优先让 AI 直接重做主演化方程
- 优先让 AI 学：
  - 截断误差如何补
  - 什么时候当前配置会失效
  - 在预算受限时如何自动配 solver

---

## 候选方向 A：ShadowFluid + Learned Residual Closure Correction

### 一句话概念

保留原始 `ShadowFluid` 的固定字典和结构化演化，AI 只学习：

**“在当前字典/closure 下，任务相关 observable 还差了多少，需要补什么 residual。”**

### 为什么最适合

这条线最符合“补短板，不替代强项”：

- `ShadowFluid` 继续负责主干演化
- AI 不重做 dynamics，只学习 missing closure effect
- 学习目标从“再学一个 solver”变成“修正一个很强 solver 的剩余误差”

这也是目前最可能真正提升结果的一条线。

### 建议的学习对象

优先做 **observable-space residual**，而不是 full-state residual。

最推荐的 v1 形式：

1. `ShadowFluid` 先给出
   - `lowfreq_base(t)`
   - `rho_base(t)`
   - `energy_base(t)`
   - `leakage`
2. 小网络预测
   - `delta_lowfreq(t)`
   - 可选 `delta_energy(t)`
3. 最终输出
   - `lowfreq_corr(t) = lowfreq_base(t) + delta_lowfreq(t)`
   - `rho_corr(t)` 由 `lowfreq_corr(t)` 做固定逆变换得到
   - `energy_corr(t) = energy_base(t) + delta_energy(t)` 或由 `lowfreq_corr` 派生

### 为什么先做 observable-space residual

- 现在 pipeline 已经以 `rho / lowfreq / energy` 为主指标
- 直接修 `lowfreq` 最贴合任务定义
- 比修 full wavefunction 更容易
- 不会把问题重新变成“学整个动力学”

### 输入给 residual 网络的特征

建议输入：

- `lowfreq_base(t)`
- `energy_base(t)`
- `time t`
- `hamiltonian_features`
- `rho0` 的小编码
- `potential_field` 的小编码
- `commutator leakage`

推荐结构：

- 一个小型 context encoder 编码 `rho0 + potential_field`
- 一个 time-wise MLP / tiny transformer / GRU 预测 `delta_lowfreq(t)`
- 参数量应明显小于 `FNO`

### 关键正则

这条线最重要的不是大模型，而是把 residual 约束成“只在需要时才修”：

1. residual shrinkage
   - `||delta_lowfreq||^2`
2. leakage-aware correction
   - leakage 小时，鼓励 residual 也小
3. optional consistency
   - `delta_energy` 与 `delta_lowfreq` 派生的能量变化一致

### 最小实验目标

如果这条线成功，应该看到：

1. 在 `ID` 上至少不弱于 `ShadowFluid`
2. 在 `OOD alpha / OOD structure` 上优于原始 `ShadowFluid`
3. 或者在更小的 reference budget 下，达到接近原始 `ShadowFluid` 的精度

第三点尤其重要，因为它更能说明：

**AI 学到的是 missing closure correction，而不是无意义的 overfit。**

### 代码实现复杂度

中等，且与当前代码最兼容。

最小改动路径：

1. 扩展 `FixedShadowFluidModel`
   - forward 返回 `lowfreq_base`
2. 在 `runner/metrics` 中支持 `pred["lowfreq"]`
   - 不再强制从 `rho` 反推 lowfreq
3. 新增模型
   - `shadowfluid_residual`
4. 加一个固定逆变换
   - `lowfreq -> rho`
5. 增加配置与训练脚本入口

### 风险

- 如果当前 `ShadowFluid` 已经非常接近最优，残差空间可能很小，提升幅度不会很大。
- 但即使如此，这条线仍然是“最合理的 AI 化”，因为它明确对准了短板。

---

## 候选方向 B：ShadowFluid + Adaptive Configuration Policy

### 一句话概念

不让 AI 学 residual，而让 AI 学：

**“给定 Hamiltonian / initial condition / budget，应该如何配置 `ShadowFluid`。”**

可学习的配置包括：

- `K0`
- `reference_budget`
- `R_hops`
- `candidate_pool_max`
- 是否需要更大 closure

### 为什么有价值

这条线更像 `amortized solver policy`：

- `ShadowFluid` 仍然是主求解器
- AI 负责在预算受限时自动选更合适的配置
- 很适合讲 accuracy-vs-cost tradeoff

### 适合回答的问题

比如：

- 在相同预算下，能否比固定配置更准？
- 在相同精度下，能否用更小 budget？
- 是否能自动识别“哪些 Hamiltonian 需要更大字典”？

### 训练方式

最简单的方式不是强化学习，而是监督式 policy learning。

先离线构造标签：

- 对小规模样本枚举或 sweep 若干配置
- 用 oracle 或近似最优配置做 supervision

然后训练：

- 输入：`hamiltonian_features`, `rho0/potential encoder`, leakage proxy
- 输出：配置类别或连续 budget

### 代码实现复杂度

中等偏高。

难点不是模型，而是：

- 需要构造配置标签
- 需要定义 cost/accuracy objective
- 需要把实验重点转成“精度-预算权衡”

### 风险

- 它更像“系统优化/元配置学习”
- 如果论文主目标是提精度而不是提效率，这条线会显得偏题

---

## 两个方向的比较

### 方向 A：Residual Closure Correction

优点：

- 最直接对准 `ShadowFluid` 的误差来源
- 最符合当前实验暴露出的痛点
- 最容易在现有 pipeline 上实现
- 最容易真正提升 accuracy

缺点：

- 若 `ShadowFluid` 已接近天花板，提升可能有限

### 方向 B：Adaptive Configuration Policy

优点：

- AI 角色更清晰
- 更容易讲“自动化”和“预算约束”
- 不破坏 solver 主体

缺点：

- 更偏系统/配置问题
- 未必直接提升当前主指标

---

## 推荐顺序

### 首选

**方向 A：ShadowFluid + Learned Residual Closure Correction**

这是下一步最值得直接落代码的方案。

### 次选

**方向 B：ShadowFluid + Adaptive Configuration Policy**

如果 residual correction 提升非常有限，再切到这条线。

---

## 2026-05-06 晚间实现记录：Residual v1 原型

已经按方向 A 落了一版最小实现：

- 新模型：`shadowfluid_residual`
- 配置：`code/configs/shadowfluid_residual_exp32.yaml`
- 训练入口：`code/scripts/train_shadowfluid_residual.py`

### v1 设计

- 保留 `ShadowFluid` 作为底层主干求解器
- 在更小的 base observable 空间上先做 rollout
  - `base_K0 = 4.0`
  - `reference_budget = 6`
  - `R_hops = 1`
- 小网络只预测
  - `delta_lowfreq(t)`
  - `delta_energy(t)`
- 最终输出：
  - `lowfreq_corr = lowfreq_base + delta_lowfreq`
  - `rho_corr` 由固定逆变换得到
  - `energy_corr = energy_base + delta_energy`

### v1 的真实结果

先看和黑盒 baseline 的对比：

- `FNO`
  - ID density: `0.2928`
  - OOD-alpha density: `0.2966`
  - OOD-structure density: `0.3325`
- `ShadowFluid + residual v1`
  - ID density: `0.2068`
  - OOD-alpha density: `0.1804`
  - OOD-structure density: `0.2036`

这说明：

- 这条 residual 线第一次稳定压过了 `FNO`
- 并且 low-pass energy 也明显更小

但更关键的公平对照是：

- `ShadowFluid base-K4 (no residual)`
  - ID density: `0.1213`
  - OOD-alpha density: `0.1267`
  - OOD-structure density: `0.1426`
- `ShadowFluid + residual v1`
  - ID density: `0.2068`
  - OOD-alpha density: `0.1804`
  - OOD-structure density: `0.2036`

这说明：

**v1 residual 还没有真正补强 under-budget ShadowFluid，反而在当前主指标 density 上把它拉差了。**

### 额外诊断：残差方向是否有用

对已训练好的 residual v1 做了一个混合系数扫描：

- `alpha = 0`：只用 base solver
- `alpha = 1`：用完整 residual correction
- `0 < alpha < 1`：线性插值

ID test 上的趋势是：

- density 随 `alpha` 单调变差
  - `alpha = 0.0`: `0.1213`
  - `alpha = 0.5`: `0.1383`
  - `alpha = 1.0`: `0.1801`
- energy 随 `alpha` 单调变好
  - `alpha = 0.0`: `0.00342`
  - `alpha = 0.5`: `0.00195`
  - `alpha = 1.0`: `0.00054`

因此当前最准确的技术判断是：

**v1 residual 学到的修正方向不是“完全没用”，而是更偏向 energy / spectral correction，而不是当前最重要的 density correction。**

### 这意味着什么

- “AI 补短板” 这个大方向仍然成立
- 但 v1 版本的 loss 与 correction target 还没有对准我们真正想保住的主指标

### 下一步更合理的修正

优先级最高的不是换大模型，而是让 residual 更保守、更 density-first：

1. 降低 `w_spectral` 与 `w_energy`
2. 提高 `w_residual`
3. 降低 `residual_scale`
4. 显式加入 correction gate / interpolation coefficient
5. 只在高-leakage 区域允许更强 correction

也就是说，下一步不是放弃 residual，而是把它从“主动改写解”改成“谨慎微调 base solver”。

### 2026-05-06 深夜更新：Residual v2（density-first, conservative）

已经按上面的判断又试了一版更保守的 residual：

- 配置：`code/configs/shadowfluid_residual_densityfirst_exp32.yaml`
- 关键变化：
  - `residual_scale = 0.1`
  - `w_spectral = 0.2`
  - `w_energy = 0.0`
  - `w_residual = 0.05`
  - residual head 默认零初始化，保证训练起点严格等于 base solver

结果：

- `ShadowFluid base-K4 (no residual)`
  - ID density: `0.1213`
  - OOD-alpha density: `0.1267`
  - OOD-structure density: `0.1426`
- `ShadowFluid + residual v2`
  - ID density: `0.1194`
  - OOD-alpha density: `0.1241`
  - OOD-structure density: `0.1389`

这说明：

**当 residual 被约束成“小而保守的 density-first 修正”时，它终于开始在同预算下稳定补强 base solver。**

而且它仍然显著强于当前的纯神经 baseline：

- `FNO`
  - ID density: `0.2928`
  - OOD-alpha density: `0.2966`
  - OOD-structure density: `0.3325`

因此到这一版为止，方向 A 已经从“概念上合理”走到了“实验上开始成立”：

- 不是替代 ShadowFluid
- 而是在一个受限 budget 的 ShadowFluid 上做小修正
- 并且这种修正在主指标上出现了稳定增益

### 2026-05-06 更深一层的 probe：什么约束最能放大 AI 增益？

又做了两组额外探针：

1. 只把 `reference_budget` 从 `6` 压到 `4`
2. 把 `base_K0` 从 `4` 压到 `3`

#### 结论一：`reference_budget 6 -> 4` 基本不构成真正瓶颈

- `ShadowFluid base-K4, budget6`
  - ID density: `0.1213`
- `ShadowFluid base-K4, budget4`
  - ID density: `0.1213`

这说明当前这个设置下，`reference_budget` 还没有真正卡住底层 solver。

#### 结论二：`base_K0 4 -> 3` 会明显削弱底层 solver

- `ShadowFluid base-K3`
  - ID density: `0.2225`
  - OOD-alpha density: `0.2389`
  - OOD-structure density: `0.2630`

这比 `base-K4` 明显更差，因此更适合检验 AI residual 是否真的在补“observable truncation”的短板。

#### 结论三：在 `base_K0 = 3` 下，residual 增益更清楚

- `ShadowFluid base-K3`
  - ID density: `0.2225`
  - OOD-alpha density: `0.2389`
  - OOD-structure density: `0.2630`
- `ShadowFluid + residual density-first (K3)`
  - ID density: `0.2083`
  - OOD-alpha density: `0.2205`
  - OOD-structure density: `0.2447`
  - long-rollout density: `0.2083`

这说明：

**AI 增益在“observable 空间更受限”的设定下更明显，而不是在 reference budget 这种当前未激活的约束上更明显。**

这其实非常符合最初的设计直觉：

- residual correction 最应该补的是 truncation / closure 不足
- 而不是去修一个其实已经没怎么受 budget 影响的 solver

从 long-rollout 曲线看，这个改进也不是只发生在单一时刻：

- `base-K3 density curve`
  - `[0.1362, 0.2518, 0.2651, 0.2475, 0.2120]`
- `residual-K3 density curve`
  - `[0.1356, 0.2356, 0.2430, 0.2271, 0.2003]`

也就是说，改进是沿整个 rollout 分布展开的，而不是只在最后一个时间点偶然更好。

### 当前最值得继续的实验主线

如果下一步要把这条方向做成更扎实的实验故事，最优先的是：

1. 继续围绕 `base_K0=3` 或类似更紧的 observable budget 做实验
2. 对 `residual density-first` 跑多 seed
3. 再看是否需要更进一步加入 leakage-gated correction

### 2026-05-06 深夜：`base_K0=3` clean multiseed + leakage gate 结果

围绕 `base_K0=3` 又补了两组 clean 3-seed 短程实验：

- `shadowfluid_residual_densityfirst_k3_short`
- `shadowfluid_residual_densityfirst_leakgate_k3_short`

对应结果表：

- `code/results/comparison_tables/exp32_k3_multiseed_compare.csv`

#### 无门控 density-first residual（3 seeds）

- ID density: `0.20830 ± 0.00017`
- OOD-alpha density: `0.22063 ± 0.00049`
- OOD-structure density: `0.24486 ± 0.00079`

对照底层 solver：

- `base-K3`
  - ID density: `0.22251`
  - OOD-alpha density: `0.23887`
  - OOD-structure density: `0.26296`

这说明：

**在更紧的 observable budget 下，conservative residual correction 的增益是稳定的，而且方差很小。**

#### leakage-gated residual（3 seeds）

- ID density: `0.21191 ± 0.00049`
- OOD-alpha density: `0.22354 ± 0.00066`
- OOD-structure density: `0.24324 ± 0.00094`

对比无门控版：

- ID：略差
- OOD-alpha：略差
- OOD-structure：略好

因此当前最准确的判断是：

**leakage gate 还没有成为明确更优的主版本。**

它有一点点把修正重心推向更难的 structure OOD，但整体上还没有超过简单的 conservative density-first residual。

### 到目前为止最稳的实验结论

1. AI 最适合补 `ShadowFluid` 的 observable truncation 短板，而不是替代主演化。
2. 这种增益在 `base_K0=3` 这类真正受限的设定下更明显。
3. 当前最稳的主线是：
   - `ShadowFluid + conservative density-first residual`
4. `leakage-gated residual` 值得保留为一个次级 ablation / follow-up variant，但还不适合顶替主版本。

### 2026-05-06 更进一步：`base_K0=2` clean multiseed

为了验证“越受限，AI 增益越大”是否真的成立，又补了 `base_K0=2` 的 clean 3-seed：

- `shadowfluid_residual_densityfirst_k2_short_multiseed`

对应汇总表：

- `code/results/comparison_tables/exp32_budget_curve_summary.csv`

`base-K2` 的底层 solver 已经明显更难：

- ID density: `0.3275`
- OOD-alpha density: `0.3406`
- OOD-structure density: `0.3732`

而 conservative residual correction 的 3-seed 结果是：

- ID density: `0.27986 ± 0.00095`
- OOD-alpha density: `0.28903 ± 0.00139`
- OOD-structure density: `0.32653 ± 0.00210`

这意味着：

- ID 改善：`0.0477`
- OOD-alpha 改善：`0.0515`
- OOD-structure 改善：`0.0467`

这组结果非常关键，因为它把趋势变得很明确：

- `base_K0=4`：有小幅稳定增益
- `base_K0=3`：有中等稳定增益
- `base_K0=2`：有显著稳定增益

也就是说：

**AI residual correction 的收益会随着 observable budget 收紧而系统性放大。**

这比“单个设定下模型稍微变好”强很多，因为它说明：

- 我们不是碰巧找到了一组调参
- 而是在抓一个真实的 accuracy-vs-budget 恢复规律

### 当前最接近论文主实验的版本

到目前为止，最像论文主结果的主线已经不是单点表，而是这条 budget curve：

1. `full ShadowFluid`
2. `budgeted ShadowFluid`
3. `budgeted ShadowFluid + conservative residual`
4. `FNO / NSO` 作为外部神经参考

其中最强的 ML 结论是：

**在更紧的 observable budget 下，small residual correction can recover a substantial fraction of the lost accuracy, and this recovery is stable across seeds and OOD splits.**

## 建议的最小实现计划

### Phase 1

实现 `shadowfluid_residual_lowfreq`

- base solver: `ShadowFluid`
- learned head: `delta_lowfreq(t)`
- outputs:
  - corrected `lowfreq`
  - corrected `rho`
  - optional corrected `energy`

### Phase 2

先只回答一个问题：

**在相同 `ShadowFluid` budget 下，residual correction 能否稳定降低 `ID / OOD / long-rollout` 误差？**

### Phase 3

如果有效，再扩成第二个问题：

**在更小 budget 下，AI correction 能否恢复到接近 full `ShadowFluid` 的精度？**

这一步最有希望形成真正有说服力的论文卖点。

---

## 当前结论

基于现在所有实验，后续最应继续的不是：

- 更复杂的 learned dictionary selector
- 更强的 candidate attention
- 更大的 latent decoder

而是：

**让 `ShadowFluid` 继续做它最擅长的主干演化，让 AI 只学它尚未闭合的残差。**

### 2026-05-06 再扩展：`N=64` pilot 也支持同一条主线

为了确认这条规律不只停留在 `N=32`，又补了一版小规模 `N=64` pilot：

- dataset:
  - `code/results/bench_data/exp64_pilot_dataset_v1.npz`
- summary:
  - `code/results/comparison_tables/exp64_pilot_budget_curve_summary.csv`

这轮先只看最关键的两档 budget：

- `base_K0=3`
- `base_K0=2`

并且补齐了：

- `ID`
- `OOD-alpha`
- `OOD-structure`
- `long-rollout`
- residual 的 clean 3-seed 汇总

#### `N=64`, `base_K0=3`

under-budget ShadowFluid:

- ID density: `0.19357`
- OOD-alpha density: `0.22036`
- OOD-structure density: `0.24281`
- long-rollout final density: `0.18548`

`conservative density-first residual` 的 3-seed 结果：

- ID density: `0.14793 ± 0.00061`
- OOD-alpha density: `0.17219 ± 0.00041`
- OOD-structure density: `0.19432 ± 0.00043`
- long-rollout final density: `0.14901`

对应改善：

- ID: `0.04564`
- OOD-alpha: `0.04818`
- OOD-structure: `0.04849`
- long-rollout final: `0.03647`

#### `N=64`, `base_K0=2`

under-budget ShadowFluid:

- ID density: `0.28401`
- OOD-alpha density: `0.30392`
- OOD-structure density: `0.32883`
- long-rollout final density: `0.29538`

`conservative density-first residual` 的 3-seed 结果：

- ID density: `0.20042 ± 0.00167`
- OOD-alpha density: `0.21659 ± 0.00232`
- OOD-structure density: `0.24681 ± 0.00185`
- long-rollout final density: `0.22925`

对应改善：

- ID: `0.08358`
- OOD-alpha: `0.08733`
- OOD-structure: `0.08202`
- long-rollout final: `0.06613`

#### 这轮 `N=64` pilot 的意义

这组结果把目前最重要的规律进一步坐实了：

1. 这条 residual 路线不是只在 `N=32` 上有效。
2. 它在 `ID / OOD-alpha / OOD-structure / long-rollout` 上都能给出同方向改善。
3. 改善幅度仍然随着 budget 变紧而明显放大：
   - `K0=3`：中等稳定增益
   - `K0=2`：更大、更稳定的增益

因此现在最可信的主实验故事已经非常清楚：

**AI residual correction 并不是在替代 ShadowFluid，而是在 aggressive observable truncation 下，稳定恢复被 budget 限制损失掉的精度；而且这个恢复效应会随着 solver budget 收紧而系统性增强。**

### 2026-05-06 正式化为论文主实验

为了避免后续继续手工拼表，又补了一个 paper-facing 汇总脚本：

- `code/scripts/make_paper_main_tables.py`

这个脚本当前会自动生成三张表：

1. `code/results/comparison_tables/paper_main_exp32_budget.csv`
2. `code/results/comparison_tables/paper_reference_baselines_exp32.csv`
3. `code/results/comparison_tables/paper_main_exp64_pilot.csv`

#### 当前建议的论文主实验结构

主表应围绕这条 budget curve 展开：

1. `ShadowFluid (full)` 作为 anchor
2. `ShadowFluid budgeted (K0=4,3,2)`
3. `ShadowFluid + residual (K0=4,3,2)`

其中：

- `exp32` 是当前主结果表
- `exp64 pilot` 是尺度扩展与趋势验证

#### 当前建议的 baseline 参考结构

baseline 不再作为主故事，而是作为参考列出：

- `FNO`
- `NSO (learned dictionary)`
- `ShadowFluid (full)`

再加上最关键的两条 budgeted 主线参考：

- `ShadowFluid budgeted (K0=3)`
- `ShadowFluid + residual (K0=3)`

当前这张参考表在：

- `code/results/comparison_tables/paper_reference_baselines_exp32.csv`

从这张表里可以直接读到目前最重要的关系：

- `ShadowFluid (full)` 仍然是最强上界
- `FNO / NSO` 是外部学习型参考
- `budgeted ShadowFluid + residual` 已经明显优于同 budget 的纯 physics 基线

也就是说，论文的主问题已经从：

- “能不能学出比 ShadowFluid 更强的 surrogate”

收敛成了：

- **“在 solver budget 受限时，AI residual 能否系统性恢复 ShadowFluid 的精度、OOD 泛化和 rollout 稳定性。”**
