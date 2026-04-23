# Rebuttal Stress Tests

这批材料对应的是对 `Reviewer oB6r` 的 rebuttal 回复。

这组文件用于支撑 rebuttal 中关于“势场更复杂时，ShadowFluid 的参考集大小、约化维度与误差如何变化”的说明。

对应脚本：

- [experiments/run_rebuttal_stress_tests.py](/home/adurey/QuantumComputing/ShadowFluid/experiments/run_rebuttal_stress_tests.py)

## 这个脚本做什么

脚本会生成 4 组结构化 stress test，用来回答审稿人关于 coupling graph 变密、`|R|` 是否快速膨胀、以及 closure depth 与误差之间 tradeoff 的问题。

默认实验包括：

1. 固定总耦合强度 `alpha_total`，增加 Fourier 分量数 `J`，并扫描 `R_hops`
2. 在最 dense 的 structured potential 上扫描时间 `t`
3. 固定每个 Fourier 分量的强度 `alpha_each`，增加 `J`
4. 在固定 `alpha_each` 的 dense case 上扫描 `R_hops`

结构化势场族按如下顺序加入 Fourier 分量：

`(1,0)`, `(0,1)`, `(1,1)`, `(1,-1)`, `(2,0)`, `(0,2)`, `(2,1)`, `(1,2)`

## 怎么运行

在仓库根目录执行：

```bash
python experiments/run_rebuttal_stress_tests.py
```

默认输出目录就是当前文件所在目录：

```text
data/rebuttal_stress_tests/
```

## 常用参数

```bash
python experiments/run_rebuttal_stress_tests.py \
  --nx 5 \
  --seed 0 \
  --K0 5.0 \
  --t 0.5 \
  --alpha-total 0.5 \
  --alpha-each 0.1 \
  --J-list 1,2,4,6,8 \
  --hops-list 1,2,3 \
  --t-list 0.2,0.5,0.8,1.0
```

参数含义：

- `--nx`: 网格幂次，实际网格大小是 `N = 2^nx`
- `--seed`: 初态种子，默认 `0` 是 canonical vortex case
- `--K0`: low-pass cutoff
- `--t`: 非 time-sweep 部分使用的固定演化时间
- `--alpha-total`: 固定总耦合强度实验使用的总强度
- `--alpha-each`: 固定单分量强度实验使用的每项强度
- `--J-list`: structured potential 中使用的 Fourier 分量数
- `--hops-list`: reference closure 的 BFS 深度
- `--t-list`: dense case time sweep 的时间列表
- `--out-dir`: 自定义输出目录

## 会生成哪些文件

- [fixed_alpha_total.csv](/home/adurey/QuantumComputing/ShadowFluid/data/rebuttal_stress_tests/fixed_alpha_total.csv)  
  固定 `alpha_total`，扫描 `J` 和 `R_hops`

- [dense_time_fixed_alpha_total.csv](/home/adurey/QuantumComputing/ShadowFluid/data/rebuttal_stress_tests/dense_time_fixed_alpha_total.csv)  
  固定 `alpha_total` 的 dense case 时间扫描

- [fixed_alpha_each.csv](/home/adurey/QuantumComputing/ShadowFluid/data/rebuttal_stress_tests/fixed_alpha_each.csv)  
  固定 `alpha_each`，扫描 `J`

- [dense_hops_fixed_alpha_each.csv](/home/adurey/QuantumComputing/ShadowFluid/data/rebuttal_stress_tests/dense_hops_fixed_alpha_each.csv)  
  固定 `alpha_each` 的 dense case closure-depth 扫描

- [summary.md](/home/adurey/QuantumComputing/ShadowFluid/data/rebuttal_stress_tests/summary.md)  
  自动生成的结果摘要，适合快速查看和往 rebuttal 表格里抄

## CSV 字段说明

每个 CSV 都包含一套统一字段，方便后续合并或作图：

- `section`: 当前记录属于哪一组实验
- `nx`, `N`, `seed`, `K0`, `t`: 基本实验配置
- `J`: Fourier 分量数
- `alpha_total`, `alpha_each`: 耦合强度设定
- `R_hops`: BFS closure 深度
- `M_K`: 任务子空间 `K` 的模式数
- `R_size`: 参考集 `R` 的大小
- `reduced_size`: `|K||R|`，即字典规模
- `err_rho_vs_full`: density error
- `err_rho_lp_vs_full`: full low-pass baseline 相对 full 的 density error
- `extra_shadow_err`: `err_rho_vs_full - err_rho_lp_vs_full`
- `err_E_LP`: low-pass task energy error
- `err_Z_frob`: `||ΔZ||_F`
- `leakage_apriori`: a priori leakage `ℓ_rms`
- `bound_apriori`: 线性上界 `t * ℓ_rms`
- `V_label`: 势场配置的字符串标签

## 怎么读这些结果

如果你的目标是回答审稿人对“复杂势场是否会破坏降维优势”的质疑，最先看这几列：

- `R_size`
- `reduced_size`
- `err_Z_frob`
- `leakage_apriori`

通常：

- `R_size` / `reduced_size` 体现 graph 变密或 closure 变深时的成本增长
- `err_Z_frob` 和 `leakage_apriori` 体现 shadow dictionary 近似本身的误差变化
- `err_rho_vs_full` 常常同时受到 low-pass truncation 的影响，因此适合和 `err_rho_lp_vs_full` 对照着看

## 当前默认配置下能直接引用的结论

以默认参数 `nx=5, K0=5, seed=0` 为例：

- 固定 `alpha_total = 0.5`，当 `J` 从 `1` 增加到 `8` 且 `R_hops = 1` 时，`R_size` 从 `3` 增加到 `17`，`reduced_size` 从 `243` 增加到 `1377`
- 在 dense case (`J = 8`) 下，把 `R_hops` 从 `1` 增加到 `3`，`R_size` 会从 `17` 增加到 `115`，但 `err_Z_frob` 会显著下降
- 这说明复杂势场下参考集确实会变大，但这种增长是可量化的，并且可以和误差下降形成明确的 cost-accuracy tradeoff

## 备注

这套脚本默认使用 `use_qiskit=False`，走的是纯经典参考路径，目的是快速稳定地产生 rebuttal 所需的结构分析数据，而不是重复主文中的量子模拟时序结果。
