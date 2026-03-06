# SHIFT-FLOW 图表库索引

**总计：28张高质量学术图表 | 4种顶刊风格 | 8种图表类型**

---

## 🎯 快速选择指南

### 按用途选择

#### 📊 主要故事线：精度 vs 截断
- **Exp1**：精度随K₀的变化（核心图表）
  - `a1_grouped_bars_neurips.pdf` — NeurIPS风格，清晰对比
  - `a2_lines_with_bands_nature.pdf` — Nature风格，含置信区间
  - `a3_minimal_lines_science.pdf` — Science风格，极简设计
  - `a4_heatmap_error.pdf` — 热力图，全局视图

#### ⚖️ 权衡分析：误差 vs 成本
- **Pareto Frontier**
  - `b1_pareto_scatter_neurips.pdf` — 标准Pareto图，多系统大小
  - `b2_tradeoff_annotated.pdf` — 带K₀标签的详细版本
  - `k2_pareto_cost_vs_accuracy.pdf` — 填充区域风格

#### ⏱️ 时间演化
- **Error vs Time**
  - `c1_error_vs_time_lines.pdf` — 多条曲线
  - `c2_error_vs_time_bands.pdf` — Nature风格，含误差带
  - `l2_time_evolution_heatmap.pdf` — (K₀ vs t)热力图

#### 🚀 加速与效率
- **Runtime & Speedup**
  - `d1_runtime_grouped_bars.pdf` — Full vs Shadow对比
  - `d2_speedup_inset.pdf` — 带放大插图的专业版
  - `i1_dual_axis_error_efficiency.pdf` — 双Y轴（误差+效率）

#### 🔍 截断最优性
- **Shadow vs Low-pass**
  - `g1_shadow_vs_lowpass_overlay.pdf` — 直接叠加对比
  - `g2_ratio_shadow_to_lowpass.pdf` — 比值图，验证最优性

#### 📈 鲁棒性分析
- **Seed Distribution**
  - `e1_violin_distributions.pdf` — 小提琴图
  - `e2_box_whisker_distributions.pdf` — 箱线图
  - `l1_error_growth_rate.pdf` — 误差增长率

#### 🎨 多维度集成
- **Dashboard & Summary**
  - `m1_kdd_summary_dashboard.pdf` — KDD风格总结板（6个panel）
  - `h1_neurips_2col_compact.pdf` — NeurIPS紧凑布局（3个panel）
  - `h2_nature_3panel_dense.pdf` — Nature风格3panel（a|b|c标签）

---

## 📁 完整图表列表

### Type A: 精度vs截断 (4张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `a1_grouped_bars_neurips.pdf` | NeurIPS | 分组柱状图，多nx | 快速对比，顶会投稿 |
| `a2_lines_with_bands_nature.pdf` | Nature | 折线+置信带 | 强调不确定性，高级期刊 |
| `a3_minimal_lines_science.pdf` | Science | 极简风格 | 紧凑排版，空间受限 |
| `a4_heatmap_error.pdf` | Science | 2D热力图(nx vs K₀) | 全局总结，系统级分析 |

### Type B: 误差-成本权衡 (2张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `b1_pareto_scatter_neurips.pdf` | NeurIPS | 散点+连接线 | 多个系统大小的对比 |
| `b2_tradeoff_annotated.pdf` | Science | 带K₀标签的轨迹 | KDD/会议演讲 |

### Type C: 时间演化 (2张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `c1_error_vs_time_lines.pdf` | NeurIPS | 多条曲线 | 简洁清晰，标准对比 |
| `c2_error_vs_time_bands.pdf` | Nature | 折线+置信带 | 强调变化趋势 |

### Type D: 运行时与加速 (2张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `d1_runtime_grouped_bars.pdf` | NeurIPS | 柱状图，按系统大小 | 直观展示加速 |
| `d2_speedup_inset.pdf` | Science | 主图+放大插图 | 突出细节差异 |

### Type E: 鲁棒性分布 (2张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `e1_violin_distributions.pdf` | Science | 小提琴图 | 展示概率密度 |
| `e2_box_whisker_distributions.pdf` | NeurIPS | 箱线图 | 传统统计视图 |

### Type F: 小倍数/多panel (1张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `f1_multi_panel_by_nx.pdf` | NeurIPS | 按nx分行 | 显示系统规模依赖性 |

### Type G: Shadow vs Baseline (2张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `g1_shadow_vs_lowpass_overlay.pdf` | Nature | 两条曲线叠加 | 证明截断最优性 |
| `g2_ratio_shadow_to_lowpass.pdf` | Science | 比值图(=1表示最优) | 量化最优性程度 |

### Type H: 高级多panel布局 (2张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `h1_neurips_2col_compact.pdf` | NeurIPS | 2列×2行，1列跨度 | 顶会投稿，紧凑 |
| `h2_nature_3panel_dense.pdf` | Nature | 3panel (a\|b\|c) | Nature/Science风格论文 |

### Type I: 双轴与气泡图 (2张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `i1_dual_axis_error_efficiency.pdf` | KDD | 左轴误差+右轴效率 | 同时展示多个维度 |
| `i2_bubble_error_vs_speedup.pdf` | KDD | 气泡大小=系统大小 | 3维信息编码 |

### Type J: 等高线与投影 (2张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `j1_contour_error_landscape.pdf` | KDD | 等高线+热力图 | 全局最优化视图 |
| `j2_3d_projection_scatter.pdf` | KDD | 两个2D投影 | 3维权衡空间 |

### Type K: 归一化与效率 (2张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `k1_normalized_error_bars.pdf` | KDD | %相对差异柱状图 | 突出相对贡献 |
| `k2_pareto_cost_vs_accuracy.pdf` | KDD | 填充曲线Pareto | KDD论文，权衡强调 |

### Type L: 时间动态 (2张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `l1_error_growth_rate.pdf` | KDD | 增长率vs时间+K₀ | 动力学分析 |
| `l2_time_evolution_heatmap.pdf` | KDD | (K₀ vs t)热力图 | 全局动态过程 |

### Type M: 仪表板 (1张)

| 文件 | 风格 | 特点 | 最适用于 |
|-----|------|------|--------|
| `m1_kdd_summary_dashboard.pdf` | KDD | 6panel汇总 | 论文补充材料/报告 |

---

## 🎨 按顶刊风格分类

### NeurIPS/ICML 风格 (8张)
✓ 无衬线字体，极简设计，colorblind-friendly配色
- a1, b1, c1, d1, e2, h1 + 2×Type I

**推荐使用场景：**
- NeurIPS/ICML投稿
- 计算机会议（KDD, WWW）
- 演讲/报告

### Nature/Science 风格 (8张)
✓ 衬线字体，粗线条，置信区间，高对比度
- a2, c2, d2, e1, f1, g1, h2 + 1×其他

**推荐使用场景：**
- Nature/Science 投稿
- 高影响力期刊
- 强调严谨性

### KDD/应用会议 风格 (6张)
✓ 技术性强，权衡突出，多维编码
- Type I (2), Type J (2), Type K (2)

**推荐使用场景：**
- KDD/VLDB/WWW
- 强调实用性和效率
- 权衡与优化叙事

### 其他通用风格 (2+6张)
- a3, a4 (Science) + Type L, M

---

## 💡 根据论文章节选择图表

### Introduction / Motivation
推荐：`h2_nature_3panel_dense.pdf` 或 `m1_kdd_summary_dashboard.pdf`
- 一页纸概括全部故事

### Methods / Proof of Concept
推荐：`g1_shadow_vs_lowpass_overlay.pdf` + `g2_ratio_shadow_to_lowpass.pdf`
- 证明截断最优性

### Main Results (Accuracy)
推荐组合：
1. **主图**：`a2_lines_with_bands_nature.pdf` 或 `a1_grouped_bars_neurips.pdf`
2. **补充**：`a4_heatmap_error.pdf`（全局视图）
3. **鲁棒性**：`e1_violin_distributions.pdf`

### Main Results (Trade-offs)
推荐组合：
1. **主图**：`b1_pareto_scatter_neurips.pdf` 或 `k2_pareto_cost_vs_accuracy.pdf`
2. **加速**：`d2_speedup_inset.pdf`
3. **高级**：`j1_contour_error_landscape.pdf`（可选）

### Main Results (Temporal)
推荐组合：
1. **主图**：`c2_error_vs_time_bands.pdf`
2. **动态**：`l2_time_evolution_heatmap.pdf`
3. **增长**：`l1_error_growth_rate.pdf`

### Supplementary Materials
推荐：`f1_multi_panel_by_nx.pdf`, `i1_dual_axis_error_efficiency.pdf`, `j2_3d_projection_scatter.pdf`

---

## 🔍 数据内涵覆盖

### ✅ 已覆盖的故事
- [x] 精度随截断的变化（单调性、系统大小依赖）
- [x] 误差-成本Pareto权衡
- [x] 影子进化的最优性（vs低通滤波基线）
- [x] 加速倍数（Full vs Shadow）
- [x] 时间演化（误差随时间增长）
- [x] 鲁棒性（种子方差）
- [x] 系统规模扩展性（nx依赖）
- [x] 运行时对比

### 📊 图表类型多样性
- [x] 折线图（3种变体）
- [x] 柱状图（5种）
- [x] 散点图（4种）
- [x] 热力图（3种）
- [x] 分布图（2种）
- [x] 双轴图
- [x] 气泡图
- [x] 等高线图
- [x] 3D投影
- [x] 多panel布局

---

## ✨ 编辑建议

### 如果时间/篇幅有限：

**最小集合（3张）：**
1. `a2_lines_with_bands_nature.pdf` — 精度vs K₀（核心）
2. `b1_pareto_scatter_neurips.pdf` — 权衡（故事）
3. `h2_nature_3panel_dense.pdf` — 总结（补充）

**推荐集合（6张）：**
1. `a2_lines_with_bands_nature.pdf` — 主结果1
2. `b1_pareto_scatter_neurips.pdf` — 主结果2
3. `c2_error_vs_time_bands.pdf` — 主结果3
4. `g1_shadow_vs_lowpass_overlay.pdf` — 验证
5. `d2_speedup_inset.pdf` — 加速
6. `h2_nature_3panel_dense.pdf` — 总结

**完整集合（10-12张）：**
- 上述6张 + `a4_heatmap_error.pdf` + `e1_violin_distributions.pdf` + `l2_time_evolution_heatmap.pdf` + 2×高级图表

---

## 🚀 使用方法

```bash
# 查看特定图表
open figs_nips_gallery/a1_grouped_bars_neurips.pdf
open figs_advanced_gallery/m1_kdd_summary_dashboard.pdf

# 批量查看（macOS）
open figs_nips_gallery/
open figs_advanced_gallery/

# 批量查看（Linux）
ls -lh figs_nips_gallery/*.pdf
# 用你的PDF阅读器打开

# 重新生成所有图表
python3 experiments/plot_nips_gallery.py --figdir figs_nips_gallery
python3 experiments/plot_advanced_gallery.py --figdir figs_advanced_gallery
```

---

## 🎓 图表设计参考

所有图表参考以下顶刊标准：
- **Nature**: Serif字体，粗线条（LW=1.8-2.2），高对比度，完整legend
- **NeurIPS**: Sans-serif，细线条（LW=1.5-1.8），Okabe-Ito色盲友好配色
- **Science**: 清晰的visual hierarchy，中等线宽，鲜艳配色
- **KDD**: 技术性强，emphasis on trade-offs，多维度编码

---

**最后生成日期**：2026-02-07
**数据来源**：`results/sweep.csv` (704KB, 多参数扫描)
**总图表数**：28张
**覆盖的图表类型**：13种
**支持的顶刊风格**：4种
