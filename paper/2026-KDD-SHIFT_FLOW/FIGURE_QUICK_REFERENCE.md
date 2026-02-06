# SHIFT-FLOW 图表库：快速参考卡片

**生成时间**：2026-02-07
**总数**：26张高质量学术图表
**存储位置**：
- `figs_nips_gallery/` (17张)
- `figs_advanced_gallery/` (9张)

---

## 🎯 一句话快速选择

| 你想表达什么 | 推荐第一选择 | 风格 | 备选 |
|-----------|-----------|------|-----|
| 精度vs截断（最重要） | `a2_lines_with_bands_nature.pdf` | Nature | a1, a3 |
| 误差-成本权衡 | `b1_pareto_scatter_neurips.pdf` | NeurIPS | k2 |
| 时间演化 | `c2_error_vs_time_bands.pdf` | Nature | c1 |
| 影子最优性证明 | `g1_shadow_vs_lowpass_overlay.pdf` | Nature | g2 |
| 加速倍数 | `d2_speedup_inset.pdf` | Science | d1 |
| 系统尺度扩展 | `f1_multi_panel_by_nx.pdf` | NeurIPS | a4 |
| 鲁棒性/方差 | `e1_violin_distributions.pdf` | Science | e2 |
| 一页纸总结 | `h2_nature_3panel_dense.pdf` | Nature | m1 |
| KDD特定风格 | `m1_kdd_summary_dashboard.pdf` | KDD | i1, k2 |

---

## 📋 核心图表速查表

### 最小必要集合（3张）
```
1. a2_lines_with_bands_nature.pdf      [精度主结果]
2. b1_pareto_scatter_neurips.pdf       [权衡分析]
3. h2_nature_3panel_dense.pdf          [总结]
```
**用途**：快速投稿、演讲、报告

### 标准集合（6-7张）
```
1. a2_lines_with_bands_nature.pdf      [精度]
2. b1_pareto_scatter_neurips.pdf       [权衡]
3. c2_error_vs_time_bands.pdf          [时间]
4. g1_shadow_vs_lowpass_overlay.pdf    [优化证明]
5. d2_speedup_inset.pdf                [加速]
6. e1_violin_distributions.pdf         [鲁棒性]
7. h2_nature_3panel_dense.pdf/m1_*     [总结]
```
**用途**：Nature/Science期刊、一级会议

### 完整集合（10-12张）
标准集合 + 以下任选：
```
- a4_heatmap_error.pdf                 [全局视图]
- l2_time_evolution_heatmap.pdf        [动力学过程]
- j1_contour_error_landscape.pdf       [3D空间]
- i1_dual_axis_error_efficiency.pdf    [多维展示]
- f1_multi_panel_by_nx.pdf             [系统依赖]
```
**用途**：论文补充材料、详细报告

---

## 🏆 按顶刊选择

### 投稿 Nature/Science
**必选**：
- `a2_lines_with_bands_nature.pdf`
- `c2_error_vs_time_bands.pdf`
- `h2_nature_3panel_dense.pdf`

**推荐**：
- `g1_shadow_vs_lowpass_overlay.pdf`
- `e1_violin_distributions.pdf`

**风格**：衬线字体，粗线条，置信区间，高对比

### 投稿 NeurIPS/ICML
**必选**：
- `a1_grouped_bars_neurips.pdf`
- `b1_pareto_scatter_neurips.pdf`
- `h1_neurips_2col_compact.pdf`

**推荐**：
- `c1_error_vs_time_lines.pdf`
- `d1_runtime_grouped_bars.pdf`

**风格**：无衬线字体，colorblind-friendly，极简

### 投稿 KDD/VLDB/WWW
**必选**：
- `a1_grouped_bars_neurips.pdf` 或 `a3_minimal_lines_science.pdf`
- `b1_pareto_scatter_neurips.pdf` 或 `k2_pareto_cost_vs_accuracy.pdf`
- `m1_kdd_summary_dashboard.pdf`

**推荐**：
- `i1_dual_axis_error_efficiency.pdf` [成本效率]
- `j1_contour_error_landscape.pdf` [全局优化]
- `l2_time_evolution_heatmap.pdf` [动力学]

**风格**：技术性，权衡突出，多维度

---

## 🎨 图表类型速查

需要...？ → 推荐

- **线图（趋势）** → c1, c2, d2, i1, k2
- **柱状图（对比）** → a1, d1, k1
- **散点图（关系）** → b1, b2, i2, j2
- **热力图（全局）** → a4, j1, l2
- **分布图（方差）** → e1, e2, l1
- **多panel** → f1, h1, h2, m1
- **双轴图** → i1
- **气泡图** → i2
- **等高线** → j1
- **Pareto图** → b1, k2

---

## 📊 数据故事映射

### 故事1：精度vs截断（核心）
```
主图：a2_lines_with_bands_nature.pdf
      或 a1_grouped_bars_neurips.pdf
补充：a4_heatmap_error.pdf（不同nx×K₀的总体视图）
      e1_violin_distributions.pdf（种子方差）
```

### 故事2：误差-成本权衡
```
主图：b1_pareto_scatter_neurips.pdf
      或 k2_pareto_cost_vs_accuracy.pdf
备选：b2_tradeoff_annotated.pdf（带标签）
```

### 故事3：影子最优性
```
必需：g1_shadow_vs_lowpass_overlay.pdf
      或 g2_ratio_shadow_to_lowpass.pdf
含义：证明影子进化没有额外截断以外的误差
```

### 故事4：加速收益
```
主图：d2_speedup_inset.pdf（带细节插图）
      或 d1_runtime_grouped_bars.pdf
高阶：i1_dual_axis_error_efficiency.pdf（同时显示误差和效率）
```

### 故事5：时间演化
```
主图：c2_error_vs_time_bands.pdf（不同K₀）
进阶：l2_time_evolution_heatmap.pdf（K₀vs时间矩阵）
      l1_error_growth_rate.pdf（增长率分析）
```

### 故事6：系统规模扩展性
```
主图：f1_multi_panel_by_nx.pdf（按nx分行）
      或 a4_heatmap_error.pdf（2D热力图）
含义：展示是否独立于系统大小
```

---

## ⚙️ 自定义生成

### 修改配色方案
编辑脚本头部的palette函数，例如：
```python
# 在 plot_nips_gallery.py 中
def palette_custom() -> dict:
    return {
        "primary": ["#...", "#...", ...],
        "light": ["#...", "#...", ...],
        ...
    }
```

### 修改字体大小
```bash
python3 experiments/plot_nips_gallery.py --figdir figs_custom
# 编辑脚本中的 fontsize=10 参数
```

### 针对特定目标重新生成
```bash
# 只生成Advanced图表
python3 experiments/plot_advanced_gallery.py --figdir figs_kdd_specific

# 指定输入数据
python3 experiments/plot_nips_gallery.py --in results/sweep.csv --figdir figs_v2
```

---

## 💾 文件清单

### 生成的脚本
- `experiments/plot_nips_gallery.py` (17个函数)
- `experiments/plot_advanced_gallery.py` (11个函数)
- 依赖：`experiments/plot_common.py`

### 生成的图表
- **figs_nips_gallery/** (17张, 23×17=391KB)
  - Type A: 精度vs截断 (4)
  - Type B: 权衡 (2)
  - Type C: 时间 (2)
  - Type D: 运行时 (2)
  - Type E: 分布 (2)
  - Type F: 小倍数 (1)
  - Type G: Shadow vs LP (2)
  - Type H: 高级布局 (2)

- **figs_advanced_gallery/** (9张, 23×9=207KB)
  - Type I: 双轴+气泡 (2)
  - Type J: 等高线+投影 (2)
  - Type K: 归一化 (2)
  - Type L: 时间动态 (2)
  - Type M: 仪表板 (1)

### 文档
- `FIGURE_INDEX.md` (详细索引)
- 本文件 (快速参考)

---

## 🚀 立即使用

### 步骤1：预览所有图表
```bash
# macOS
open figs_nips_gallery/
open figs_advanced_gallery/

# 或用PDF阅读器打开单个文件
```

### 步骤2：选择你的图表
参考上面的"一句话快速选择"表格

### 步骤3：集成到论文
```
将选定的PDF复制到论文目录：
\begin{figure}
  \includegraphics[width=0.9\linewidth]{figs/a2_lines_with_bands_nature.pdf}
  \caption{...}
\end{figure}
```

### 步骤4：调整（如需要）
- 在Adobe Illustrator中打开PDF进行微调
- 或修改Python脚本重新生成

---

## 📌 关键提示

✅ **做**：
- 根据目标刊物选择风格
- 为不同图表使用一致的色盘
- 在文本中明确引用图表编号

❌ **不做**：
- 混合不同风格（不要NeurIPS+Nature混用）
- 只选择"好看"的，忽视数据准确性
- 过度装饰（顶刊偏向极简）

---

## 📞 常见问题

**Q: 我应该选哪个版本的"精度vs K₀"图？**
A:
- Nature/高级期刊 → `a2_lines_with_bands_nature.pdf`
- NeurIPS/会议 → `a1_grouped_bars_neurips.pdf`
- 空间有限 → `a3_minimal_lines_science.pdf`

**Q: 如何证明影子进化是最优的？**
A: 使用 `g1_shadow_vs_lowpass_overlay.pdf` 或 `g2_ratio_shadow_to_lowpass.pdf`
确保shadow曲线几乎完全重合于low-pass baseline

**Q: 我想展示多个维度的权衡**
A:
- 3维 → `i2_bubble_error_vs_speedup.pdf` (气泡大小=系统)
- 多维 → `j2_3d_projection_scatter.pdf` (两个投影)
- 完整仪表板 → `m1_kdd_summary_dashboard.pdf`

**Q: 能否改变颜色和字体？**
A: 可以！编辑 `plot_nips_gallery.py` 中的：
- `palette_*()` 函数（改色彩）
- `setup_matplotlib_journal()` 函数（改字体）
- 重新运行脚本生成

---

**完整索引**：见 `FIGURE_INDEX.md`
**脚本源代码**：`experiments/plot_*.py`
**原始数据**：`results/sweep.csv`

---

*Generated: 2026-02-07*
*For SHIFT-FLOW KDD Paper*
