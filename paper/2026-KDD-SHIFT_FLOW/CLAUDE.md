# ShiftFlow 论文写作规范

## LaTeX 编辑常驻规则

编辑 `.tex` 文件时，以下规则始终生效：

### 语体与风格
- 使用标准学术书面语，严禁缩写形式（don't → does not, it's → it is, can't → cannot）
- 避免名词所有格（METHOD's performance → the performance of METHOD）
- 禁止使用 \item 列表呈现正文内容，必须保持连贯段落
- 不使用破折号（—），用从句或同位语替代
- 不主动添加 \textbf 或 \emph 强调格式，除非原文已有

### 去 AI 味
- 禁用词汇：leverage, delve into, tapestry, utilize, facilitate, aforementioned, it is worth noting that, first and foremost
- 替代方案：use, investigate, context, help, above, (直接删除冗余过渡词)
- 用词朴实精准（Simple & Clear），不堆砌华丽辞藻或生僻词

### 时态
- 一般现在时：描述方法、架构、实验结论
- 过去时：仅用于明确的历史事件或已完成的具体实验操作

### LaTeX 规范
- 必须转义特殊字符：`%` → `\%`，`_` → `\_`，`&` → `\&`
- 保留所有 LaTeX 命令原样（\cite{}, \ref{}, \label{}, \eg, \ie）
- 不展开领域通用缩写（LLM, CFD, PDE 等保持原样）
- 保持数学公式原样（保留 $ 符号）

### 内容纪律
- 不编造数据、不夸大实验结果
- 不随意增删原文的技术细节或限定条件
- 修改前先读取文件，理解上下文

## 绘图规范

- 使用 `plot_common.py` 中的 `apply_paper_rcparams()` + `set_paper_style(ax)` 统一风格
- 配色：PALETTE_PASTEL（pastel 柔和色调）
- 白底、无网格、仅 left/bottom spines、无衬线字体
- 输出 PDF（矢量）+ 300dpi PNG 备份

## 可用 Slash Commands

论文写作相关命令（输入 `/` 查看完整列表）：
- `/zh2en` 中→英翻译 | `/en2zh` 英→中翻译 | `/zh2zh` 中文重构
- `/polish-en` 英文润色 | `/polish-zh` 中文润色
- `/condense` 缩写 | `/expand` 扩写
- `/de-ai` 去AI味 | `/logic-check` 逻辑审查
- `/review` Reviewer审稿 | `/exp-analysis` 实验分析
- `/fig-caption` 图标题 | `/tab-caption` 表标题
- `/fig-recommend` 绘图推荐 | `/paper-diagram` 架构图
