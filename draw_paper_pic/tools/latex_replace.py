import os
import re

# ==== 配置部分 ====
FOLDER_PATH = "."

# 替换规则
REPLACEMENTS_COMMON = {
    "---": ", ",
    "—": ", ",
    "--": "-",
    "–": "-",
    "’": "'",
    "”": "''",
    "‘": "`",
    "“": "``",
    r"\paragraph": r"\textit",
    "Section~": r"\S~",
    "Sec.~": r"\S~",
    " ,": r",",
    " .": r".",
    "Fig.~": r"Figure~",
    "Tab.~": r"Table~",
    r"\textsc": r"\textit",
    r"↓": r"$\downarrow$",
    r"↑": r"$\uparrow$",
}

# 括号编号类替换（只在非 sec0_main.tex、packages.tex 文件中启用）
REPLACEMENTS_PAREN = {
    "(1)": r"\first",
    "(2)": r"\second",
    "(3)": r"\third",
    "(4)": r"\fourth",
    "(5)": r"\fifth",
    #"(a)": r"\first",
    #"(b)": r"\second",
    #"(c)": r"\third",
    #"(d)": r"\fourth",
    #"(e)": r"\fifth",
    "(i)": r"\first",
    "(ii)": r"\second",
    "(iii)": r"\third",
    "(iv)": r"\fourth",
    "(v)": r"\fifth",
    "e.g.,": r"\eg",
    "i.e.,": r"\ie",
    r"\textbf": r"\textit",
    r"\[": r"\begin{equation}",
    r"\]": r"\end{equation}",
}

BACKUP = False


# ==== 主逻辑 ====
def safe_paren_replace(text, old, new):
    """
    安全替换 (a)、(b)... 等编号。
    跳过：
      1. 前面是命令 (\textbf{(a)})
      2. 前面是右括号类符号 }(a)、](b)、)(c)
      3. 前面是 \ref{...}、\cref{...}、\Cref{...}
    允许：
      顶格 (a)、前面只有空格/换行/Tab 的 (a)
    """
    pattern = re.compile(re.escape(old))

    def replacement(match):
        start = match.start()

        # 截取前缀，清除 BOM、Tab、回车、空格
        prefix = text[max(0, start - 120):start]
        prefix = prefix.replace("\ufeff", "").replace("\r", "")
        # 如果前缀全是空白（包括换行），直接替换
        if prefix.strip() == "":
            return new

        prev_char = prefix[-1]

        # 1️⃣ 前面是命令
        if re.search(r"\\[A-Za-z]+$", prefix):
            return match.group(0)

        # 2️⃣ 前一个字符是右括号类符号
        if prev_char in "}])":
            return match.group(0)

        # 3️⃣ 前面是 \ref、\cref、\Cref
        if re.search(r"(?:~|\s)*\\[cC]?ref\{[^}]+\}\s*$", prefix):
            return match.group(0)

        # ✅ 其他情况替换
        return new

    # 去掉文件级 BOM
    text = text.replace("\ufeff", "")
    return pattern.sub(replacement, text)


def smart_replace_line(line, replacements_common, replacements_paren, enable_paren=True):
    """对单行执行替换，跳过注释行"""
    stripped = line.lstrip()
    if stripped.startswith("%"):
        return line  # 注释行不动

    new_line = line

    # 通用替换
    for old, new in replacements_common.items():
        new_line = new_line.replace(old, new)

    # 括号编号替换
    if enable_paren:
        for old, new in replacements_paren.items():
            # 对 \[ 和 \] 做特殊处理
            if old == r"\[":
                new_line = re.sub(r'(^|[^\\])\\\[\s*', lambda m: m.group(1) + new + '\n', new_line)
                continue
            elif old == r"\]":
                new_line = re.sub(r'(^|[^\\])\\\]\s*', lambda m: m.group(1) + new + '\n', new_line)
                continue

            new_line = safe_paren_replace(new_line, old, new)
    if re.fullmatch(r"\s*\\par\s*", new_line):
        return "\n"
    
    return new_line

def find_table_regions(lines):
    """返回哪些行在 table 环境内"""
    in_table = False
    table_mask = []
    for line in lines:
        if "\\begin{table" in line:
            in_table = True
        table_mask.append(in_table)
        if "\\end{table" in line:
            in_table = False
    return table_mask


def process_file(file_path):
    if not file_path.lower().endswith(".tex"):
        return

    filename = os.path.basename(file_path)
    enable_paren = not (filename in ["sec0_main.tex", "packages.tex"])

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 🔍 找出每行是否在 table 中
    table_mask = find_table_regions(lines)

    new_lines = []
    for line, in_table in zip(lines, table_mask):

        # ⭐ 在 table 内，不进行 \textbf → \textit 的替换
        if in_table:
            replacements_common = {
                k: v for k, v in REPLACEMENTS_COMMON.items()
                if k != r"\textbf"
            }
            replacements_paren = {
                k: v for k, v in REPLACEMENTS_PAREN.items()
                if k != r"\textbf"
            }
        else:
            replacements_common = REPLACEMENTS_COMMON
            replacements_paren = REPLACEMENTS_PAREN

        new_lines.append(
            smart_replace_line(line, replacements_common, replacements_paren, enable_paren)
        )

    if new_lines != lines:
        if BACKUP:
            backup_path = file_path + ".bak"
            with open(backup_path, "w", encoding="utf-8") as bf:
                bf.writelines(lines)

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        tag = "✅ (含编号)" if enable_paren else "✅ (跳过编号)"
        print(f"{tag} 已处理: {file_path}")
    else:
        print(f"— 无变化: {file_path}")


def process_folder(folder):
    for root, _, files in os.walk(folder):
        for filename in files:
            full_path = os.path.join(root, filename)
            process_file(full_path)


if __name__ == "__main__":
    process_folder(FOLDER_PATH)
    print("🎯 所有文件处理完毕！")
