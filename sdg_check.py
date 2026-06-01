#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SDG 正確性驗證腳本
比對 計劃總覽.txt 與 114md/ 資料夾中各 MD 檔的 SDG 標注
"""

import os
import re
from pathlib import Path

# ── 路徑設定 ──
OVERVIEW_PATH = r"C:\Users\Kuo\desktop\usr\web\qa_data\計劃總覽.txt"
MD_DIR = r"C:\Users\Kuo\desktop\usr\web\114md"
OUTPUT_PATH = r"C:\Users\Kuo\desktop\usr\web\sdg_check_result.txt"


# ── 1. 讀取計劃總覽.txt ──
def load_overview(path):
    """
    回傳:
      overview[sdg_num] = [(school, plan), ...]   # sdg_num: int 1-17
      all_plans[(school, plan)] = set of sdg_nums  # 每筆計畫屬於哪些 SDG
    """
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            with open(path, encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue

    overview = {}          # sdg_num -> list of (school, plan)
    plan_to_sdgs = {}      # (school, plan) -> set of sdg_nums

    current_sdg = None
    for line in text.splitlines():
        # 偵測 SDGx 段落標題，e.g. "SDG3 良好健康與福祉（共 80 個計畫）："
        m = re.match(r"SDG(\d+)\s", line.strip())
        if m:
            current_sdg = int(m.group(1))
            if current_sdg not in overview:
                overview[current_sdg] = []
            continue

        # 偵測計畫列表項目，e.g. "  - 學校名 — 計畫名"
        if current_sdg and line.strip().startswith("- "):
            content = line.strip()[2:].strip()
            if " — " in content:
                parts = content.split(" — ", 1)
                school = parts[0].strip()
                plan = parts[1].strip()
                if plan:  # 忽略計畫名為空的行
                    overview[current_sdg].append((school, plan))
                    key = (school, plan)
                    if key not in plan_to_sdgs:
                        plan_to_sdgs[key] = set()
                    plan_to_sdgs[key].add(current_sdg)

    return overview, plan_to_sdgs


# ── 2. 讀取所有 MD 檔，解析 SDG 號碼 ──
def extract_sdgs_from_md(filepath):
    """
    從 MD 檔的「基本資料表」區段（壹、之前）找 SDGs關聯議題 行，
    提取 1-17 之間的數字。
    回傳 set of int。
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return set()

    # 只看壹、標題之前的區段
    # 找「壹、」的位置
    idx_wa = -1
    for marker in ["壹、", "壹 、", "壹,"]:
        idx_wa = text.find(marker)
        if idx_wa != -1:
            break

    if idx_wa != -1:
        header_text = text[:idx_wa]
    else:
        # 找不到壹、就取前 3000 字元
        header_text = text[:3000]

    # 找含 SDG 和關聯 的行，或直接含 SDGs關聯議題 的行
    sdg_nums = set()
    for line in header_text.splitlines():
        if ("SDGs" in line or "SDG" in line) and ("關聯" in line or "議題" in line):
            nums = re.findall(r'\b(1[0-7]|[1-9])\b', line)
            for n in nums:
                v = int(n)
                if 1 <= v <= 17:
                    sdg_nums.add(v)
    return sdg_nums


def load_md_files(md_dir):
    """
    掃描所有 MD 檔，回傳:
      md_data = list of {
        'filename': str,
        'school': str,   # 底線前的部分
        'plan': str,     # 底線後到 ( 或 .md 之前的部分
        'sdgs': set of int,
        'filepath': str,
      }
    """
    md_data = []
    for fname in os.listdir(md_dir):
        if not fname.endswith(".md"):
            continue
        filepath = os.path.join(md_dir, fname)
        base = fname[:-3]  # 去掉 .md

        # 學校名稱：第一個底線前
        if "_" in base:
            school = base.split("_", 1)[0]
            rest = base.split("_", 1)[1]
        else:
            school = base
            rest = ""

        # 計畫名稱：去掉括號部分（如 (114USR-...)）和 _formatted 等後綴
        plan = re.sub(r'\(114USR[^)]*\)', '', rest).strip()
        plan = re.sub(r'_formatted.*$', '', plan).strip()
        plan = re.sub(r'\(\d+\)$', '', plan).strip()
        plan = plan.rstrip('_').strip()

        sdgs = extract_sdgs_from_md(filepath)

        md_data.append({
            'filename': fname,
            'school': school,
            'plan': plan,
            'sdgs': sdgs,
            'filepath': filepath,
        })

    return md_data


# ── 3. 比對邏輯 ──
def match_plan_to_md(school_ov, plan_ov, md_data):
    """
    用「包含」比對學校名和計畫名，找到最佳匹配的 MD 記錄。
    回傳 md_record or None。
    """
    candidates = []

    for md in md_data:
        # 學校名比對（互相包含）
        school_match = (
            school_ov in md['school'] or
            md['school'] in school_ov or
            # 去掉常見前綴再比
            school_ov.replace("國立", "") in md['school'].replace("國立", "") or
            md['school'].replace("國立", "") in school_ov.replace("國立", "")
        )
        if not school_match:
            continue

        # 計畫名比對（互相包含）
        plan_md = md['plan']
        plan_match = (
            plan_ov in plan_md or
            plan_md in plan_ov or
            # 取前20字元比對（計畫名可能被截斷）
            (len(plan_ov) >= 8 and plan_ov[:12] in plan_md) or
            (len(plan_md) >= 8 and plan_md[:12] in plan_ov)
        )
        if plan_match:
            # 計算相似度（共同字元比例）
            common = sum(1 for c in plan_ov if c in plan_md)
            score = common / max(len(plan_ov), 1)
            candidates.append((score, md))

    if not candidates:
        return None

    # 取分數最高的
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def run_check(overview, plan_to_sdgs, md_data):
    """
    執行比對，回傳:
      missing: list of (sdg_num, school, plan)   # 漏列
      extra:   list of (sdg_num, school, plan)   # 多列
      not_found: list of (school, plan)           # 找不到MD
    """
    missing = []  # MD有但總覽沒有
    extra = []    # 總覽有但MD沒有
    not_found = []

    processed = set()  # 避免同一計畫重複處理

    for (school, plan), overview_sdgs in plan_to_sdgs.items():
        key = (school, plan)
        if key in processed:
            continue
        processed.add(key)

        md_rec = match_plan_to_md(school, plan, md_data)

        if md_rec is None:
            not_found.append((school, plan))
            continue

        md_sdgs = md_rec['sdgs']

        # 漏列：MD有但總覽沒有
        for s in md_sdgs:
            if s not in overview_sdgs:
                missing.append((s, school, plan))

        # 多列：總覽有但MD沒有
        for s in overview_sdgs:
            if s not in md_sdgs:
                extra.append((s, school, plan))

    # 排序
    missing.sort()
    extra.sort()
    not_found.sort()

    return missing, extra, not_found


# ── 4. 輸出結果 ──
def write_result(missing, extra, not_found, output_path):
    lines = []
    lines.append("=== SDG 比對結果 ===")
    lines.append("")

    lines.append(f"【漏列】（MD有但總覽沒有）共 {len(missing)} 筆")
    for sdg, school, plan in missing:
        lines.append(f"  SDG{sdg}: {school} — {plan}")
    lines.append("")

    lines.append(f"【多列】（總覽有但MD沒有）共 {len(extra)} 筆")
    for sdg, school, plan in extra:
        lines.append(f"  SDG{sdg}: {school} — {plan}")
    lines.append("")

    lines.append(f"【找不到對應MD檔】共 {len(not_found)} 筆")
    for school, plan in not_found:
        lines.append(f"  {school} — {plan}")
    lines.append("")

    lines.append(f"總計：漏列 {len(missing)} 筆，多列 {len(extra)} 筆，找不到 {len(not_found)} 筆")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return "\n".join(lines)


# ── 主程式 ──
if __name__ == "__main__":
    print("載入計劃總覽.txt ...")
    overview, plan_to_sdgs = load_overview(OVERVIEW_PATH)
    total_plans = len(plan_to_sdgs)
    print(f"  共 {len(overview)} 個 SDG，{total_plans} 筆計畫記錄（去重後）")

    print("載入 MD 檔案 ...")
    md_data = load_md_files(MD_DIR)
    print(f"  共 {len(md_data)} 個 MD 檔案")

    print("執行比對 ...")
    missing, extra, not_found = run_check(overview, plan_to_sdgs, md_data)

    print("寫入結果 ...")
    result_text = write_result(missing, extra, not_found, OUTPUT_PATH)
    print(f"\n結果已寫入: {OUTPUT_PATH}")
    print("\n" + "="*50)
    print(result_text)
