"""
build_keyword_index.py
從 114md/ 和 113md/ 的計畫書中提取關鍵字，建立搜尋索引。
輸出：
  keyword_index.json       — {keyword: [school, ...]}  反向索引
  keyword_groundtruth.csv  — keyword, schools  驗證用對照表
"""

import os
import re
import json
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import google.genai as genai

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL      = "gemini-flash-latest"  # 批次處理用輕量模型
MAX_WORKERS    = 10   # 並行 LLM 呼叫數
READ_CHARS     = 3000 # 每份計畫書讀前幾字

_client = genai.Client(api_key=GOOGLE_API_KEY)

def _llm_invoke(prompt: str) -> str:
    res = _client.models.generate_content(model=LLM_MODEL, contents=prompt)
    return res.text or ""

EXTRACT_PROMPT = """你是 USR 計畫書分析專家。
請從以下計畫書內容，提取 5~8 個最能代表此計畫主題的關鍵詞。

要求：
- 每個關鍵詞為 2~8 個字的中文詞組
- 聚焦於主題領域（例如：海洋保育、高齡照護、食農教育、原住民文化、地方創生、環境永續）
- 聚焦於執行場域特色（例如：溪流生態、偏鄉教育、離島社區）
- 不要輸出「USR」、「大學社會責任」、「計畫執行」等通用詞
- 輸出 JSON 陣列，不要其他文字

計畫書內容：
{content}

只輸出 JSON 陣列，例如：["食農教育", "有機農業", "農村體驗", "偏鄉小學", "地方創生"]"""


def _extract_school_from_stem(stem: str) -> str:
    """從檔名 stem 取出學校名稱（底線前）。"""
    return stem.split("_")[0].strip()


def _read_md_head(path: Path, chars: int = READ_CHARS) -> str:
    """讀取 MD 前 N 字，嘗試多種編碼。"""
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            text = path.read_text(encoding=enc)
            return text[:chars]
        except Exception:
            continue
    return ""


def _extract_keywords_for_file(md_path: Path) -> tuple[str, str, list[str]]:
    """
    回傳 (school, plan_name, keywords)
    """
    school    = _extract_school_from_stem(md_path.stem)
    plan_name = re.sub(r'\([^)]*\)|_formatted', '', md_path.stem).strip("_").strip()
    plan_name = plan_name[len(school):].lstrip("_").strip()

    content = _read_md_head(md_path)
    if not content:
        return school, plan_name, []

    prompt = EXTRACT_PROMPT.format(content=content)
    try:
        text = _llm_invoke(prompt).strip()
        m    = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            kws = json.loads(m.group())
            kws = [k for k in kws if isinstance(k, str) and 2 <= len(k) <= 20]
            print(f"  [OK] {school}: {kws}")
            return school, plan_name, kws
    except Exception as e:
        print(f"  [FAIL] {school}: {e}")
    return school, plan_name, []


def build_index(year: str = "114"):
    md_dir = Path(f"{year}md")
    if not md_dir.exists():
        print(f"[WARN] {md_dir} 不存在，跳過 {year} 年。")
        return {}

    md_files = sorted(md_dir.glob("*.md"))
    print(f"\n[BUILD] {year} 年，共 {len(md_files)} 份計畫書，並行 {MAX_WORKERS} 個 LLM worker...\n")

    results = []  # [(school, plan_name, keywords), ...]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_extract_keywords_for_file, p): p for p in md_files}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"  [FAIL] future: {e}")

    # 建立反向索引：keyword → [school, ...]
    keyword_index: dict[str, list[str]] = {}
    for school, plan_name, kws in results:
        for kw in kws:
            keyword_index.setdefault(kw, [])
            if school not in keyword_index[kw]:
                keyword_index[kw].append(school)

    # 按出現學校數降冪排序
    keyword_index = dict(
        sorted(keyword_index.items(), key=lambda x: len(x[1]), reverse=True)
    )

    return keyword_index


def save_outputs(index_114: dict, index_113: dict):
    # 合併兩年（以 114 為主，113 補充）
    merged: dict[str, list[str]] = {}
    for kw, schools in {**index_113, **index_114}.items():
        merged.setdefault(kw, [])
        for s in schools:
            if s not in merged[kw]:
                merged[kw].append(s)

    # 儲存 keyword_index.json（分年版）
    out = {"114": index_114, "113": index_113, "merged": merged}
    with open("keyword_index.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] keyword_index.json  ({len(merged)} 個關鍵字)")

    # 儲存 keyword_groundtruth.csv（驗證用）
    with open("keyword_groundtruth.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["關鍵字", "學校數", "學校清單（114年）"])
        for kw in sorted(index_114, key=lambda k: len(index_114[k]), reverse=True):
            schools = index_114[kw]
            writer.writerow([kw, len(schools), "、".join(schools)])
    print(f"[SAVE] keyword_groundtruth.csv")


if __name__ == "__main__":
    index_114 = build_index("114")
    index_113 = build_index("113")
    save_outputs(index_114, index_113)
    print("\n完成！")
