#!/usr/bin/env python3
"""
llm_wiki_concept_compiler.py — Phase 2 概念百科頁編譯器

讀取 llm_wiki_data/ 的計畫 Wiki 頁面，找出所有高頻概念，
讓 Gemini 融合多份計畫的描述，生成每個概念的「百科文章」。

輸出到 llm_wiki_data/concepts/

用法：
    python llm_wiki_concept_compiler.py              # 編譯全部概念
    python llm_wiki_concept_compiler.py --min 30     # 只編譯出現 30 次以上的
    python llm_wiki_concept_compiler.py --force      # 強制覆蓋已有
    python llm_wiki_concept_compiler.py --dry-run    # 預覽不呼叫 API
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

LLM_WIKI_DIR  = Path("llm_wiki_data")
CONCEPTS_DIR  = LLM_WIKI_DIR / "concepts"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL  = os.getenv("LLM_MODEL", "gemini-2.5-flash")
MIN_COUNT      = 15   # 預設最低出現次數

# ── Prompt ──────────────────────────────────────────────────────
CONCEPT_PROMPT = """\
你是台灣大學社會責任（USR）知識百科的資深主編。

以下是來自 {count} 份不同大學 USR 計畫書，對「{concept}」的描述與應用紀錄：

{descriptions}

請根據以上資料，撰寫一篇關於「{concept}」在台灣 USR 實踐脈絡中的百科文章。

**撰寫規則：**
- 融合多份計畫的角度，寫出這個概念在 USR 脈絡中的共同意義
- 所有相關概念、地名、機構都用 [[雙中括號]] 標記
- 簡潔有力，每個章節 3-5 句或條列式
- 直接輸出 Markdown，不要前言或解釋

**格式：**

# {concept}

## 定義與背景

（在台灣 USR 脈絡中的定義與重要性，2-3 句）

## 主要實踐樣貌

（整合多個計畫的具體做法，條列式，每點附 [[概念]] 連結）

## 核心連結概念

（與本概念最密切相關的其他知識節點，用 [[連結]] 標記並簡短說明）

## 代表性計畫與學校

（列出 3-5 個最具代表性的計畫，格式：**學校** — 計畫名稱）
"""


def _safe_read(path: Path) -> str | None:
    for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return None


def _extract_concept_entries(wiki_text: str, concept: str) -> list[str]:
    """
    從一份 wiki 頁面提取關於 concept 的描述。
    包含：核心概念章節中的說明行、以及計畫基本資訊。
    """
    entries = []

    # 取 H1 計畫名稱
    plan_m = re.search(r'^#\s+(.+)', wiki_text, re.MULTILINE)
    plan_name = plan_m.group(1).strip() if plan_m else ""

    # 取執行學校
    uni_m = re.search(r'\|\s*執行學校\s*\|\s*(.+?)\s*\|', wiki_text)
    uni = re.sub(r'\[\[([^\]]+)\]\]', r'\1', uni_m.group(1).strip()) if uni_m else ""

    header = f"【{uni}｜{plan_name}】" if uni or plan_name else ""

    # 核心概念與連結 section 中對應 concept 的說明行
    core_m = re.search(r'##\s*核心概念與連結(.+?)(?:^##|\Z)',
                       wiki_text, re.DOTALL | re.MULTILINE)
    if core_m:
        for line in core_m.group(1).split('\n'):
            if f'[[{concept}]]' in line and '—' in line:
                desc = line.split('—', 1)[-1].strip()
                if desc:
                    entries.append(f"{header}：{desc}")
                    return entries  # 一份計畫只取一條

    # 沒有專屬說明行，改取主要活動段落中含 [[concept]] 的句子
    for line in wiki_text.split('\n'):
        if f'[[{concept}]]' in line and len(line.strip()) > 20:
            clean = re.sub(r'\[\[([^\]]+)\]\]', r'\1', line).strip(' -•*')
            if clean:
                entries.append(f"{header}：{clean[:120]}")
                return entries

    return entries


def _gather_concept_data(concept: str) -> tuple[list[str], list[tuple[str, str]]]:
    """
    掃描所有 wiki 檔案，收集提到 concept 的描述。
    回傳 (descriptions, [(uni, plan), ...])
    """
    descriptions = []
    plans = []
    pattern = f'[[{concept}]]'

    for path in sorted(LLM_WIKI_DIR.rglob("*.md")):
        if path.parent.name == "concepts":
            continue
        text = _safe_read(path)
        if not text or pattern not in text:
            continue

        entries = _extract_concept_entries(text, concept)
        descriptions.extend(entries)

        # 收集代表性計畫資訊
        plan_m = re.search(r'^#\s+(.+)', text, re.MULTILINE)
        uni_m  = re.search(r'\|\s*執行學校\s*\|\s*(.+?)\s*\|', text)
        if plan_m and uni_m:
            uni  = re.sub(r'\[\[([^\]]+)\]\]', r'\1', uni_m.group(1).strip())
            plan = plan_m.group(1).strip()
            plans.append((uni, plan))

    return descriptions, plans


def _build_llm(model: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
    )


def _compile_concept(llm, concept: str, descriptions: list[str],
                     plans: list[tuple[str, str]]) -> str | None:
    from langchain_core.messages import HumanMessage

    # 最多取 40 條描述，避免 token 超限
    sample_desc = descriptions[:40]
    desc_text = "\n".join(f"- {d}" for d in sample_desc)

    prompt = CONCEPT_PROMPT.format(
        concept=concept,
        count=len(plans),
        descriptions=desc_text,
    )

    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        raw = res.content
        if isinstance(raw, list):
            raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
        return raw.strip()
    except Exception as e:
        print(f"    [API ERROR] {e}")
        return None


def _get_top_concepts(min_count: int) -> list[tuple[str, int]]:
    """掃描所有 wiki 檔，統計 [[concept]] 出現次數。"""
    from collections import Counter
    cnt: Counter = Counter()
    skip = re.compile(r'^SDG\d+$')

    for path in sorted(LLM_WIKI_DIR.rglob("*.md")):
        if path.parent.name == "concepts":
            continue
        text = _safe_read(path)
        if not text:
            continue
        for tag in re.findall(r'\[\[([^\]]+)\]\]', text):
            if not skip.match(tag) and len(tag) >= 2:
                cnt[tag] += 1

    return [(c, n) for c, n in cnt.most_common() if n >= min_count]


def main():
    parser = argparse.ArgumentParser(description="USR LLM Concept Compiler")
    parser.add_argument("--min",     type=int,   default=MIN_COUNT,
                        help=f"最低出現次數（預設 {MIN_COUNT}）")
    parser.add_argument("--force",   action="store_true", help="強制覆蓋已有")
    parser.add_argument("--dry-run", action="store_true", help="只預覽，不呼叫 API")
    parser.add_argument("--delay",   type=float, default=2.0, help="API 間隔秒數")
    parser.add_argument("--model",   type=str,   default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not LLM_WIKI_DIR.exists():
        print("[ERROR] 找不到 llm_wiki_data/，請先執行 llm_wiki_compiler.py")
        sys.exit(1)

    CONCEPTS_DIR.mkdir(exist_ok=True)

    concepts = _get_top_concepts(args.min)
    total = len(concepts)

    print("=" * 60)
    print("  USR LLM Concept Compiler (Phase 2)")
    print("=" * 60)
    print(f"  概念數量  : {total} 個（出現 ≥{args.min} 次）")
    print(f"  輸出目錄  : {CONCEPTS_DIR}")
    print(f"  模型      : {args.model}")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry-run] 待編譯概念：")
        for i, (c, n) in enumerate(concepts, 1):
            out = CONCEPTS_DIR / f"{c}.md"
            status = "✓ 已完成" if out.exists() else "  待編譯"
            print(f"  {i:>3}. [{status}] {c}（{n} 份計畫）")
        return

    llm = _build_llm(args.model)
    done, skipped, failed = 0, 0, 0

    for i, (concept, count) in enumerate(concepts, 1):
        out_path = CONCEPTS_DIR / f"{concept}.md"

        if out_path.exists() and not args.force:
            print(f"[{i:>3}/{total}] SKIP  {concept}（{count} 份）")
            skipped += 1
            continue

        print(f"[{i:>3}/{total}] 編譯  {concept}（{count} 份計畫）")
        descriptions, plans = _gather_concept_data(concept)

        if not descriptions:
            print(f"         ⚠ 無有效描述，略過")
            failed += 1
            continue

        wiki_md = _compile_concept(llm, concept, descriptions, plans)

        if wiki_md:
            out_path.write_text(wiki_md, encoding="utf-8")
            done += 1
            print(f"         ✓ {out_path.name}")
        else:
            failed += 1
            print(f"         ✗ 失敗")

        if i < total:
            time.sleep(args.delay)

    print("\n" + "=" * 60)
    print(f"  完成：成功 {done}，略過 {skipped}，失敗 {failed}")
    print(f"  輸出：{CONCEPTS_DIR.resolve()}")
    print("=" * 60)

    if done > 0:
        print("\n下一步：重新啟動 app.py，wiki-map 點擊節點將顯示百科文章")


if __name__ == "__main__":
    main()
