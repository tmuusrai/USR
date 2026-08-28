# -*- coding: utf-8 -*-
"""
預先掃描 FAISS vectorstore，建立 USR_TOPIC_KEYWORDS 詞 → chunk 清單，
輸出至 114_output/kw_chunks.json。

使用向量語意搜尋（similarity_search），每個 keyword 取前 600 個最相近 chunk。
每個 (keyword, plan) 最多保留 3 個 chunk，截短至 200 字。
每個 keyword 最多 50 件計畫（cap）。

執行方式：
    python build_kw_chunks.py
"""
import os
import sys
import re
import json
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings

from usr_topics import USR_TOPIC_KEYWORDS

# ── 路徑設定 ──────────────────────────────────────────
INDEX_DIR     = Path(os.getenv("INDEX_DIR",     "faiss_index"))
INDEX_DIR_113 = Path(os.getenv("INDEX_DIR_113", "faiss_index_113"))
OUTPUT_PATH   = Path("114_output/kw_chunks.json")

_PLAN_CODE_RE = re.compile(r'\b[A-Z]{1,3}\d{3,}-\d+-\d+[A-Z]?\b')
_STEM_CLEAN_RE = re.compile(r'\s*\(\d{3}USR-[^)]*\)\s*|\s*\(\d+\)$')

def _clean_plan_code(text: str) -> str:
    return _PLAN_CODE_RE.sub('', text)

def _clean_stem(stem: str) -> str:
    return _STEM_CLEAN_RE.sub('', stem).strip()


def _build_chunks(year: str, vs) -> dict[str, list[dict]]:
    """向量相似度搜尋：每個 keyword embed 一次，找語意相近的 chunk。
    每個 (keyword, plan) 最多保留 3 個 chunk，截短至 200 字。
    每個 keyword 最多 50 件計畫。
    """
    all_kws: list[str] = sorted({kw for kws in USR_TOPIC_KEYWORDS.values() for kw in kws})
    kw_result: dict[str, list[dict]] = {}
    plan_count: set[str] = set()
    t0 = time.perf_counter()
    K = 600

    for i, kw in enumerate(all_kws):
        docs = vs.similarity_search(kw, k=K)
        plan_chunks: dict[str, list[str]] = {}

        for doc in docs:
            src = doc.metadata.get("source", "")
            if "qa_custom" in src:
                continue
            text = _clean_plan_code(doc.page_content)
            text = re.sub(r'(?m)^\s*-{3,}\s*$\n?', '', text).strip()
            if text.count("姓名：") >= 2:
                continue
            if text.count("課程名稱") + text.count("修課人次") + text.count("修課人數") >= 2:
                continue
            stem = _clean_stem(Path(src).stem)
            parts = stem.split('_', 1)
            if len(parts) < 2:
                continue
            school = parts[0]
            if not any(s in school for s in ('大學', '學院', '科大', '科技大')):
                continue
            plan = f"{school}：{parts[1]}"
            plan_count.add(plan)

            chunks = plan_chunks.setdefault(plan, [])
            if len(chunks) < 3 and text[:200] not in chunks:
                chunks.append(text[:200])

        entries: list[dict] = []
        for plan, texts in list(plan_chunks.items())[:20]:
            school = plan.split('：', 1)[0]
            for t in texts:
                entries.append({"school": school, "plan": plan, "text": t, "hits": 1})
        if entries:
            kw_result[kw] = entries

        elapsed_s = round(time.perf_counter() - t0)
        print(f"  [{i+1}/{len(all_kws)}] {kw}：{len(plan_chunks)} 件（{elapsed_s}s）")

    elapsed = round((time.perf_counter() - t0) * 1000)
    total = sum(len(v) for v in kw_result.values())
    print(f"[BUILD] {year} 年：{len(kw_result)} 個詞，{len(plan_count)} 份計畫，"
          f"{total} entries，耗時 {elapsed}ms")
    return kw_result


def _load_vs(index_dir: Path) -> FAISS | None:
    if not index_dir.exists():
        print(f"  找不到索引目錄：{index_dir}")
        return None
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        model="voyage-multilingual-2",
    )
    try:
        vs = FAISS.load_local(
            str(index_dir), embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"  載入 {index_dir} 成功，{len(vs.docstore._dict)} 個 chunk")
        return vs
    except Exception as e:
        print(f"  載入失敗：{e}")
        return None


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if OUTPUT_PATH.exists():
        try:
            existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    for year, index_dir in [("114", INDEX_DIR)]:
        print(f"\n=== {year} 年 ===")
        vs = _load_vs(index_dir)
        if vs is None:
            continue
        chunks = _build_chunks(year, vs)
        existing[year] = chunks

    OUTPUT_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成！輸出至 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
