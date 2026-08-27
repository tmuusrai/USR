# -*- coding: utf-8 -*-
"""
預先掃描 FAISS vectorstore，建立 USR_TOPIC_KEYWORDS 詞 → chunk 清單，
輸出至 114_output/kw_chunks.json。

每個 (keyword, plan) 只保留命中次數最多的一個 chunk，截短至 400 字。

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
INDEX_DIR      = Path(os.getenv("INDEX_DIR",      "faiss_index"))
INDEX_DIR_113  = Path(os.getenv("INDEX_DIR_113",  "faiss_index_113"))
OUTPUT_PATH    = Path("114_output/kw_chunks_test.json")
LABEL_INDEX    = Path("114_output/label_index.json")

_PRIMARY_TOPICS = {
    "在地關懷", "環境永續", "健康促進與食品安全",
    "產業鏈結與經濟永續", "文化永續", "其他社會實踐",
}

_PLAN_CODE_RE = re.compile(r'\b[A-Z]{1,3}\d{3,}-\d+-\d+[A-Z]?\b')
# 清理 filename stem 的計畫代碼和 chunk 序號
_STEM_CLEAN_RE = re.compile(r'\s*\(\d{3}USR-[^)]*\)\s*|\s*\(\d+\)$')

def _clean_plan_code(text: str) -> str:
    return _PLAN_CODE_RE.sub('', text)

def _clean_stem(stem: str) -> str:
    return _STEM_CLEAN_RE.sub('', stem).strip()

_PUNCT_NORM_RE = re.compile(r'[‧·•．・]')

def _norm_plan(s: str) -> str:
    """正規化計畫名稱，消除標點字元差異。"""
    s = _PUNCT_NORM_RE.sub('·', s)
    return re.sub(r'\s+', ' ', s).strip()


def _build_kw_allowed(year: str, label_data: dict) -> dict[str, set[str]]:
    """依六大議題的 label_index，建立 keyword → 允許的學校集合。
    非六大議題的 keyword 不受限制（回傳空 set 代表全部允許）。
    """
    year_data = label_data.get(year, {})
    topic_plans: dict[str, set[str]] = {}
    for topic in _PRIMARY_TOPICS:
        entries = year_data.get(topic, [])
        topic_plans[topic] = {_norm_plan(p) for p in entries if isinstance(p, str)}

    kw_allowed: dict[str, set[str]] = {}
    for topic, kws in USR_TOPIC_KEYWORDS.items():
        if topic in _PRIMARY_TOPICS:
            allowed = topic_plans.get(topic, set())
            if allowed:  # 若無 label 資料（空 set），不限制來源
                for kw in kws:
                    kw_allowed[kw] = allowed
        # 非六大議題的 keyword 不加入 kw_allowed → 視為全校允許

    for topic in _PRIMARY_TOPICS:
        p = topic_plans.get(topic, set())
        print(f"  [{topic}] {len(p)} 件計畫")
    return kw_allowed


def _build_chunks(year: str, vs, kw_allowed: dict[str, set[str]]) -> dict[str, list[dict]]:
    """向量相似度搜尋：每個 keyword embed 一次，找語意相近的 chunk。
    六大議題 keyword 只取該議題的計畫；其餘 keyword 無限制。
    每個 (keyword, plan) 只保留相似度最高的一個 chunk，截短至 400 字。
    """
    all_kws: list[str] = sorted({kw for kws in USR_TOPIC_KEYWORDS.values() for kw in kws})
    kw_result: dict[str, list[dict]] = {}
    plan_count: set[str] = set()
    t0 = time.perf_counter()
    K = 600  # 每個 keyword 取前 K 個最相近的 chunk

    for i, kw in enumerate(all_kws):
        allowed = kw_allowed.get(kw)  # None = 無限制
        docs = vs.similarity_search(kw, k=K)
        plan_chunks: dict[str, list[str]] = {}  # plan → 最多 3 個 chunk text

        for doc in docs:
            src = doc.metadata.get("source", "")
            if "qa_custom" in src:
                continue
            text = _clean_plan_code(doc.page_content)
            text = re.sub(r'(?m)^\s*-{3,}\s*$\n?', '', text).strip()
            # 名單型 chunk 跳過
            if text.count("姓名：") >= 2:
                continue
            # 課程表格型 chunk 跳過
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

            # 六大議題 keyword：只允許該議題的計畫
            if allowed is not None and _norm_plan(plan) not in allowed:
                continue

            # 每個計畫最多存 3 個不同 chunk（相似度由高到低）
            chunks = plan_chunks.setdefault(plan, [])
            if len(chunks) < 3 and text[:200] not in chunks:
                chunks.append(text[:200])

        # 每個 keyword 最多 50 件計畫，每件拆成多筆 entry
        entries: list[dict] = []
        for plan, texts in list(plan_chunks.items())[:50]:
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

    label_data: dict = {}
    if LABEL_INDEX.exists():
        try:
            label_data = json.loads(LABEL_INDEX.read_text(encoding="utf-8"))
            print(f"載入 label_index：{LABEL_INDEX}")
        except Exception as e:
            print(f"  label_index 載入失敗：{e}")

    for year, index_dir in [("114", INDEX_DIR)]:
        print(f"\n=== {year} 年 ===")
        vs = _load_vs(index_dir)
        if vs is None:
            continue
        kw_allowed = _build_kw_allowed(year, label_data)
        chunks = _build_chunks(year, vs, kw_allowed)
        existing[year] = chunks

    OUTPUT_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成！輸出至 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
