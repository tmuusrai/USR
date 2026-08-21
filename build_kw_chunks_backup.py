# -*- coding: utf-8 -*-
"""
測試版：六大官方議題的關鍵字只從 label_index 對應計畫找 chunk，
使用向量相似度搜尋（語意相關），不限精確字串比對。
輸出至 114_output/kw_chunks_test.json。

執行方式：
    python build_kw_chunks_backup.py
"""
import os, sys, re, json, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings
from usr_topics import USR_TOPIC_KEYWORDS

INDEX_DIR     = Path(os.getenv("INDEX_DIR",     "faiss_index"))
INDEX_DIR_113 = Path(os.getenv("INDEX_DIR_113", "faiss_index_113"))
LABEL_PATH    = Path("114_output/label_index.json")
OUTPUT_PATH   = Path("114_output/kw_chunks_test.json")

_PLAN_CODE_RE  = re.compile(r'\b[A-Z]{1,3}\d{3,}-\d+-\d+[A-Z]?\b')
_STEM_CLEAN_RE = re.compile(r'\s*\(\d{3}USR-[^)]*\)\s*|\s*\(\d+\)$|_formatted(?:\(\d+\))?$')

def _clean_plan_code(text): return _PLAN_CODE_RE.sub('', text)
def _clean_stem(stem): return _STEM_CLEAN_RE.sub('', stem).strip()

_LABELED_TOPICS = {
    "在地關懷", "環境永續", "健康促進與食品安全",
    "產業鏈結與經濟永續", "文化永續", "其他社會實踐",
}

def _load_label_plans(year: str) -> dict[str, set[str]]:
    if not LABEL_PATH.exists():
        print(f"  找不到 {LABEL_PATH}，不做 label 過濾")
        return {}
    data = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
    yr = data.get(year, {})
    result = {}
    for topic in _LABELED_TOPICS:
        entries = yr.get(topic, [])
        result[topic] = set(entries) if entries else set()
        print(f"  label[{topic}]：{len(result[topic])} 件")
    return result

def _build_kw_to_topic() -> dict[str, str]:
    mapping = {}
    for topic, kws in USR_TOPIC_KEYWORDS.items():
        if topic in _LABELED_TOPICS:
            for kw in kws:
                mapping[kw] = topic
    return mapping

def _build_chunks(year: str, vs, label_plans: dict[str, set[str]]) -> dict[str, list[dict]]:
    all_kws = sorted({kw for kws in USR_TOPIC_KEYWORDS.values() for kw in kws})
    kw_to_topic = _build_kw_to_topic()
    kw_result: dict[str, list[dict]] = {}
    plan_count: set[str] = set()
    total_docs = len(vs.docstore._dict)
    t0 = time.perf_counter()

    for i, kw in enumerate(all_kws):
        topic = kw_to_topic.get(kw)
        allowed_plans = label_plans.get(topic, set()) if topic else None

        # 向量相似度搜尋：query embed 一次，FAISS 本地比對全部 chunk
        docs = vs.similarity_search(kw, k=total_docs)

        plan_best: dict[str, dict] = {}
        for doc in docs:
            src = doc.metadata.get("source", "")
            if "qa_custom" in src:
                continue
            stem = _clean_stem(Path(src).stem)
            parts = stem.split('_', 1)
            if len(parts) < 2:
                continue
            plan = f"{parts[0]}：{parts[1]}"
            plan_count.add(plan)

            # 六大官方議題：只保留 label 裡的計畫
            if allowed_plans is not None and plan not in allowed_plans:
                continue

            # 每個計畫只保留相似度最高的 chunk（similarity_search 回傳已排序）
            if plan not in plan_best:
                text = _clean_plan_code(doc.page_content)
                text = re.sub(r'(?m)^\s*-{3,}\s*$\n?', '', text).strip()
                plan_best[plan] = {
                    "school": parts[0],
                    "plan": plan,
                    "text": text[:400],
                    "hits": 1,
                }

        if plan_best:
            kw_result[kw] = list(plan_best.values())

        elapsed_s = round(time.perf_counter() - t0)
        print(f"  [{i+1}/{len(all_kws)}] {kw}：{len(plan_best)} 件（{elapsed_s}s）")

    elapsed = round((time.perf_counter() - t0) * 1000)
    total = sum(len(v) for v in kw_result.values())
    print(f"[BUILD] {year}年：{len(kw_result)} 個詞，{len(plan_count)} 份計畫，{total} entries，耗時 {elapsed}ms")
    return kw_result

def _load_vs(index_dir: Path):
    if not index_dir.exists():
        print(f"  找不到 {index_dir}")
        return None
    emb = VoyageAIEmbeddings(voyage_api_key=os.getenv("VOYAGE_API_KEY", ""), model="voyage-multilingual-2")
    try:
        vs = FAISS.load_local(str(index_dir), emb, allow_dangerous_deserialization=True)
        print(f"  載入 {index_dir} 成功，{len(vs.docstore._dict)} 個 chunk")
        return vs
    except Exception as e:
        print(f"  載入失敗：{e}")
        return None

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = {}

    for year, index_dir in [("114", INDEX_DIR), ("113", INDEX_DIR_113)]:
        print(f"\n=== {year} 年 ===")
        vs = _load_vs(index_dir)
        if vs is None:
            continue
        label_plans = _load_label_plans(year)
        chunks = _build_chunks(year, vs, label_plans)
        existing[year] = chunks

    OUTPUT_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成！輸出至 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
