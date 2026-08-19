# -*- coding: utf-8 -*-
"""
預先掃描 FAISS vectorstore，建立 USR_TOPIC_KEYWORDS 詞 → chunk 清單，
輸出至 114_output/kw_chunks.json。

對 6 個有 label 的官方議題，keyword → plan 對應只保留 label_index 裡的計畫。

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
LABEL_PATH    = Path("114_output/label_index.json")

_PLAN_CODE_RE = re.compile(r'\b[A-Z]{1,3}\d{3,}-\d+-\d+[A-Z]?\b')
# 清理 filename stem 的計畫代碼和 chunk 序號
_STEM_CLEAN_RE = re.compile(r'\s*\(\d{3}USR-[^)]*\)\s*|\s*\(\d+\)$')

def _clean_plan_code(text: str) -> str:
    return _PLAN_CODE_RE.sub('', text)

def _clean_stem(stem: str) -> str:
    return _STEM_CLEAN_RE.sub('', stem).strip()

# 有 label 的官方 USR 議題
_LABELED_TOPICS = {
    "在地關懷", "環境永續", "健康促進與食品安全",
    "產業鏈結與經濟永續", "文化永續", "其他社會實踐",
}

# keyword → 所屬官方議題（只建 6 個 topic 的對應）
def _build_kw_to_topic() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for topic, kws in USR_TOPIC_KEYWORDS.items():
        if topic in _LABELED_TOPICS:
            for kw in kws:
                mapping[kw] = topic
    return mapping


def _load_label_plans(label_path: Path, year: str) -> dict[str, set[str]]:
    """載入 label_index.json，回傳 topic → set of plan_keys。"""
    if not label_path.exists():
        print(f"  找不到 {label_path}，不做 label 過濾")
        return {}
    data = json.loads(label_path.read_text(encoding="utf-8"))
    yr = data.get(year, data)
    result: dict[str, set[str]] = {}
    for topic in _LABELED_TOPICS:
        entries = yr.get(topic, [])
        result[topic] = set(entries if isinstance(entries[0], str) else [e if isinstance(e, str) else f"{e['school']}：{e['plan']}" for e in entries]) if entries else set()
        print(f"  label[{topic}]：{len(result[topic])} 件")
    return result


def _build_chunks(year: str, vs, label_plans: dict[str, set[str]]) -> dict[str, list[dict]]:
    """掃 vectorstore，建立 keyword → chunk 清單。
    - 6 個官方議題的 keyword：只保留 label 裡有的計畫
    - 其他 keyword：保留所有命中計畫
    每個 (keyword, plan) 只保留命中次數最多的一個 chunk，截短至 400 字。
    """
    all_kws: set[str] = {kw for kws in USR_TOPIC_KEYWORDS.values() for kw in kws}
    kw_to_topic = _build_kw_to_topic()
    kw_best: dict[str, dict[str, dict]] = {kw: {} for kw in all_kws}
    plan_count: set[str] = set()
    t0 = time.perf_counter()

    for doc in vs.docstore._dict.values():
        src = doc.metadata.get("source", "")
        if "qa_custom" in src:
            continue
        text = _clean_plan_code(doc.page_content)
        # 移除 --- 分隔線
        text = re.sub(r'(?m)^\s*-{3,}\s*$\n?', '', text).strip()
        stem = _clean_stem(Path(src).stem)
        parts = stem.split('_', 1)
        if len(parts) < 2:
            continue
        plan = f"{parts[0]}：{parts[1]}"
        school = parts[0]
        plan_count.add(plan)

        for kw in all_kws:
            hits = text.count(kw)
            if hits == 0:
                continue

            # 官方議題 keyword：只保留 label 裡有的計畫
            topic = kw_to_topic.get(kw)
            if topic and plan not in label_plans.get(topic, set()):
                continue

            cur = kw_best[kw].get(plan)
            if cur is None or hits > cur["hits"]:
                kw_best[kw][plan] = {
                    "school": school,
                    "plan": plan,
                    "text": text[:400],
                    "hits": hits,
                }

    elapsed = round((time.perf_counter() - t0) * 1000)
    kw_result = {kw: list(plans.values()) for kw, plans in kw_best.items() if plans}
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

    for year, index_dir in [("114", INDEX_DIR), ("113", INDEX_DIR_113)]:
        print(f"\n=== {year} 年 ===")
        vs = _load_vs(index_dir)
        if vs is None:
            continue
        print(f"  載入 label_index...")
        label_plans = _load_label_plans(LABEL_PATH, year)
        chunks = _build_chunks(year, vs, label_plans)
        existing[year] = chunks

    OUTPUT_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成！輸出至 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
