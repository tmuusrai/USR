# -*- coding: utf-8 -*-
"""
針對每件國外計畫，用其 overseas_countries 名稱去 FAISS 搜尋，
找含有國家名稱的段落，存進 kw_chunks["114"]["國外"]。
"""
import sys
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings

LOC_IDX   = Path("114_output/location_index.json")
KW_CHUNKS = Path("114_output/kw_chunks.json")
INDEX_DIR = Path("faiss_index")
YEAR      = "114"
K         = 50   # 每次搜幾筆
MAX_PER_PLAN = 3  # 每件計畫最多幾個 chunk

def _normalize_plan(s: str) -> str:
    return re.sub(r'[\s　]+', '', s).lower()

def main():
    print("載入 FAISS 索引...")
    emb = VoyageAIEmbeddings(
        voyage_api_key=os.environ["VOYAGE_API_KEY"],
        model="voyage-4-large",
    )
    vs = FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
    print("FAISS 載入完成")

    loc_data = json.loads(LOC_IDX.read_text(encoding="utf-8"))
    loc_plans = loc_data.get(YEAR, {}).get("plans", {})

    # 找出所有有 overseas_countries 的計畫
    overseas = {
        plan_key: info["overseas_countries"]
        for plan_key, info in loc_plans.items()
        if info.get("overseas_countries")
    }
    print(f"國外場域計畫：{len(overseas)} 件\n")

    kw_data = json.loads(KW_CHUNKS.read_text(encoding="utf-8"))
    year_chunks = kw_data.setdefault(YEAR, {})

    overseas_entries: list[dict] = []

    for plan_key, countries in sorted(overseas.items()):
        school, _, plan_name = plan_key.partition("：")
        found_chunks: list[str] = []

        # 用每個國家名稱去搜
        for country in countries:
            query = f"{plan_name} {country} 國外實踐場域"
            docs = vs.similarity_search(query, k=K)
            for doc in docs:
                text = doc.page_content
                meta_plan = doc.metadata.get("plan_name", "") or doc.metadata.get("source", "")

                # 確認是同一件計畫的文字
                plan_match = (
                    _normalize_plan(plan_key) in _normalize_plan(meta_plan)
                    or _normalize_plan(meta_plan) in _normalize_plan(plan_key)
                    or _normalize_plan(school) in _normalize_plan(meta_plan)
                )
                if not plan_match:
                    continue

                # 確認含有國家名稱
                if country not in text:
                    continue

                if text not in found_chunks:
                    found_chunks.append(text)

                if len(found_chunks) >= MAX_PER_PLAN:
                    break

            if len(found_chunks) >= MAX_PER_PLAN:
                break

        if found_chunks:
            for txt in found_chunks:
                overseas_entries.append({
                    "school": school,
                    "plan": plan_key,
                    "text": txt,
                    "hits": 1,
                })
            print(f"  OK {plan_key}：{len(found_chunks)} chunk（國家：{'、'.join(countries)}）")
        else:
            print(f"  -- {plan_key}：0 chunk（找不到含國家名稱的段落）")

    year_chunks["國外"] = overseas_entries
    kw_data[YEAR] = year_chunks
    KW_CHUNKS.write_text(json.dumps(kw_data, ensure_ascii=False), encoding="utf-8")
    print(f"\nkw_chunks.json 已更新，國外 entries：{len(overseas_entries)} 個")


if __name__ == "__main__":
    main()
