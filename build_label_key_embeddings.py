# -*- coding: utf-8 -*-
"""
預先計算 label_index.json 所有 key 的 embedding，
輸出至 114_output/label_key_embeddings.json。

執行方式：
    python build_label_key_embeddings.py
"""
import os
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from langchain_voyageai import VoyageAIEmbeddings

LABEL_INDEX = Path("114_output/label_index.json")
OUTPUT = Path("114_output/label_key_embeddings.json")

def main():
    if not LABEL_INDEX.exists():
        print(f"找不到 {LABEL_INDEX}")
        return

    data = json.loads(LABEL_INDEX.read_text(encoding="utf-8"))

    emb = VoyageAIEmbeddings(
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        model="voyage-4-large",
    )

    result: dict = {}
    for yr, entries in data.items():
        if not isinstance(entries, dict):
            continue
        keys = list(entries.keys())
        print(f"{yr} 年：{len(keys)} 個 key，開始 embed...")
        vecs = emb.embed_documents(keys)
        result[yr] = {k: v for k, v in zip(keys, vecs)}
        print(f"  完成")

    OUTPUT.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"\n完成！輸出至 {OUTPUT}")

if __name__ == "__main__":
    main()
