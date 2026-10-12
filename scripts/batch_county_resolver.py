#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_county_resolver.py
對 scripts/batches/ 下每個未處理的 batch，呼叫 Gemini 查詢台灣縣市，
結果存到 scripts/results/batch_XX_results.json，
最後將所有結果套用回 114_output/location_index.json。

用法：
    python scripts/batch_county_resolver.py           # 處理全部未完成的 batch
    python scripts/batch_county_resolver.py --apply   # 只套用已有結果，不呼叫 API
    python scripts/batch_county_resolver.py --dry-run # 僅列出待處理 batch
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

BATCHES_DIR  = Path(__file__).parent / "batches"
RESULTS_DIR  = Path(__file__).parent / "results"
LOCATION_IDX = Path(__file__).parent.parent / "114_output" / "location_index.json"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL  = os.getenv("LLM_MODEL", "gemini-2.5-flash")

COUNTY_NAMES = (
    "台北、新北、基隆、桃園、新竹、苗栗、台中、彰化、南投、"
    "雲林、嘉義、台南、高雄、屏東、台東、花蓮、宜蘭、澎湖、金門、馬祖"
)

PROMPT_TMPL = """\
以下是台灣大學社會責任（USR）計畫的合作地點或機構名稱列表。
請判斷每個名稱在台灣的哪個縣市（用短名）以及鄉鎮區（可留空）。

縣市短名請使用：{counties}

輸出 **純 JSON**，格式如下（不要任何前言或說明文字）：
{{
  "地點名稱A": {{"county": "縣市短名", "district": "鄉鎮區或空字串"}},
  "地點名稱B": {{"county": "", "district": ""}},
  ...
}}

若確實無法判斷，county 和 district 都填空字串。

地點列表：
{locations}
"""


def call_gemini(locations: list[str], model: str) -> dict:
    """呼叫 Gemini API，回傳 location -> {{county, district}} 字典。"""
    try:
        import google.generativeai as genai
    except ImportError:
        print("請先安裝：pip install google-generativeai", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=GOOGLE_API_KEY)
    client = genai.GenerativeModel(model)

    locs_text = "\n".join(f"- {loc}" for loc in locations)
    prompt = PROMPT_TMPL.format(counties=COUNTY_NAMES, locations=locs_text)

    response = client.generate_content(prompt)
    raw = response.text.strip()

    # 去掉可能的 ```json ... ``` 包裝
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    return json.loads(raw)


def apply_results_to_index(all_results: dict) -> tuple[int, int]:
    """將縣市結果套用到 location_index.json，回傳 (填入數, 跳過數)。"""
    data = json.loads(LOCATION_IDX.read_text(encoding="utf-8"))
    plans = data.get("114", {}).get("plans", {})

    filled = skipped = 0
    for plan_data in plans.values():
        for field in plan_data.get("fields", []):
            if field.get("county"):
                continue
            loc = field.get("location", "")
            if loc in all_results:
                result = all_results[loc]
                county = result.get("county", "")
                if county:
                    field["county"] = county
                    if result.get("district"):
                        field["district"] = result["district"]
                    filled += 1
                else:
                    skipped += 1

    LOCATION_IDX.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return filled, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply",   action="store_true", help="只套用現有結果，不呼叫 API")
    parser.add_argument("--dry-run", action="store_true", help="列出待處理 batch，不執行")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--delay",   type=float, default=2.0, help="每個 batch 後等待秒數")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    # 找出所有 batch 和已有 results
    batch_files = sorted(BATCHES_DIR.glob("batch_*.json"))
    done = {f.stem.replace("_results", "") for f in RESULTS_DIR.glob("batch_*_results.json")}

    pending = [bf for bf in batch_files if bf.stem not in done]
    print(f"Batch 總計：{len(batch_files)}，已完成：{len(done)}，待處理：{len(pending)}")

    if args.dry_run:
        for bf in pending:
            with open(bf, encoding="utf-8") as f:
                locs = json.load(f)
            print(f"  {bf.name}: {len(locs)} locations")
        return

    # 處理未完成的 batch
    if not args.apply:
        if not GOOGLE_API_KEY:
            print("錯誤：GOOGLE_API_KEY 未設定", file=sys.stderr)
            sys.exit(1)

        for i, bf in enumerate(pending, 1):
            with open(bf, encoding="utf-8") as f:
                locations = json.load(f)
            print(f"\n[{i}/{len(pending)}] 處理 {bf.name}（{len(locations)} 個地點）…", flush=True)

            try:
                result = call_gemini(locations, args.model)
            except Exception as e:
                print(f"  !! 錯誤：{e}，跳過此 batch", file=sys.stderr)
                continue

            result_path = RESULTS_DIR / bf.name.replace(".json", "_results.json")
            result_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  ✓ 儲存 {result_path.name}（{len(result)} 筆）")

            if i < len(pending):
                time.sleep(args.delay)

    # 套用所有 results 到 location_index.json
    all_results: dict = {}
    for rf in sorted(RESULTS_DIR.glob("batch_*_results.json")):
        with open(rf, encoding="utf-8") as f:
            all_results.update(json.load(f))
    print(f"\n套用結果：共 {len(all_results)} 筆地點資料…")

    filled, skipped = apply_results_to_index(all_results)
    print(f"  填入縣市：{filled} 筆")
    print(f"  無法判斷（縣市仍空）：{skipped} 筆")
    print(f"location_index.json 已更新。")


if __name__ == "__main__":
    main()
