#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
county_second_pass.py
對 location_index.json 中仍缺縣市的地點，帶入所屬計畫名稱當 context，
再次呼叫 Gemini 判斷縣市，並直接更新 location_index.json。

用法：
    python scripts/county_second_pass.py           # 執行第二輪查詢並套用
    python scripts/county_second_pass.py --dry-run # 只列出待處理數量
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

LOCATION_IDX   = Path(__file__).parent.parent / "114_output" / "location_index.json"
SECOND_RESULTS = Path(__file__).parent / "results" / "second_pass_results.json"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL  = os.getenv("LLM_MODEL", "gemini-2.5-flash")
BATCH_SIZE     = 40

COUNTY_NAMES = (
    "台北、新北、基隆、桃園、新竹、苗栗、台中、彰化、南投、"
    "雲林、嘉義、台南、高雄、屏東、台東、花蓮、宜蘭、澎湖、金門、馬祖"
)

PROMPT_TMPL = """\
以下是台灣大學社會責任（USR）計畫的合作地點清單。
每個地點會附上所屬計畫的學校名稱，供你判斷縣市時參考。

縣市短名請使用：{counties}

輸出 **純 JSON**，格式（不要任何前言或說明文字）：
{{
  "地點名稱A": {{"county": "縣市短名", "district": "鄉鎮區或空字串"}},
  "地點名稱B": {{"county": "", "district": ""}},
  ...
}}

若確實無法判斷（如全國性機構、活動名稱、外國地點），county 和 district 填空字串。

地點清單（格式：地點名稱 ← 所屬計畫學校）：
{locations}
"""


def call_gemini(items: list[tuple[str, str]], model: str) -> dict:
    """items: [(location, university), ...]，回傳 location -> {county, district}。"""
    try:
        import google.generativeai as genai
    except ImportError:
        print("請先安裝：pip install google-generativeai", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=GOOGLE_API_KEY)
    client = genai.GenerativeModel(model)

    locs_text = "\n".join(f"- {loc} ← {univ}" for loc, univ in items)
    prompt = PROMPT_TMPL.format(counties=COUNTY_NAMES, locations=locs_text)

    response = client.generate_content(prompt)
    raw = response.text.strip()

    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    return json.loads(raw)


def collect_missing(data: dict) -> dict[str, set[str]]:
    """回傳 {location: {所屬計畫的學校名稱, ...}}。"""
    loc_to_schools: dict[str, set[str]] = {}
    for plan_key, plan_data in data["114"]["plans"].items():
        school = plan_key.split("：")[0]
        for field in plan_data.get("fields", []):
            if not field.get("county") and field.get("location"):
                loc = field["location"]
                loc_to_schools.setdefault(loc, set()).add(school)
    return loc_to_schools


def apply_results(data: dict, results: dict) -> int:
    filled = 0
    for plan_data in data["114"]["plans"].values():
        for field in plan_data.get("fields", []):
            if field.get("county"):
                continue
            loc = field.get("location", "")
            if loc in results and results[loc].get("county"):
                field["county"] = results[loc]["county"]
                if results[loc].get("district"):
                    field["district"] = results[loc]["district"]
                filled += 1
    return filled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--delay",   type=float, default=2.0)
    args = parser.parse_args()

    data = json.loads(LOCATION_IDX.read_text(encoding="utf-8"))
    loc_to_schools = collect_missing(data)

    # 排除明顯無法判斷的（日期格式、空字串）
    import re
    invalid = {loc for loc in loc_to_schools if re.match(r'^\d{4}[/-]', loc) or not loc.strip()}
    loc_to_schools = {k: v for k, v in loc_to_schools.items() if k not in invalid}

    print(f"仍缺縣市：{len(loc_to_schools)} 個唯一地點（排除 {len(invalid)} 個無效項）")

    if args.dry_run:
        return

    if not GOOGLE_API_KEY:
        print("錯誤：GOOGLE_API_KEY 未設定", file=sys.stderr)
        sys.exit(1)

    # 載入已有的第二輪結果（支援中斷續跑）
    existing: dict = {}
    if SECOND_RESULTS.exists():
        existing = json.loads(SECOND_RESULTS.read_text(encoding="utf-8"))
        print(f"載入已有第二輪結果：{len(existing)} 筆")

    pending = [(loc, next(iter(schools)))
               for loc, schools in loc_to_schools.items()
               if loc not in existing]
    print(f"本次待查詢：{len(pending)} 個地點，每批 {BATCH_SIZE} 個")

    new_results: dict = dict(existing)
    batches = [pending[i:i+BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]

    for i, batch in enumerate(batches, 1):
        print(f"\n[{i}/{len(batches)}] 查詢 {len(batch)} 個地點…", flush=True)
        try:
            result = call_gemini(batch, args.model)
            new_results.update(result)
            # 每批後立即存檔（中斷保護）
            SECOND_RESULTS.write_text(
                json.dumps(new_results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            filled_this = sum(1 for v in result.values() if v.get("county"))
            print(f"  ✓ 本批填入 {filled_this}/{len(batch)} 筆縣市")
        except Exception as e:
            print(f"  !! 錯誤：{e}，跳過此批", file=sys.stderr)

        if i < len(batches):
            time.sleep(args.delay)

    # 套用到 location_index.json
    print(f"\n套用第二輪結果到 location_index.json…")
    filled = apply_results(data, new_results)
    LOCATION_IDX.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  新增填入：{filled} 筆")
    print("完成！")


if __name__ == "__main__":
    main()
