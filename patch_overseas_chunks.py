# -*- coding: utf-8 -*-
"""
從現有 kw_chunks.json 中，把 location_index overseas_countries 的 15 件計畫
已有的 chunk 複製到 kw_chunks["114"]["國外"]，不需重跑 FAISS。
"""
import sys
import json
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

KW_CHUNKS    = Path("114_output/kw_chunks.json")
LOC_IDX      = Path("114_output/location_index.json")
YEAR         = "114"
MAX_PER_PLAN = 3   # 每件計畫最多保留幾個 chunk

_STEM_CLEAN_RE = re.compile(r'\s*\(\d{3}USR-[^)]*\)\s*|\s*_formatted(?:\(\d+\))?\s*')

def _normalize(s: str) -> str:
    """激進正規化：去空白、統一標點，用於模糊比對 plan 名稱。"""
    s = _STEM_CLEAN_RE.sub('', s)
    s = re.sub(r'[\s　]+', '', s)          # 去所有空白
    s = re.sub(r'[‧·•．・·]', '·', s)    # 統一中點
    s = re.sub(r'[×x✕]', 'x', s, flags=re.IGNORECASE)  # 統一乘號
    s = re.sub(r'[\-－]', '-', s)          # 統一減號
    return s.strip('_ ').strip().lower()


def _build_norm_index(year_chunks: dict) -> dict[str, str]:
    """建立 normalized plan name → 原始 plan name 的查找表。"""
    norm_to_raw: dict[str, str] = {}
    for entries in year_chunks.values():
        for e in entries:
            if not isinstance(e, dict):
                continue
            raw = e.get("plan", "")
            if raw:
                norm_to_raw[_normalize(raw)] = raw
    return norm_to_raw


def _match_plan(loc_plan: str, norm_to_raw: dict[str, str]) -> str | None:
    """嘗試精確 → 正規化 → 學校+部分計畫名 三層比對。"""
    # 1. 精確
    if loc_plan in norm_to_raw.values():
        return loc_plan

    # 2. 正規化比對
    n = _normalize(loc_plan)
    if n in norm_to_raw:
        return norm_to_raw[n]

    # 3. 學校相同 + 計畫名部分包含（處理截斷/OCR 差異）
    school, _, plan_part = loc_plan.partition("：")
    n_plan = _normalize(plan_part)
    best = None
    best_score = 0
    for raw in norm_to_raw.values():
        rs, _, rp = raw.partition("：")
        if _normalize(rs) != _normalize(school):
            continue
        rp_n = _normalize(rp)
        if n_plan in rp_n or rp_n in n_plan:
            score = len(min(n_plan, rp_n, key=len))
            if score > best_score:
                best, best_score = raw, score
    return best


def main():
    loc_data = json.loads(LOC_IDX.read_text(encoding="utf-8"))
    overseas_plans: set[str] = {
        plan
        for plan, info in loc_data.get(YEAR, {}).get("plans", {}).items()
        if info.get("overseas_countries")
    }
    print(f"國外場域計畫：{len(overseas_plans)} 件")

    kw_data = json.loads(KW_CHUNKS.read_text(encoding="utf-8"))
    year_chunks: dict[str, list[dict]] = kw_data.get(YEAR, {})

    # 建正規化查找表（排除 "國外" 本身）
    norm_to_raw = _build_norm_index(
        {k: v for k, v in year_chunks.items() if k != "國外"}
    )

    # 建 matched_plan → 原始 kw_chunks plan key 的對應
    plan_match: dict[str, str] = {}   # loc_plan → kw_plan
    for loc_plan in sorted(overseas_plans):
        matched = _match_plan(loc_plan, norm_to_raw)
        if matched:
            plan_match[loc_plan] = matched
            if matched != loc_plan:
                print(f"  ~ {loc_plan}\n    -> {matched}")
        else:
            print(f"  ! 無 chunk：{loc_plan}")

    # 收集各計畫已有的 chunk（去重，每計畫最多 MAX_PER_PLAN 個）
    plan_chunks: dict[str, list[dict]] = {}   # loc_plan → chunk list
    for kw_key, entries in year_chunks.items():
        if kw_key == "國外":
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            kw_plan = e.get("plan", "")
            # 找對應的 loc_plan
            loc_plan = next((lp for lp, mp in plan_match.items() if mp == kw_plan), None)
            if loc_plan is None:
                continue
            chunks = plan_chunks.setdefault(loc_plan, [])
            if len(chunks) < MAX_PER_PLAN:
                txt = e.get("text", "")
                if txt and txt not in [c["text"] for c in chunks]:
                    chunks.append({
                        "school": e.get("school", loc_plan.split("：", 1)[0]),
                        "plan":   loc_plan,   # 統一用 location_index 的 key
                        "text":   txt,
                        "hits":   1,
                    })

    # 輸出統計
    print()
    overseas_entries: list[dict] = []
    for loc_plan in sorted(overseas_plans):
        chunks = plan_chunks.get(loc_plan, [])
        if chunks:
            overseas_entries.extend(chunks)
            print(f"  OK {loc_plan}：{len(chunks)} chunk")
        else:
            print(f"  -- {loc_plan}：0 chunk（略過）")

    year_chunks["國外"] = overseas_entries
    kw_data[YEAR] = year_chunks
    KW_CHUNKS.write_text(json.dumps(kw_data, ensure_ascii=False), encoding="utf-8")
    print(f"\nkw_chunks.json 已更新，國外 entries：{len(overseas_entries)} 個")


if __name__ == "__main__":
    main()
