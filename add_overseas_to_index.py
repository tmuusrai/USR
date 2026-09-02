# -*- coding: utf-8 -*-
"""
解析 基本資料表合集.txt，抽取「國外實踐場域」資料，
更新 location_index.json（加 overseas_countries）
更新 label_index.json（加 國外 鍵）
"""
import json
import re
from pathlib import Path

SRC_TXT   = Path("extra_docs/基本資料表合集.txt")
LOC_IDX   = Path("114_output/location_index.json")
LBL_IDX   = Path("114_output/label_index.json")
YEAR      = "114"

# ── 解析基本資料表合集.txt ─────────────────────────────────────────────────────

def _read(path: Path) -> str:
    for enc in ["utf-8-sig", "utf-8", "cp950"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def _parse_overseas(text: str) -> dict[str, dict]:
    """
    回傳 {plan_key: {"school": ..., "plan_name": ..., "countries": [...]}}
    plan_key = "學校：計畫名" 格式（與 location_index key 一致）
    """
    results: dict[str, dict] = {}
    section_re = re.compile(r'^===\s*.+?\s*===$', re.MULTILINE)
    sections = section_re.split(text)

    for sec in sections:
        # 學校名稱：取所有匹配中，值不是「學校名稱」本身的最後一筆
        school_candidates = re.findall(r'學校名稱[ 　\t]*[：:\s]?[ 　\t]*([^\n：:（(]{2,30})', sec)
        school_candidates = [s.strip().lstrip('：: ') for s in school_candidates
                             if s.strip() and '學校名稱' not in s and '計畫名稱' not in s]
        # 計畫名稱：取所有匹配，排除值等於學校名（有些格式錯誤）的，取最後一筆
        plan_candidates = re.findall(r'計畫名稱[ 　\t]*[：:\s]?[ 　\t]*([^\n]{4,})', sec)
        plan_candidates = [p.strip().lstrip('：: ') for p in plan_candidates
                           if p.strip() and '計畫名稱' not in p]

        if not school_candidates or not plan_candidates:
            continue

        school = school_candidates[-1].strip()
        # 過濾掉值等於學校名的計畫名稱（中山大學那種錯位格式）
        valid_plans = [p for p in plan_candidates if _normalize(p) != _normalize(school)]
        if not valid_plans:
            continue
        plan = valid_plans[-1].strip()
        # 清理多餘空格（有些計畫名稱有全形空格或斷行）
        plan = re.sub(r'\s+', '', plan)

        # 找「國外實踐場域」段落
        ov_m = re.search(r'國外實踐場域(.*?)(?=國內實踐場域|計畫主持人|計畫聯絡人|計畫經費|$)',
                         sec, re.DOTALL)
        if not ov_m:
            continue

        ov_text = ov_m.group(1)
        # 抽取國家
        countries = re.findall(r'國家\s*[：:]\s*([^\n,，。、（\s]{2,20})', ov_text)
        # 過濾空字串、純標點
        countries = [c.strip() for c in countries if c.strip() and len(c.strip()) >= 2]
        # 排除「無」「N/A」等
        countries = [c for c in countries if c not in {"無", "N/A", "NA", "n/a"}]

        if not countries:
            continue

        key = f"{school}：{plan}"
        results[key] = {
            "school": school,
            "plan_name": plan,
            "countries": list(dict.fromkeys(countries)),  # 去重保序
        }

    return results


# ── 模糊匹配 location_index key ────────────────────────────────────────────────

def _normalize(s: str) -> str:
    return re.sub(r'[\s　\-_\-–—·‧‐]+', '', s).lower()


def _match_key(parsed_key: str, loc_keys: list[str]) -> str | None:
    """嘗試在 location_index keys 中找到最佳匹配。"""
    n_parsed = _normalize(parsed_key)

    # 1. 精確
    for k in loc_keys:
        if _normalize(k) == n_parsed:
            return k

    # 2. 部分包含（parsed 包含 loc key，或反向）
    school_part, _, plan_part = parsed_key.partition("：")
    n_school = _normalize(school_part)
    n_plan   = _normalize(plan_part)

    best = None
    best_score = 0.0
    for k in loc_keys:
        ks, _, kp = k.partition("：")
        nks = _normalize(ks)
        nkp = _normalize(kp)
        if nks != n_school:
            continue
        # 子字串包含
        if n_plan in nkp or nkp in n_plan:
            score = len(min(n_plan, nkp, key=len))
            if score > best_score:
                best = k
                best_score = score
            continue
        # 字元重疊比率（處理 OCR 截斷等雜訊）
        common = sum(1 for c in set(n_plan) if c in nkp)
        ratio = 2 * common / (len(set(n_plan)) + len(set(nkp))) if (n_plan and nkp) else 0
        if ratio >= 0.75 and ratio > best_score:
            best = k
            best_score = ratio

    return best


# ── 主程式 ─────────────────────────────────────────────────────────────────────

def main():
    print("讀取基本資料表合集.txt ...")
    raw = _read(SRC_TXT)
    overseas_map = _parse_overseas(raw)
    print(f"解析到 {len(overseas_map)} 個有國外場域的計畫")

    loc_data = json.loads(LOC_IDX.read_text(encoding="utf-8"))
    lbl_data = json.loads(LBL_IDX.read_text(encoding="utf-8"))

    loc_plans = loc_data.get(YEAR, {}).get("plans", {})
    loc_keys  = list(loc_plans.keys())

    overseas_plan_names: list[str] = []  # 成功對應的 location_index key 清單
    matched = 0
    unmatched = []

    for parsed_key, info in overseas_map.items():
        matched_key = _match_key(parsed_key, loc_keys)
        if matched_key:
            loc_plans[matched_key]["overseas_countries"] = info["countries"]
            overseas_plan_names.append(matched_key)
            print(f"  ✓ {matched_key} → {info['countries']}")
            matched += 1
        else:
            unmatched.append(parsed_key)

    if unmatched:
        print(f"\n未匹配（{len(unmatched)} 件）：")
        for k in unmatched:
            print(f"  ✗ {k}")

    # 寫回 location_index
    LOC_IDX.write_text(json.dumps(loc_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nlocation_index.json 已更新，{matched} 件計畫加入 overseas_countries")

    # 更新 label_index（加 國外 鍵）
    if overseas_plan_names:
        existing = set(lbl_data.get("國外", []))
        merged = list(existing | set(overseas_plan_names))
        lbl_data["國外"] = merged
        LBL_IDX.write_text(json.dumps(lbl_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"label_index.json 已更新，國外 鍵共 {len(merged)} 件計畫")


if __name__ == "__main__":
    main()
