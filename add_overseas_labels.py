# -*- coding: utf-8 -*-
"""
將 Excel 海外實踐場域資料寫入：
  1. location_index.json  → 每個計畫的 overseas_fields（詳細場域+時間）
  2. label_index.json     → 海外場域、各國家、各區域 label
"""
import json, sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

try:
    import openpyxl
except ImportError:
    print("請先安裝 openpyxl：pip install openpyxl")
    sys.exit(1)

EXCEL_PATH       = r"C:\Users\Kuo\Desktop\USR\資料\114年度USR計畫海外實踐場域_學校+機構_總資料庫.xlsx"
LOCATION_PATH    = Path("114_output/location_index.json")
LABEL_INDEX_PATH = Path("114_output/label_index.json")

# ── 讀 Excel ──────────────────────────────────────────────────────────────────
wb = openpyxl.load_workbook(EXCEL_PATH, read_only=True, data_only=True)
ws = wb["海外實踐場域總資料庫"]
rows = list(ws.iter_rows(values_only=True))
headers = rows[0]
col = {h: i for i, h in enumerate(headers) if h}

# 彙整：plan_key → 該計畫所有海外場域列表
plan_fields   = defaultdict(list)   # key → [field_dict, ...]
plan_countries = defaultdict(set)
plan_regions   = defaultdict(set)

country_to_plans = defaultdict(set)
region_to_plans  = defaultdict(set)
all_plans        = set()

def _fmt_date(val):
    """把 datetime / str 統一轉成 YYYY-MM-DD 字串，None 回傳 None。"""
    if val is None:
        return None
    import datetime
    if isinstance(val, (datetime.date, datetime.datetime)):
        return val.strftime("%Y-%m-%d")
    s = str(val).strip()
    # 處理 2025/02/01 格式
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 3:
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
    return s if s else None

for r in rows[1:]:
    school      = r[col["學校"]]
    plan        = r[col["計畫名稱"]]
    country     = r[col["國家/地區"]]
    region_raw  = r[col["海外區域"]]
    city        = r[col["城市/行政區"]]
    location    = r[col["海外實踐場域（學校/機構）"]]
    site_type   = r[col["場域類型"]]
    partner     = r[col["合作單位性質"]]
    period      = r[col["資料出現期間"]]
    date_h1     = _fmt_date(r[col["上半年日期"]])
    date_h2     = _fmt_date(r[col["下半年日期"]])
    start_date  = _fmt_date(r[col["合作起始日(最早)"]])

    if not school or not plan:
        continue

    key = f"{school}：{plan}"
    all_plans.add(key)

    # 國家（可能有 ；分隔）
    countries = [c.strip() for c in str(country).split("；") if c.strip()] if country else []
    # 區域取第一段（去括號、去斜線後半）
    region = str(region_raw).split("（")[0].split("／")[0].strip() if region_raw else ""

    for c in countries:
        plan_countries[key].add(c)
        country_to_plans[c].add(key)
    if region:
        plan_regions[key].add(region)
        region_to_plans[region].add(key)

    # 收集詳細場域（去重：同計畫同場域名稱只留一次）
    existing_locs = {f["location"] for f in plan_fields[key]}
    if location and location not in existing_locs:
        dates = [d for d in [date_h1, date_h2] if d]
        plan_fields[key].append({
            "country":      str(country).strip() if country else "",
            "region":       region,
            "city":         str(city).strip() if city else "",
            "location":     str(location).strip(),
            "site_type":    str(site_type).strip() if site_type else "",
            "partner_type": str(partner).strip() if partner else "",
            "period":       str(period).strip() if period else "",
            "start_date":   start_date,
            "dates":        dates,
        })

print(f"讀取完成：{len(all_plans)} 件計畫，{len(country_to_plans)} 國，{len(region_to_plans)} 區域")
total_fields = sum(len(v) for v in plan_fields.values())
print(f"海外場域總筆數（去重）：{total_fields}")

# ── 更新 location_index.json ──────────────────────────────────────────────────
loc_data = json.loads(LOCATION_PATH.read_text(encoding="utf-8"))
loc_plans = loc_data.setdefault("114", {}).setdefault("plans", {})

updated = 0
added   = 0
for key, fields in plan_fields.items():
    if key not in loc_plans:
        loc_plans[key] = {}
        added += 1
    else:
        updated += 1
    loc_plans[key]["overseas_countries"] = sorted(plan_countries[key])
    loc_plans[key]["overseas_fields"]    = fields

LOCATION_PATH.write_text(
    json.dumps(loc_data, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"location_index.json：更新 {updated} 件、新增 {added} 件 → 已存檔")

# ── 更新 label_index.json ─────────────────────────────────────────────────────
label_data  = json.loads(LABEL_INDEX_PATH.read_text(encoding="utf-8"))
labels_114  = label_data.setdefault("114", {})

def _sl(s): return sorted(s)

for country, plans in country_to_plans.items():
    labels_114[country] = _sl(plans)
for region, plans in region_to_plans.items():
    labels_114[region] = _sl(plans)

label_data["國外"] = _sl(all_plans)

LABEL_INDEX_PATH.write_text(
    json.dumps(label_data, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"label_index.json：海外場域/國家/區域 label 寫入 → 已存檔")
print("\n完成！")
