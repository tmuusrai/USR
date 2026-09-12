# -*- coding: utf-8 -*-
"""
add_domestic_fields.py
從國內實踐場域 Excel 匯入場域名稱到 location_index.json。
每個計畫只存場域名稱（location）與所屬大區（region）。
"""
import json, sys, openpyxl
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

EXCEL_PATH     = r"C:\Users\Kuo\Downloads\114年度USR計畫國內實踐場域_學校+機構_資料庫new.xlsx"
LOCATION_PATH  = Path("114_output/location_index.json")

wb = openpyxl.load_workbook(EXCEL_PATH, read_only=True, data_only=True)
ws = wb["國內實踐場域總資料庫"]
rows = list(ws.iter_rows(values_only=True))
headers = rows[0]
col = {h: i for i, h in enumerate(headers) if h}

# 每個計畫 → [(region, location), ...]（去重）
plan_fields: dict[str, list[dict]] = defaultdict(list)
plan_seen:   dict[str, set]        = defaultdict(set)

for r in rows[1:]:
    school = r[col["學校名稱"]]
    plan   = r[col["計畫名稱"]]
    field  = r[col["國內實踐場域（學校/機構）"]]
    if not (school and plan and field):
        continue
    key  = f"{school}：{plan}"
    loc  = str(field).strip()
    if loc in plan_seen[key]:
        continue
    plan_seen[key].add(loc)
    plan_fields[key].append({"location": loc})

print(f"讀取完成：{len(plan_fields)} 件計畫")

# 更新 location_index
loc_data = json.loads(LOCATION_PATH.read_text(encoding='utf-8'))
plans_114 = loc_data.setdefault("114", {}).setdefault("plans", {})

added_plans = 0
updated_plans = 0
added_fields = 0

for key, new_entries in plan_fields.items():
    if key not in plans_114:
        plans_114[key] = {}
        added_plans += 1
    else:
        updated_plans += 1

    existing = plans_114[key].setdefault("fields", [])
    existing_locs = {e["location"] for e in existing}

    for entry in new_entries:
        if entry["location"] not in existing_locs:
            existing.append(entry)
            existing_locs.add(entry["location"])
            added_fields += 1

print(f"新增計畫：{added_plans}  更新計畫：{updated_plans}  新增場域：{added_fields}")

LOCATION_PATH.write_text(
    json.dumps(loc_data, ensure_ascii=False, indent=2),
    encoding='utf-8'
)
print(f"✓ 已寫入 {LOCATION_PATH}")
