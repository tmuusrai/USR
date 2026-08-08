import re
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

PLAN_FILE = Path('qa_data/計劃總覽.txt')
KW_INDEX_FILE = Path('keyword_index.json')

SDG_NAMES = {
    1: "SDG1", 2: "SDG2", 3: "SDG3", 4: "SDG4",
    5: "SDG5", 6: "SDG6", 7: "SDG7", 8: "SDG8",
    9: "SDG9", 10: "SDG10", 11: "SDG11", 12: "SDG12",
    13: "SDG13", 14: "SDG14", 15: "SDG15", 16: "SDG16", 17: "SDG17",
}

text = PLAN_FILE.read_text(encoding='utf-8')

# 切成各計畫 block
blocks = re.split(r'===.+?===', text)

# SDG → 學校：計畫名 的清單
sdg_plans: dict[str, list[str]] = {f"SDG{i}": [] for i in range(1, 18)}

for block in blocks:
    school_m = re.search(r'學校名稱[　\s]+(.+)', block)
    plan_m   = re.search(r'計畫名稱[　\s]+(.+)', block)
    sdg_m    = re.search(r'SDGs關聯議題[　\t ]+(.+)', block)
    if not school_m or not sdg_m:
        continue

    school  = school_m.group(1).strip().rstrip('-').strip()
    plan    = plan_m.group(1).strip().rstrip('-').strip() if plan_m else ''
    sdg_str = sdg_m.group(1).strip()
    entry   = f"{school}：{plan}" if plan else school

    for num_str in re.findall(r'\b(1[0-7]|[1-9])\b', sdg_str):
        num = int(num_str)
        if 1 <= num <= 17:
            key = f"SDG{num}"
            if entry not in sdg_plans[key]:
                sdg_plans[key].append(entry)

# 印統計
for key, plans in sdg_plans.items():
    print(f"{key}: {len(plans)} 件")

# 讀現有 keyword_index.json
if KW_INDEX_FILE.exists():
    kw_data = json.loads(KW_INDEX_FILE.read_text(encoding='utf-8'))
else:
    kw_data = {}

kw_114 = kw_data.setdefault('114', {})

# 寫入 SDG 資料（學校：計畫名 格式）
for key, plans in sdg_plans.items():
    if plans:
        kw_114[key] = sorted(plans)

KW_INDEX_FILE.write_text(
    json.dumps(kw_data, ensure_ascii=False, indent=2),
    encoding='utf-8'
)
total = sum(len(v) for v in sdg_plans.values())
print(f"\nkeyword_index.json 已更新，共寫入 {total} 筆 SDG-計畫對應。")
