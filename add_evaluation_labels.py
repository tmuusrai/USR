# -*- coding: utf-8 -*-
"""
掃描 114md 計畫文件，偵測各種成效評估方法，寫入 label_index.json。
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

MD_DIR          = Path("114md")
LABEL_INDEX_PATH = Path("114_output/label_index.json")

# ── 評估方法關鍵字定義 ──────────────────────────────────────────────────────────
EVAL_KEYWORDS: dict[str, list[str]] = {
    "KPI":       ["KPI", "關鍵績效指標", "績效指標"],
    "OKR":       ["OKR", "目標與關鍵成果"],
    "邏輯模型":   ["邏輯模型", "logic model", "Logic Model", "邏輯架構模型"],
    "改變理論":   ["改變理論", "theory of change", "Theory of Change", "TOC理論", "變革理論"],
    "CIPP":      ["CIPP", "CIPP模式", "CIPP評估"],
    "SROI":      ["SROI", "社會投資報酬率", "社會投報率"],
    "IMM":       ["IMM", "影響力管理與測量", "影響力衡量"],
    "SDGs對應":  ["SDGs對應", "SDG對應", "SDGs mapping", "SDG mapping", "永續發展目標對應"],
    "前後測":    ["前後測", "前測後測", "前測", "後測"],
    "問卷調查":  ["問卷調查", "問卷", "量表調查", "滿意度問卷", "調查問卷"],
    "焦點團體":  ["焦點團體", "焦點訪談", "焦點座談"],
    "深度訪談":  ["深度訪談", "深入訪談", "半結構訪談", "質性訪談"],
    "學習歷程":  ["學習歷程", "作品分析", "學習作品", "歷程檔案", "portfolio"],
    "rubric":    ["rubric", "Rubric", "評分規準", "評量規準", "評量準則"],
    "成本效益分析": ["成本效益分析", "CBA", "成本效益", "cost-benefit"],
    "校務研究":  ["校務研究", "IR", "Institutional Research", "機構研究"],
    "OGSM":     ["OGSM"],
}

def _extract_plan_key(path: Path) -> str:
    """從檔名取得『學校：計畫名』格式的 key。"""
    stem = path.stem
    stem = re.sub(r'\s*\(\d{3}USR-[^)]*\)', '', stem)   # 去掉 (114USR-xxx)
    stem = re.sub(r'\(\d+\)$', '', stem).strip()          # 去掉結尾 (1)(2)
    parts = stem.split('_', 1)
    if len(parts) == 2:
        return f"{parts[0].strip()}：{parts[1].strip()}"
    return stem.strip()

# ── 掃描 ────────────────────────────────────────────────────────────────────────
label_to_plans: dict[str, set[str]] = defaultdict(set)

md_files = list(MD_DIR.glob("*.md"))
print(f"掃描 {len(md_files)} 份計畫文件...")

for md in md_files:
    try:
        text = md.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    plan_key = _extract_plan_key(md)
    text_lower = text.lower()

    for label, kws in EVAL_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in text_lower:
                label_to_plans[label].add(plan_key)
                break   # 同 label 命中一個詞就算

# ── 輸出結果 ─────────────────────────────────────────────────────────────────────
print("\n=== 掃描結果 ===")
for label, plans in sorted(label_to_plans.items()):
    print(f"{label}：{len(plans)} 件")
    for p in sorted(plans)[:3]:
        print(f"  - {p[:60]}")
    if len(plans) > 3:
        print(f"  ... 共 {len(plans)} 件")

# ── 寫入 label_index.json ────────────────────────────────────────────────────────
label_data = json.loads(LABEL_INDEX_PATH.read_text(encoding="utf-8"))
labels_114 = label_data.setdefault("114", {})

updated = 0
for label, plans in label_to_plans.items():
    labels_114[label] = sorted(plans)
    updated += 1

LABEL_INDEX_PATH.write_text(
    json.dumps(label_data, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"\n✓ 寫入 {updated} 個評估標籤至 label_index.json")
