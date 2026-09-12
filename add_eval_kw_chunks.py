# -*- coding: utf-8 -*-
"""
add_eval_kw_chunks.py
從 114md 計畫文件抽取成效評估關鍵字（KPI、SROI、rubric 等）的文字段落，
patch 進現有 kw_chunks_test.json(.gz)。

不重建整個 kw_chunks，只新增評估方法這些 key。
"""
import json, re, sys, gzip
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

MD_DIR         = Path("114md")
KW_CHUNKS_JSON = Path("114_output/kw_chunks_test.json")
KW_CHUNKS_GZ   = Path("114_output/kw_chunks_test.json.gz")

CHUNK_WINDOW = 400   # 關鍵字前後取多少字
MAX_PER_PLAN = 6     # 每個 (keyword, plan) 最多幾個 chunk

# ── 嚴格比對詞（避免廣義詞造成假陽性）──────────────────────────────────────
EVAL_PATTERNS: dict[str, list[str]] = {
    "KPI":          ["KPI"],
    "OKR":          ["OKR", "目標與關鍵成果"],
    "邏輯模型":     ["邏輯模型", "logic model", "Logic Model", "邏輯架構模型"],
    "改變理論":     ["改變理論", "theory of change", "Theory of Change", "TOC理論", "變革理論"],
    "CIPP":         ["CIPP", "CIPP模式", "CIPP評估"],
    "SROI":         ["SROI", "社會投資報酬率", "社會投報率"],
    "IMM":          ["IMM", "影響力管理與測量", "影響力衡量"],
    "SDGs對應":     ["SDGs對應", "SDG對應", "SDGs mapping", "SDG mapping", "永續發展目標對應"],
    "前後測":       ["前後測", "前測後測"],
    "問卷調查":     ["問卷調查", "量表調查", "滿意度問卷", "調查問卷"],
    "焦點團體":     ["焦點團體", "焦點訪談", "焦點座談"],
    "深度訪談":     ["深度訪談", "深入訪談", "半結構訪談", "質性訪談"],
    "學習歷程":     ["學習歷程", "作品分析", "歷程檔案"],
    "rubric":       ["rubric", "Rubric", "評分規準", "評量規準", "評量準則"],
    "成本效益分析": ["成本效益分析", "cost-benefit", "CBA"],
    "校務研究":     ["校務研究", "Institutional Research"],  # IR 另外用 regex
    "OGSM":         ["OGSM"],
}

_IR_RE = re.compile(r'(?<![a-zA-Z])IR(?![a-zA-Z])')


def _plan_key(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r'\s*\(\d{3}USR-[^)]*\)', '', stem)
    stem = re.sub(r'\(\d+\)$', '', stem).strip()
    parts = stem.split('_', 1)
    return f"{parts[0].strip()}：{parts[1].strip()}" if len(parts) == 2 else stem.strip()


def _get_chunks(text: str, label: str, patterns: list[str]) -> list[str]:
    """回傳文件中含有評估關鍵字的段落列表（已去重、截短）。"""
    paragraphs = re.split(r'\n{2,}', text)
    results, seen = [], set()
    for para in paragraphs:
        para = para.strip()
        if len(para) < 30:
            continue
        hit = any(p.lower() in para.lower() for p in patterns)
        if label == "校務研究" and not hit:
            hit = bool(_IR_RE.search(para))
        if not hit:
            continue
        # 段落太長：截取關鍵字前後 CHUNK_WINDOW 字
        if len(para) > CHUNK_WINDOW:
            for p in patterns:
                idx = para.lower().find(p.lower())
                if idx >= 0:
                    s = max(0, idx - 150)
                    para = para[s: s + CHUNK_WINDOW]
                    break
        if para not in seen:
            seen.add(para)
            results.append(para)
        if len(results) >= MAX_PER_PLAN:
            break
    return results


# ── 載入現有 kw_chunks ─────────────────────────────────────────────────────
if KW_CHUNKS_JSON.exists():
    kw_data = json.loads(KW_CHUNKS_JSON.read_text(encoding='utf-8'))
    save_as_gz = False
elif KW_CHUNKS_GZ.exists():
    with gzip.open(KW_CHUNKS_GZ, 'rt', encoding='utf-8') as f:
        kw_data = json.load(f)
    save_as_gz = True
else:
    print("找不到 kw_chunks 檔案，從空白建立。")
    kw_data = {}
    save_as_gz = False

data_114 = kw_data.setdefault("114", {})

# ── 掃描 md 文件 ───────────────────────────────────────────────────────────
md_files = list(MD_DIR.glob("*.md"))
print(f"掃描 {len(md_files)} 份計畫文件...\n")

collected: dict[str, list[dict]] = defaultdict(list)

for md in md_files:
    try:
        text = md.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        continue
    pk = _plan_key(md)
    for label, patterns in EVAL_PATTERNS.items():
        for chunk in _get_chunks(text, label, patterns):
            collected[label].append({"plan": pk, "text": chunk})

# ── 合併進現有 kw_chunks ──────────────────────────────────────────────────
total_added = 0
for label, entries in sorted(collected.items()):
    existing = data_114.setdefault(label, [])
    # 現有的 plan → chunk 數量
    plan_count: dict[str, int] = defaultdict(int)
    for e in existing:
        plan_count[e.get("plan", "")] += 1

    added = 0
    seen_texts = {e.get("text", "") for e in existing}
    by_plan: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        by_plan[e["plan"]].append(e["text"])

    for plan, texts in by_plan.items():
        room = MAX_PER_PLAN - plan_count[plan]
        for txt in texts:
            if room <= 0:
                break
            if txt in seen_texts:
                continue
            existing.append({"plan": plan, "text": txt})
            seen_texts.add(txt)
            plan_count[plan] += 1
            room -= 1
            added += 1

    total_added += added
    print(f"  {label:12s}: {len(by_plan):3d} 件計畫，新增 {added} 個 chunk")

# ── 儲存 ──────────────────────────────────────────────────────────────────
if save_as_gz:
    with gzip.open(KW_CHUNKS_GZ, 'wt', encoding='utf-8') as f:
        json.dump(kw_data, f, ensure_ascii=False)
    print(f"\n✓ 寫入 {KW_CHUNKS_GZ}，共新增 {total_added} 個 chunk")
else:
    KW_CHUNKS_JSON.write_text(
        json.dumps(kw_data, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f"\n✓ 寫入 {KW_CHUNKS_JSON}，共新增 {total_added} 個 chunk")
