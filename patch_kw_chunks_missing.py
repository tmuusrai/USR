# -*- coding: utf-8 -*-
"""
修補 kw_chunks.json 中缺失的計畫（耕莘健康管理+馬偕醫護）。
直接從 md 文件切 chunk，按關鍵字比對後寫入。
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

KW_CHUNKS_PATH = Path("114_output/kw_chunks.json")

# 缺失計畫的 md 路徑（精確對應）
MISSING_PLANS: list[tuple[str, str, str]] = [
    ("耕莘健康管理專科學校",
     "耕莘健康管理專科學校：循環共伴、永續莘未來~ 循環經濟 X 綠色照顧",
     "114md/耕莘健康管理專科學校_循環共伴、永續莘未來~ 循環經濟 X 綠色照顧(114USR-RMF-CTCN-UB1).md"),
    ("耕莘健康管理專科學校",
     "耕莘健康管理專科學校：惜食共好莘願景",
     "114md/耕莘健康管理專科學校_惜食共好莘願景(114USR-RMF-CTCN-UA1).md"),
    ("耕莘健康管理專科學校",
     "耕莘健康管理專科學校：為愛啟程、串起微笑~跨國溫莘社會處方照顧計畫",
     "114md/耕莘健康管理專科學校_為愛啟程、串起微笑~跨國溫莘社會處方照顧計畫(114USR-RMF-CTCN-UA1).md"),
    ("耕莘健康管理專科學校",
     "耕莘健康管理專科學校：身心復原力 x 建構療癒村：照顧者身心適能與心理韌性促進計畫",
     "114md/耕莘健康管理專科學校_身心復原力 x 建構療癒村：照顧者身心適能與心理韌性促進計畫(114USR-RMF-CTCN-UA1).md"),
    ("馬偕醫護管理專科學校",
     "馬偕醫護管理專科學校：「GO ! 馬偕特派員2.0」- 跨領域共學共創，牽手微伴青銀樂活與創生計畫",
     "114md/馬偕醫護管理專科學校_「GO ! 馬偕特派員2.0」- 跨領域共學共創，牽手微伴青銀樂活與創生計畫(114USR-RMF-MKC-UA1).md"),
    ("馬偕醫護管理專科學校",
     "馬偕醫護管理專科學校：醫護有愛，偏鄉營照，數位共照，健康永續",
     "114md/馬偕醫護管理專科學校_醫護有愛，偏鄉營照，數位共照，健康永續(114USR-RMF-MKC-UA1).md"),
]

# 關鍵字 → 相關詞（用於比對 chunk 文字）
KEYWORD_RELATED: dict[str, list[str]] = {
    "循環經濟": ["循環", "回收", "廢棄物", "惜食", "剩食", "再利用", "資源循環", "永續"],
    "高齡照護": ["高齡", "長者", "老人", "銀髮", "失智", "照顧", "長照", "老化", "日照"],
    "社區健康促進": ["健康促進", "健康識能", "慢性病", "運動", "衛教", "健康", "促進"],
    "健康促進": ["健康促進", "健康", "衛教", "促進", "健康識能", "慢性病", "運動"],
    "食農教育": ["食農", "農業", "食材", "飲食", "農產", "農耕", "有機", "營養", "食品"],
    "食品安全": ["食品安全", "食安", "食品", "衛生", "農藥", "添加物", "檢驗"],
    "偏鄉教育": ["偏鄉", "偏遠", "原鄉", "資源不足", "城鄉落差", "弱勢", "教育"],
    "偏鄉照顧": ["偏鄉", "偏遠", "醫療資源", "巡迴", "在宅", "照顧", "服務"],
    "社區營造": ["社區", "社區發展", "社區活化", "里民", "鄰里", "社區組織"],
    "弱勢扶助": ["弱勢", "低收入", "貧困", "助人", "社會救助", "關懷"],
    "跨域合作": ["跨域", "跨領域", "跨校", "跨科", "合作", "協作", "整合"],
    "服務學習": ["服務學習", "志工", "社會實踐", "課程", "學生", "服務"],
    "在地關懷": ["在地", "社區", "關懷", "服務", "陪伴", "志工"],
    "新住民關懷": ["新住民", "移民", "移工", "外籍", "東南亞", "外配"],
    "原住民族關懷": ["原住民", "部落", "原民", "族語"],
    "銀髮族": ["銀髮", "高齡", "長者", "老人", "年長", "樂齡"],
    "青銀共創": ["青銀", "世代", "共創", "年輕", "高齡", "互動", "共學"],
    "社會處方": ["社會處方", "非藥物", "處方", "社區介入", "健康"],
    "心理健康": ["心理", "身心", "韌性", "復原力", "療癒", "情緒"],
    "環境永續": ["環境", "永續", "生態", "綠色", "永續發展"],
    "在地產業": ["在地", "產業", "地方", "農業", "農產", "特色"],
    "地方創生": ["地方創生", "創生", "返鄉", "地方", "活化"],
    "共學共創": ["共學", "共創", "合作", "協作", "學習", "創新"],
    "長照": ["長照", "長期照護", "照顧", "高齡", "老人", "居家"],
    "社區醫療": ["社區醫療", "醫療", "社區", "衛生", "診所", "醫護"],
    "護理": ["護理", "醫護", "照護", "護士", "護理師"],
    "數位": ["數位", "科技", "AI", "資訊", "智慧", "數位化"],
    "世代共融": ["世代", "共融", "青銀", "跨世代", "老少", "代間"],
}


def _split_chunks(text: str, max_len: int = 350, overlap: int = 50) -> list[str]:
    """按段落切 chunk，超過 max_len 則強制切割。"""
    paragraphs = re.split(r'\n{2,}', text)
    chunks: list[str] = []
    cur = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(cur) + len(para) + 1 <= max_len:
            cur = (cur + "\n" + para).strip()
        else:
            if cur:
                chunks.append(cur)
            # 長段落強制切割
            while len(para) > max_len:
                chunks.append(para[:max_len])
                para = para[max_len - overlap:]
            cur = para
    if cur:
        chunks.append(cur)
    return chunks


def _chunk_matches(chunk: str, related: list[str]) -> bool:
    return any(w in chunk for w in related)


# ── 主流程 ──────────────────────────────────────────────────────────────────
data = json.loads(KW_CHUNKS_PATH.read_text(encoding="utf-8"))
chunks_114 = data.setdefault("114", {})

added_total = 0
MAX_PER_PLAN = 3   # 每個 (keyword, plan) 最多 3 個 chunk

for school, plan_key, md_path in MISSING_PLANS:
    path = Path(md_path)
    if not path.exists():
        print(f"[MISS] {path.name}")
        continue

    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = _split_chunks(text)
    print(f"\n[{school.split('管理')[0]}...] {plan_key[len(school)+1:25]}... → {len(chunks)} chunks")

    plan_added = 0
    for kw, related in KEYWORD_RELATED.items():
        matching = [c for c in chunks if _chunk_matches(c, related)]
        if not matching:
            continue
        existing = chunks_114.setdefault(kw, [])
        # 已有此 plan 的 chunk 數
        already = sum(1 for e in existing if isinstance(e, dict) and e.get("plan") == plan_key)
        to_add = MAX_PER_PLAN - already
        for chunk in matching[:to_add]:
            existing.append({"school": school, "plan": plan_key, "text": chunk})
            plan_added += 1
            added_total += 1
        if to_add > 0 and matching:
            print(f"  {kw}: +{min(len(matching), to_add)} chunks")

KW_CHUNKS_PATH.write_text(
    json.dumps(data, ensure_ascii=False, indent=1),
    encoding="utf-8"
)
print(f"\n完成！共新增 {added_total} 個 chunk entry → kw_chunks.json 已存檔")
