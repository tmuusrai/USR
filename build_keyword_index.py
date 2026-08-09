"""
建立完整 keyword_index.json：
  - SDG1~17 → 計畫名單（從 計劃總覽.txt 讀取）
  - USR_TOPIC_KEYWORDS 每個詞 → 計畫名單（掃 vectorstore chunk）

執行方式：
  python build_keyword_index.py [--year 114] [--year 113]
  預設只建 114 年。
"""
import sys
import re
import json
import time
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings
import os

# ── 路徑設定 ──────────────────────────────────────────
PLAN_FILE     = Path("qa_data/計劃總覽.txt")
KW_INDEX_FILE = Path("keyword_index.json")

INDEX_DIRS = {
    "114": Path(os.getenv("INDEX_DIR",     "faiss_index")),
    "113": Path(os.getenv("INDEX_DIR_113", "faiss_index_113")),
}

# ── USR_TOPIC_KEYWORDS（與 app.py 保持一致）──────────
USR_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "在地關懷": [
        "高齡照護", "偏鄉教育", "社區營造", "弱勢扶助", "在地關懷", "社區關懷",
        "社區培力", "青銀共創", "新住民關懷", "原住民族關懷", "偏鄉照顧", "弱勢關懷",
        "場域合作", "MOU", "在地認同", "在地需求", "韌性社區", "地方特色", "區域發展",
        "在地化經濟", "在地產業", "青年就業", "城市轉型", "共學共創", "部落共學", "世代共融",
    ],
    "環境永續": [
        "淨零碳排", "循環經濟", "環境教育", "生態保育", "永續環境", "環境永續",
        "節能減碳", "氣候變遷", "生物多樣性", "水資源", "廢棄物減量", "永續校園",
        "防災韌性", "再生能源", "綠色生活", "臺南濕地", "碳足跡", "碳中和",
        "環境治理", "水資源管理",
    ],
    "健康促進與食品安全": [
        "社區健康促進", "食品安全", "食農教育", "健康識能", "健康促進",
        "身心健康", "社區健康", "營養教育", "高齡健康", "慢性病預防", "心理健康",
        "樸門農法", "永續農業", "運動促進", "在地食材", "食品溯源", "農產品安全",
        "健康飲食", "預防保健", "社區關懷", "社區共餐", "疾病防治",
    ],
    "產業鏈結與經濟永續": [
        "地方創生", "產學合作", "青年返鄉", "產業升級", "產業鏈結", "在地產業",
        "數位轉型", "社會企業", "微型創業", "品牌行銷", "農業加值", "創新創業",
        "永續商業模式", "人才培育", "就業媒合", "區域經濟", "在地經濟", "技術轉移",
        "在地產品", "產業活化", "就業輔導", "產業轉型",
    ],
    "文化永續": [
        "文化保存", "文化傳承", "地方文史", "數位典藏", "文化永續", "文化資產",
        "傳統技藝", "無形文化資產", "原住民族文化", "地方語言", "社區記憶",
        "口述歷史", "文化創意", "文化觀光", "歷史建築活化", "跨世代傳承", "老街活化",
        "宮廟文化", "台灣在地特色", "串聯人文地景", "空間再生", "地方走讀",
        "文創設計", "部落共學",
    ],
    "其他社會實踐": [
        "社會創新", "教育平權", "數位平權", "公民參與", "多元文化", "性別平等",
        "防災教育", "社區安全", "法律扶助", "科技導入", "智慧社區", "新住民支持",
        "身心障礙者支持", "動物保護", "人權教育", "社會共融", "國際合作", "特殊議題",
        "政策倡議", "減少不平等",
    ],
    "計畫行政": [
        "計畫申請", "申請資格", "經費核銷", "績效指標", "成果報告", "配合款",
        "SIG", "SROI", "ESG", "SDGs", "助理經費", "協同主持人", "場域變更",
        "經費變更", "IRB", "計畫執行策略", "團隊架構", "聯繫窗口", "訪視",
        "實地訪視", "委員", "成效評估", "資料填報", "計畫書撰寫", "經費編列",
        "典範轉移", "幼老共學", "人事異動", "社會影響力", "社會影響力評估",
        "USR EXPO", "年報", "計畫推廣", "共培活動", "共培", "PBL", "頂石",
        "學程", "中長程校務", "USR學分學程", "USR亮點案例", "USR影響力評估",
        "USR經費補助", "USR服務學習課程", "USR特色教學", "USR產學永續聯盟",
        "跨國社會實踐計畫", "優良案例", "永續經營",
        "萌芽型", "深耕型", "計畫主持人", "管考機制", "期中報告", "USR推動中心",
        "專任助理", "經常門", "資本門", "自籌款", "經費流用", "量化指標", "質化指標",
        "利害關係人", "成果發表會", "場域盤點", "跨校合作", "諮詢委員", "內部管考",
        "滿意度調查",
    ],
    "轉型正義": [
        "轉型正義", "威權統治", "白色恐怖", "戒嚴時期", "二二八事件", "政治受難者",
        "加害者識別", "加害者處置", "司法不法", "行政不法", "名譽回復", "財產返還",
        "人身自由侵害賠償", "政治暴力創傷", "療癒照顧", "不義遺址", "威權象徵",
        "中正紀念堂轉型", "政治檔案", "檔案解密", "口述歷史", "國家人權記憶庫",
        "臺灣轉型正義資料庫", "原住民族轉型正義", "轉型正義教育", "人權教育",
        "促進轉型正義基金", "黨產", "附隨組織", "黨營機構",
        "歷史記憶", "社會和解", "真相調查", "人權走讀", "空間解嚴", "歷史正義",
        "世代對話", "政治受難者遺族", "人權地景", "司法平反", "黨外運動", "威權遺緒",
        "人權策展", "美麗島事件", "在地記憶", "創傷知情",
    ],
    "社會福利": [
        "社會救助", "自立脫貧", "災害救助", "遊民輔導", "社區發展", "福利社區化",
        "社區培力", "在地共生社區", "志願服務", "高齡志工", "長照", "家庭暴力防治",
        "性侵害防治", "性騷擾防制", "兒童少年保護", "兒童少年性剝削防制",
        "目睹家庭暴力", "身心障礙者保護", "老人保護", "原鄉部落服務", "離島",
        "偏遠地區社工", "社會工作", "社工督導", "心理輔導", "創傷療癒",
        "低收入戶", "中低收入戶",
        "新住民服務", "獨居老人", "弱勢家庭", "社會安全網", "青銀共創", "活躍老化",
        "早期療育", "喘息服務", "食物銀行", "街友關懷", "隔代教養", "社會包容",
        "無障礙環境", "照顧者支持", "邊緣戶", "社會創新", "婦女培力", "弱勢關懷",
        "友善社區", "社會支持系統",
    ],
    "健康醫療": [
        "智慧醫療", "智慧照顧", "長照3.0", "醫療人才培育", "醫護工作條件",
        "分級醫療", "垂直整合", "區域聯防", "永續醫療", "社會責任醫療",
        "精準醫療", "臨床AI", "健康台灣", "醫療資源", "偏鄉醫療", "醫院社會責任",
        "高齡友善", "在地老化", "遠距醫療", "健康促進", "健康識能", "預防延緩失能",
        "社區照護", "失智照護", "醫療平權", "原鄉醫療", "居家醫療", "活躍老化",
        "數位健康", "心理健康", "跨領域照護", "衛生教育", "慢性病管理", "弱勢照護",
    ],
    "科技研究": [
        "異種器官移植", "再生醫療", "智慧機器人", "智慧製造", "半導體",
        "精準醫學", "綠能", "再生能源", "太空任務", "衛星科學", "癌症轉譯研究",
        "原住民族研究", "族群研究", "氣候韌性", "生物多樣性", "青年學者",
        "科技研發", "國際學術合作",
        "人工智慧", "物聯網", "智慧農業", "智慧醫療", "淨零排放", "循環經濟",
        "大數據分析", "防災科技", "無人機應用", "產學合作", "跨領域研究", "減碳技術",
        "輔具科技", "海洋科學", "技術移轉", "永續科技", "智慧城市", "區塊鏈技術",
    ],
}


def _clean(text: str) -> str:
    return re.sub(r'[_\-【】\[\]「」『』（）()]', ' ', text)


def _source_to_plan(source: str) -> str | None:
    stem = Path(source).stem
    parts = stem.split('_', 1)
    if len(parts) == 2:
        return f"{parts[0]}：{parts[1]}"
    return None


def build_sdg_plans(year: str) -> dict[str, list[str]]:
    """從 計劃總覽.txt 建立 SDG → 計畫名單。"""
    sdg_plans: dict[str, list[str]] = {f"SDG{i}": [] for i in range(1, 18)}
    if not PLAN_FILE.exists():
        print(f"[WARN] 找不到 {PLAN_FILE}，跳過 SDG 建索引")
        return sdg_plans

    text = PLAN_FILE.read_text(encoding="utf-8")
    blocks = re.split(r'===.+?===', text)
    for block in blocks:
        school_m = re.search(r'學校名稱[　\s]+(.+)', block)
        plan_m   = re.search(r'計畫名稱[　\s]+(.+)', block)
        sdg_m    = re.search(r'SDGs關聯議題[　\t ]+(.+)', block)
        if not school_m or not sdg_m:
            continue
        school = school_m.group(1).strip().rstrip('-').strip()
        plan   = plan_m.group(1).strip().rstrip('-').strip() if plan_m else ''
        entry  = f"{school}：{plan}" if plan else school
        sdg_str = sdg_m.group(1).strip()
        for num_str in re.findall(r'\b(1[0-7]|[1-9])\b', sdg_str):
            key = f"SDG{int(num_str)}"
            if entry not in sdg_plans[key]:
                sdg_plans[key].append(entry)

    for key, plans in sdg_plans.items():
        print(f"  {key}: {len(plans)} 件")
    return sdg_plans


def build_topic_plans(vs) -> dict[str, list[str]]:
    """掃 vectorstore，建立 USR_TOPIC_KEYWORDS 詞 → 計畫名單。"""
    all_kws: set[str] = set()
    for kws in USR_TOPIC_KEYWORDS.values():
        all_kws.update(kws)

    # 建 {kw: set(plan_name)}
    kw_plans: dict[str, set[str]] = {kw: set() for kw in all_kws}

    docs = list(vs.docstore._dict.values())
    t0 = time.perf_counter()
    for doc in docs:
        text = _clean(doc.page_content)
        plan = _source_to_plan(doc.metadata.get("source", ""))
        if not plan:
            continue
        for kw in all_kws:
            if kw in text:
                kw_plans[kw].add(plan)

    elapsed = round((time.perf_counter() - t0) * 1000)
    total = sum(len(v) for v in kw_plans.values())
    print(f"  掃描完成：{len(all_kws)} 個關鍵字，{total} 筆對應，耗時 {elapsed}ms")

    return {kw: sorted(plans) for kw, plans in kw_plans.items() if plans}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", action="append", default=None,
                        help="要建索引的年份，可多次指定（預設：114）")
    args = parser.parse_args()
    years = args.year or ["114"]

    kw_data: dict = {}
    if KW_INDEX_FILE.exists():
        kw_data = json.loads(KW_INDEX_FILE.read_text(encoding="utf-8"))

    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=VOYAGE_API_KEY,
        model="voyage-3-large",
        batch_size=64,
    )

    for year in years:
        print(f"\n=== 建立 {year} 年索引 ===")
        idx_dir = INDEX_DIRS.get(year)
        if not idx_dir or not (idx_dir / "index.faiss").exists():
            print(f"  找不到 {year} 年 vectorstore，跳過")
            continue

        print("  載入 vectorstore...")
        vs = FAISS.load_local(
            str(idx_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        yr_data: dict = kw_data.setdefault(year, {})

        print("  建立 SDG 計畫名單...")
        sdg_plans = build_sdg_plans(year)
        for key, plans in sdg_plans.items():
            if plans:
                yr_data[key] = sorted(plans)

        print("  建立 USR_TOPIC_KEYWORDS 計畫名單...")
        topic_plans = build_topic_plans(vs)
        for kw, plans in topic_plans.items():
            yr_data[kw] = plans

        total_kws = len(yr_data)
        total_entries = sum(len(v) for v in yr_data.values())
        print(f"  {year} 年完成：{total_kws} 個關鍵字，共 {total_entries} 筆")

    KW_INDEX_FILE.write_text(
        json.dumps(kw_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nkeyword_index.json 已更新。")


if __name__ == "__main__":
    main()
