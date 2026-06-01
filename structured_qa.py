"""
結構化 QA 模組：
  1. 從 計劃總覽.txt 自動解析 SDG 索引（列舉型問題）
  2. 從 qa_custom.txt 載入人工撰寫的 Q&A 對（知識型問題）

用法：
    from structured_qa import init_qa, try_structured_answer
    init_qa(Path("114txt"))
    answer = try_structured_answer("SDG8 有哪些學校？")  # str 或 None
"""
import re
from pathlib import Path

# ── SDG 名稱對照 ──────────────────────────────────────────
_SDG_NAMES = {
    1: "消除貧窮",        2: "零飢餓",           3: "良好健康與福祉",
    4: "優質教育",        5: "性別平等",          6: "乾淨用水及衛生",
    7: "可負擔及乾淨能源", 8: "合宜工作與經濟成長", 9: "產業創新與基礎設施",
    10: "減少不平等",     11: "永續城市及社區",    12: "負責任的消費及生產",
    13: "氣候行動",       14: "水下生物",          15: "陸地生物",
    16: "和平正義與強大機構", 17: "全球夥伴關係",
}

# ── 內部儲存 ──────────────────────────────────────────────
_SDG_INDEX: dict[int, list[dict]] = {}   # {sdg_num: [{"uni": ..., "plan": ...}]}
_SUMMARY: dict = {}                       # 全局統計（總計畫數、總校數…）
_CUSTOM_QA: list[dict] = []              # [{"keywords": [...], "answer": str}]
_READY = False

# qa_custom.txt 預設與 structured_qa.py 同層目錄
_QA_CUSTOM_PATH = Path(__file__).parent / "qa_custom.txt"


def init_qa(txt_dir: Path) -> None:
    """啟動時呼叫一次，解析 計劃總覽.txt 並載入 qa_custom.txt。"""
    global _READY
    overview = _find_overview(txt_dir)
    if overview is None:
        print("[QA] 找不到計劃總覽.txt，結構化 QA 停用。")
        return
    text = _read_text(overview)
    if not text:
        print("[QA] 計劃總覽.txt 讀取失敗，結構化 QA 停用。")
        return
    _parse_sdg(text)

    _load_custom_qa(_QA_CUSTOM_PATH)

    _READY = True
    total_plans = sum(len(v) for v in _SDG_INDEX.values())
    print(f"[QA] SDG 索引：{len(_SDG_INDEX)} 個 SDG，{total_plans} 筆計畫。"
          f"  自訂 QA：{len(_CUSTOM_QA)} 組。")


def try_structured_answer(question: str) -> str | None:
    """
    若問題命中結構化 QA，回傳完整答案字串；否則回傳 None（交給 RAG 處理）。
    優先順序：SDG 索引 → 自訂 QA
    """
    if not _READY:
        return None

    q = question.strip()

    # ── 優先 1：SDG# + 列舉/數量意圖 ──
    sdg_m = re.search(r'SDG\s*(\d{1,2})', q, re.IGNORECASE)
    if sdg_m:
        num = int(sdg_m.group(1))
        if num in _SDG_INDEX:
            plans = _SDG_INDEX[num]
            name = _SDG_NAMES.get(num, "")

            if re.search(r'幾個|幾件|幾所|幾間|幾校|幾項|共有多少|共幾', q):
                return (
                    f"SDG{num}「{name}」共有 **{len(plans)}** 個計畫，"
                    f"由 {len({p['uni'] for p in plans})} 間學校參與。"
                )

            list_kw = ["學校", "大學", "計畫", "哪些", "列出", "清單", "有哪",
                       "對應", "名單", "全部", "所有", "列表"]
            if any(kw in q for kw in list_kw):
                return _format_sdg_list(num, name, plans)

    # ── 優先 2：所有 SDG 概覽 ──
    if re.search(r'(全部|所有|每個|各個).{0,6}SDG|SDG.{0,6}(全部|所有|概覽|總覽|清單|列表)', q, re.IGNORECASE):
        return _format_all_sdg_overview()

    # ── 優先 3：整體統計（學校數/計畫數）──
    if re.search(r'共.*幾.*計畫|計畫.*共.*幾|總共.*計畫|計畫.*總數|計畫總件數', q):
        total = _SUMMARY.get("total_plans", sum(len(v) for v in _SDG_INDEX.values()))
        unis = _SUMMARY.get("total_unis", 0)
        return f"114 年度 USR 計畫共收錄 **{total} 件**計畫，涵蓋 **{unis} 間**大專院校。"

    if re.search(r'幾間學校|幾所學校|幾間大學|幾所大學|學校.{0,4}幾間|大學.{0,4}幾間', q):
        unis = _SUMMARY.get("total_unis", 0)
        total = _SUMMARY.get("total_plans", sum(len(v) for v in _SDG_INDEX.values()))
        if unis:
            return f"114 年度共有 **{unis} 間**大專院校參與 USR 計畫（共 {total} 件計畫）。"

    # ── 優先 4：自訂 QA（qa_custom.txt）──
    custom = _match_custom_qa(q)
    if custom:
        return custom

    return None


# ── 自訂 QA 載入與比對 ────────────────────────────────────

def _load_custom_qa(path: Path) -> None:
    """解析 qa_custom.txt，每組 Q&A 支援多個同義問法（用 | 分隔）。"""
    if not path.exists():
        print(f"[QA] 找不到 {path.name}，自訂 QA 停用。")
        return

    text = _read_text(path)
    if not text:
        return

    current_keywords: list[str] = []
    current_answer_lines: list[str] = []
    in_answer = False

    def _flush():
        if current_keywords and current_answer_lines:
            answer = "\n".join(current_answer_lines).strip()
            if answer:
                _CUSTOM_QA.append({
                    "keywords": current_keywords[:],
                    "answer": answer,
                })

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()

        # 忽略註解行
        if line.lstrip().startswith("#"):
            continue

        if line.startswith("Q:"):
            # 開始新的一組 QA：先存前一組
            _flush()
            current_keywords = []
            current_answer_lines = []
            in_answer = False
            qs = line[2:].strip()
            # 支援 | 分隔多個同義問法
            for q in qs.split("|"):
                q = q.strip()
                if q:
                    current_keywords.append(q)

        elif line.startswith("A:"):
            in_answer = True
            content = line[2:].strip()
            if content:
                current_answer_lines.append(content)

        elif in_answer:
            # 空白行代表一組結束
            if line.strip() == "":
                _flush()
                current_keywords = []
                current_answer_lines = []
                in_answer = False
            else:
                current_answer_lines.append(line)

    _flush()  # 最後一組


def _match_custom_qa(question: str) -> str | None:
    """
    比對邏輯：
    - 問題中包含 Q 的任一同義句，視為命中
    - 同義句比對：問題包含該句的所有「實詞」（≥2字元，非助詞）
    """
    q_lower = question.lower()
    best_score = 0
    best_answer = None

    for entry in _CUSTOM_QA:
        for kw_phrase in entry["keywords"]:
            score = _phrase_score(q_lower, kw_phrase.lower())
            if score > best_score:
                best_score = score
                best_answer = entry["answer"]

    # 閾值：至少 0.7 的關鍵詞命中率才回答
    if best_score >= 0.7:
        return best_answer
    return None


# 比對時過濾掉的短助詞
_STOP = {"是", "的", "了", "嗎", "呢", "啊", "有", "在", "和", "或", "與", "及",
         "請問", "請", "問", "可以", "告訴我", "什麼", "怎麼", "如何", "哪些",
         "一下", "介紹", "說明"}


def _phrase_score(question: str, phrase: str) -> float:
    """計算 phrase 的關鍵詞在 question 中的命中率（0.0 ~ 1.0）。"""
    # 切出長度 >= 2 且不在停用詞的詞
    tokens = [t for t in re.findall(r'[一-鿿\w]{2,}', phrase) if t not in _STOP]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in question)
    return hits / len(tokens)


# ── SDG 索引解析 ──────────────────────────────────────────

def _find_overview(txt_dir: Path) -> Path | None:
    for p in txt_dir.glob("*.txt"):
        if "計劃總覽" in p.stem or "計画総覧" in p.stem:
            return p
    return None


def _read_text(path: Path) -> str | None:
    for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    try:
        return path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return None


def _parse_sdg(text: str) -> None:
    global _SUMMARY
    current_sdg = None

    for line in text.split("\n"):
        stripped = line.strip()

        m_header = re.match(r'^SDG(\d{1,2})\s', stripped)
        if m_header:
            current_sdg = int(m_header.group(1))
            if current_sdg not in _SDG_INDEX:
                _SDG_INDEX[current_sdg] = []
            continue

        if current_sdg is not None and stripped.startswith("- "):
            content = stripped[2:].strip()
            if " — " in content:
                uni, plan = content.split(" — ", 1)
                plan = plan.strip()
                if plan:
                    _SDG_INDEX[current_sdg].append({"uni": uni.strip(), "plan": plan})
            continue

        if "SDG 索引結束" in stripped:
            current_sdg = None

        m_total = re.search(r'共有\s*(\d+)\s*件', stripped)
        if m_total and "total_plans" not in _SUMMARY:
            _SUMMARY["total_plans"] = int(m_total.group(1))

        m_unis = re.search(r'(\d+)\s*間.*大專院校', stripped)
        if m_unis and "total_unis" not in _SUMMARY:
            _SUMMARY["total_unis"] = int(m_unis.group(1))


def _format_sdg_list(num: int, name: str, plans: list[dict]) -> str:
    uni_set = {p["uni"] for p in plans}
    lines = [f"**SDG{num}「{name}」** 共有 {len(plans)} 個計畫，涉及 {len(uni_set)} 間學校：\n"]
    for i, p in enumerate(plans, 1):
        lines.append(f"{i}. **{p['uni']}**　{p['plan']}")
    return "\n".join(lines)


def _format_all_sdg_overview() -> str:
    lines = ["**114 年度 USR 計畫 — 各 SDG 計畫數量概覽**\n"]
    for num in range(1, 18):
        name = _SDG_NAMES.get(num, "")
        count = len(_SDG_INDEX.get(num, []))
        uni_count = len({p["uni"] for p in _SDG_INDEX.get(num, [])})
        lines.append(f"- SDG{num:02d}「{name}」：{count} 個計畫 / {uni_count} 間學校")
    total = sum(len(v) for v in _SDG_INDEX.values())
    lines.append(f"\n合計（含重複）：{total} 筆（一所學校可對應多個 SDG）")
    return "\n".join(lines)
