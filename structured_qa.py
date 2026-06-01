"""
結構化 QA 模組：從 計劃總覽.txt 解析固定答案，繞過 RAG 直接回應列舉型問題。
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
_READY = False


def init_qa(txt_dir: Path) -> None:
    """啟動時呼叫一次，解析 計劃總覽.txt 建立索引。"""
    global _READY
    overview = _find_overview(txt_dir)
    if overview is None:
        print("[QA] 找不到計劃總覽.txt，結構化 QA 停用。")
        return
    text = _read_text(overview)
    if not text:
        print("[QA] 計劃總覽.txt 讀取失敗，結構化 QA 停用。")
        return
    _parse(text)
    _READY = True
    total_plans = sum(len(v) for v in _SDG_INDEX.values())
    print(f"[QA] 結構化索引就緒：{len(_SDG_INDEX)} 個 SDG，共 {total_plans} 筆計畫紀錄。")


def try_structured_answer(question: str) -> str | None:
    """
    若問題命中結構化 QA，回傳完整答案字串；否則回傳 None（交給 RAG 處理）。
    """
    if not _READY:
        return None

    q = question.strip()

    # ── 模式 1：SDG# + 列舉意圖 ──
    sdg_m = re.search(r'SDG\s*(\d{1,2})', q, re.IGNORECASE)
    if sdg_m:
        num = int(sdg_m.group(1))
        if num in _SDG_INDEX:
            plans = _SDG_INDEX[num]
            name = _SDG_NAMES.get(num, "")

            # 只問數量
            if re.search(r'幾個|幾件|幾所|幾間|幾校|幾項|共有多少|共幾', q):
                return (
                    f"SDG{num}「{name}」共有 **{len(plans)}** 個計畫，"
                    f"由 {len({p['uni'] for p in plans})} 間學校參與。"
                )

            # 列出計畫或學校
            list_kw = ["學校", "大學", "計畫", "哪些", "列出", "清單", "有哪",
                       "對應", "名單", "全部", "所有", "列表"]
            if any(kw in q for kw in list_kw):
                return _format_sdg_list(num, name, plans)

    # ── 模式 2：問所有 SDG 概覽 ──
    if re.search(r'(全部|所有|每個|各個).{0,6}SDG|SDG.{0,6}(全部|所有|概覽|總覽|清單|列表)', q, re.IGNORECASE):
        return _format_all_sdg_overview()

    # ── 模式 3：問整體統計 ──
    if re.search(r'共.*幾.*計畫|計畫.*共.*幾|總共.*計畫|計畫.*總數|計畫總件數', q):
        total = _SUMMARY.get("total_plans", sum(len(v) for v in _SDG_INDEX.values()))
        unis = _SUMMARY.get("total_unis", 0)
        return f"114 年度 USR 計畫共收錄 **{total} 件**計畫，涵蓋 **{unis} 間**大專院校。"

    if re.search(r'幾間學校|幾所學校|幾間大學|幾所大學|學校.{0,4}幾間|大學.{0,4}幾間', q):
        unis = _SUMMARY.get("total_unis", 0)
        total = _SUMMARY.get("total_plans", sum(len(v) for v in _SDG_INDEX.values()))
        if unis:
            return f"114 年度共有 **{unis} 間**大專院校參與 USR 計畫（共 {total} 件計畫）。"

    # ── 模式 4：問特定學校參與哪些 SDG ──
    if re.search(r'SDG|哪些.*SDG|對應.*SDG', q, re.IGNORECASE) and not sdg_m:
        return None  # 交 RAG 處理

    return None


# ── 內部解析 ──────────────────────────────────────────────

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


def _parse(text: str) -> None:
    global _SUMMARY
    current_sdg = None

    for line in text.split("\n"):
        stripped = line.strip()

        # SDG 標題行："SDG8 合宜工作與經濟成長（共 46 個計畫）："
        m_header = re.match(r'^SDG(\d{1,2})\s', stripped)
        if m_header:
            current_sdg = int(m_header.group(1))
            if current_sdg not in _SDG_INDEX:
                _SDG_INDEX[current_sdg] = []
            continue

        # 計畫行："  - 學校 — 計畫名"
        if current_sdg is not None and stripped.startswith("- "):
            content = stripped[2:].strip()
            if " — " in content:
                uni, plan = content.split(" — ", 1)
                plan = plan.strip()
                if plan:
                    _SDG_INDEX[current_sdg].append({"uni": uni.strip(), "plan": plan})
            continue

        # 結束標記
        if "SDG 索引結束" in stripped:
            current_sdg = None

        # 統計數字
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
