"""
結構化 QA 模組：
  1. 從 qa_custom.txt 載入人工撰寫的 Q&A 對（含 SDG 索引與知識型問題）
  2. 從 計劃總覽.txt 載入各計畫基本資料（學校名稱、主持人、實踐場域等）

用法：
    from structured_qa import init_qa, try_structured_answer
    init_qa()
    answer = try_structured_answer("SDG8 有哪些學校？")  # str 或 None
"""
import re
import unicodedata
from pathlib import Path


def _half(s: str) -> str:
    """全形英數 → 半形（NFKC normalization）"""
    return unicodedata.normalize('NFKC', s)

# ── 內部儲存 ──────────────────────────────────────────────
_CUSTOM_QA_BY_YEAR: dict[str, list[dict]] = {"114": [], "113": []}

# plan basics: year → school_name → list of {plan_name, text}
_PLAN_BASICS_BY_YEAR: dict[str, dict[str, list[dict]]] = {"114": {}, "113": {}}
_PLAN_BASIC_SCHOOLS_BY_YEAR: dict[str, list[str]] = {"114": [], "113": []}

_READY = False

# qa_data/ 資料夾預設與 structured_qa.py 同層
_QA_DIR = Path(__file__).parent / "qa_data"

# 基本資料型問題觸發詞
_PLAN_INFO_RE = re.compile(
    r'基本資料|主持人|實踐場域|計畫類別|計畫議題|SDGs關聯|聯絡人|計畫類型|核定類別|計畫期程|經費期程'
)


def init_qa() -> None:
    """啟動時呼叫一次，載入各年度 qa_custom.txt 及 summary 目錄。"""
    global _READY
    _base = Path(__file__).parent
    for year, qa_fname, summary_dir in [
        ("114", "qa_custom_114.txt", _base / "114_output" / "summary"),
        ("113", "qa_custom_113.txt", _base / "113_output" / "summary"),
    ]:
        _load_custom_qa(_QA_DIR / qa_fname, year)
        print(f"[QA] {year} 年自訂 QA：{len(_CUSTOM_QA_BY_YEAR[year])} 組。")
        _load_plan_basics(summary_dir, year)
    _READY = True


def try_structured_answer(question: str, year: str = "114") -> str | None:
    """
    若問題命中對應年度的 qa_custom 或計劃基本資料，回傳完整答案字串；
    否則回傳 None（交給 RAG 處理）。
    """
    if not _READY:
        return None
    q = question.strip()

    # ① 計劃基本資料優先（學校名稱 + 基本資料型關鍵字 → 不被 qa_custom 通用答案攔截）
    plan_ans = _match_plan_basics(q, year)
    if plan_ans:
        return plan_ans

    # ② qa_custom
    qa_list = _CUSTOM_QA_BY_YEAR.get(year, _CUSTOM_QA_BY_YEAR["114"])
    return _match_custom_qa(q, qa_list)


# ── 計劃基本資料載入與比對 ────────────────────────────────

def _load_plan_basics(summary_dir: Path, year: str = "114") -> None:
    """從 summary 目錄讀取各計畫摘要，作為基本資料來源。"""
    if not summary_dir.exists():
        print(f"[QA] 找不到 {summary_dir}，{year} 年計劃基本資料停用。")
        return

    basics = _PLAN_BASICS_BY_YEAR[year]
    count = 0
    for f in summary_dir.glob("*.txt"):
        stem = f.stem
        # 檔名格式：學校_計畫名(id) 或 學校_計畫名
        parts = stem.split("_", 1)
        if len(parts) < 2:
            continue
        school = parts[0].strip()
        plan = re.sub(r'\s*[\(（][^\)）]*[\)）]', '', parts[1]).strip()
        text = _read_text(f)
        if not text:
            continue
        basics.setdefault(school, []).append({"plan_name": plan, "text": text})
        count += 1

    _PLAN_BASIC_SCHOOLS_BY_YEAR[year] = sorted(basics.keys(), key=len, reverse=True)
    print(f"[QA] {year} 年 summary 基本資料：{len(basics)} 間學校，{count} 件計畫。")


def _match_plan_basics(question: str, year: str = "114") -> str | None:
    basics = _PLAN_BASICS_BY_YEAR.get(year, {})
    schools = _PLAN_BASIC_SCHOOLS_BY_YEAR.get(year, [])
    if not basics:
        return None

    q_norm = _half(question).replace(" ", "").replace("　", "")

    matched_school: str | None = None
    for school in schools:
        if school in q_norm:
            matched_school = school
            break
        m_fa = re.search(r'財團法人(.+)', school)
        if m_fa and m_fa.group(1).strip() in q_norm:
            matched_school = school
            break

    if not matched_school:
        return None

    if not _PLAN_INFO_RE.search(question):
        return None

    plans = basics.get(matched_school, [])
    if not plans:
        return None

    # try to find specific plan by plan name keyword matching
    best_plan: dict | None = None
    best_score = 0.0
    for entry in plans:
        score = _phrase_score(q_norm, _half(entry["plan_name"]).lower())
        if score > best_score:
            best_score = score
            best_plan = entry

    if best_score >= 0.5 and best_plan:
        return f"**{matched_school} — {best_plan['plan_name']}** 基本資料\n\n{best_plan['text']}"

    # no specific plan → output all school's plans
    parts = [f"**{matched_school}** 共有 {len(plans)} 件計畫基本資料："]
    for i, entry in enumerate(plans, 1):
        parts.append(f"\n{'─' * 40}\n**計畫 {i}：{entry['plan_name']}**\n{entry['text']}")
    return "\n".join(parts)


# ── 自訂 QA 載入與比對 ────────────────────────────────────

def _load_custom_qa(path: Path, year: str = "114") -> None:
    """解析 qa_custom.txt，每組 Q&A 支援多個同義問法（用 | 分隔）。"""
    if not path.exists():
        print(f"[QA] 找不到 {path.name}，{year} 年自訂 QA 停用。")
        return

    text = _read_text(path)
    if not text:
        return

    current_keywords: list[str] = []
    current_answer_lines: list[str] = []
    in_answer = False
    qa_list = _CUSTOM_QA_BY_YEAR[year]

    def _flush():
        if current_keywords and current_answer_lines:
            # 去掉尾端空行，但保留中間空行（格式用）
            lines = current_answer_lines[:]
            while lines and not lines[-1].strip():
                lines.pop()
            answer = "\n".join(lines).strip()
            if answer:
                qa_list.append({
                    "keywords": current_keywords[:],
                    "answer": answer,
                })

    for raw_line in text.split("\n"):
        line = raw_line.rstrip()

        if line.lstrip().startswith("#"):
            continue

        if line.startswith("Q:"):
            _flush()
            current_keywords = []
            current_answer_lines = []
            in_answer = False
            qs = line[2:].strip()
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
            current_answer_lines.append(line)

    _flush()


def _match_custom_qa(question: str, qa_list: list[dict]) -> str | None:
    """
    比對邏輯：
    - 問題中包含 Q 的任一同義句，視為命中
    - 同義句比對：問題包含該句的所有「實詞」（≥2字元，非助詞）
    """
    q_lower = _half(question).lower()
    best_score = 0
    best_answer = None

    for entry in qa_list:
        for kw_phrase in entry["keywords"]:
            score = _phrase_score(q_lower, _half(kw_phrase).lower())
            if score > best_score:
                best_score = score
                best_answer = entry["answer"]

    if best_score >= 0.8:
        return best_answer
    return None


# 比對時過濾掉的短助詞
_STOP = {"是", "的", "了", "嗎", "呢", "啊", "有", "在", "和", "或", "與", "及",
         "請問", "請", "問", "可以", "告訴我", "什麼", "怎麼", "如何", "哪些",
         "一下", "介紹", "說明"}

# 單字停用詞：中文段落的切分點（例如「的」會把「對應的大學」分成「對應」+「大學」）
_STOP_CHARS = frozenset(s for s in _STOP if len(s) == 1)


def _tokenize(text: str) -> list[str]:
    """切分 text 為有意義的詞：英數詞 + 中文內容詞（單字停用詞作切分點）。"""
    tokens: list[str] = []
    for seg in re.findall(r'[A-Za-z0-9]+|[一-鿿]+', text):
        if seg[0].isascii():
            if seg not in _STOP:
                tokens.append(seg)
        else:
            cur = ""
            for ch in seg:
                if ch in _STOP_CHARS:
                    if len(cur) >= 2 and cur not in _STOP:
                        tokens.append(cur)
                    cur = ""
                else:
                    cur += ch
            if len(cur) >= 2 and cur not in _STOP:
                tokens.append(cur)
    return tokens


def _phrase_score(question: str, phrase: str) -> float:
    """計算 phrase 的關鍵詞在 question 中的命中率（0.0 ~ 1.0）。"""
    tokens = _tokenize(phrase)
    if not tokens:
        return 0.0
    # 去除空格，容忍「SDG2 對應的大學」vs「SDG2對應的大學」
    q_norm = question.replace(" ", "").replace("　", "")
    hits = sum(1 for t in tokens if t in q_norm)
    return hits / len(tokens)


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
