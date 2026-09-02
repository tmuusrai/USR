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
from difflib import SequenceMatcher
from pathlib import Path


def _half(s: str) -> str:
    """全形英數 → 半形（NFKC normalization）"""
    return unicodedata.normalize('NFKC', s)

# ── 內部儲存 ──────────────────────────────────────────────
_CUSTOM_QA_BY_YEAR: dict[str, list[dict]] = {"114": [], "113": []}

_READY = False

# qa_data/ 資料夾預設與 structured_qa.py 同層
_QA_DIR = Path(__file__).parent / "qa_data"

def init_qa() -> None:
    """啟動時呼叫一次，載入各年度 qa_custom.txt。"""
    global _READY
    for year, qa_fname in [("114", "qa_custom_114.txt"), ("113", "qa_custom_113.txt")]:
        _load_custom_qa(_QA_DIR / qa_fname, year)
        print(f"[QA] {year} 年自訂 QA：{len(_CUSTOM_QA_BY_YEAR[year])} 組。")
    _READY = True


def try_structured_answer(question: str, year: str = "114") -> str | None:
    """
    若問題命中對應年度的 qa_custom，回傳完整答案字串；
    否則回傳 None（交給 RAG 處理）。
    """
    if not _READY:
        return None
    qa_list = _CUSTOM_QA_BY_YEAR.get(year, _CUSTOM_QA_BY_YEAR["114"])
    return _match_custom_qa(question.strip(), qa_list)


# ── 計劃基本資料載入與比對 ────────────────────────────────

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


def _bigram_score(a: str, b: str) -> float:
    """字元 bigram Dice 係數：容忍同義詞、不同語序的模糊比對。"""
    a_c = re.sub(r'\s+', '', a)
    b_c = re.sub(r'\s+', '', b)
    bg_a = {a_c[i:i+2] for i in range(len(a_c) - 1)}
    bg_b = {b_c[i:i+2] for i in range(len(b_c) - 1)}
    if not bg_a or not bg_b:
        return 0.0
    return 2 * len(bg_a & bg_b) / (len(bg_a) + len(bg_b))


def _seq_score(a: str, b: str) -> float:
    """difflib 字元序列相似度。"""
    a_c = re.sub(r'\s+', '', a)
    b_c = re.sub(r'\s+', '', b)
    return SequenceMatcher(None, a_c, b_c).ratio()


def _match_custom_qa(question: str, qa_list: list[dict]) -> str | None:
    """
    模糊比對邏輯（三層）：
    1. Token 命中率：phrase 的關鍵詞有幾個出現在問題中
    2. Bigram Dice：字元 bigram 重疊（容忍「幾件」vs「幾個」等近義詞）
    3. SequenceMatcher：整句字元序列相似度
    取三者最高分，門檻 0.75。收集所有候選後，依分數高→低找第一個
    _has_extra_content 通過的 phrase，避免高分但被擋的 phrase 遮蔽正確 phrase。
    """
    q_lower = _half(question).lower()
    candidates: list[tuple[float, str, str]] = []  # (score, answer, phrase)

    for entry in qa_list:
        for kw_phrase in entry["keywords"]:
            phrase_norm = _half(kw_phrase).lower()
            token_s  = _phrase_score(q_lower, phrase_norm)
            bigram_s = _bigram_score(q_lower, phrase_norm)
            seq_s    = _seq_score(q_lower, phrase_norm)
            score = max(token_s, bigram_s * 0.9, seq_s * 0.85)
            if score >= 0.75:
                candidates.append((score, entry["answer"], kw_phrase))

    candidates.sort(reverse=True)
    for _score, answer, phrase in candidates:
        if not _has_extra_content(question, phrase):
            return answer
    return None


# 列舉/疑問停用詞（不算額外內容詞）
_QUERY_STOPS = {
    "哪些", "哪幾", "有哪", "有幾", "多少", "計畫", "學校", "大學", "相關",
    "有關", "請問", "告訴", "說明", "介紹", "列出", "幾個", "幾間", "幾所",
    "幾件", "這些", "那些", "所有", "全部", "各", "裡", "中", "的", "有",
    "是", "哪", "什麼", "為何", "怎麼", "如何", "嗎", "呢", "做", "在", "及",
}

def _has_extra_content(question: str, phrase: str) -> bool:
    """若問題有實質內容詞完全不在 phrase 覆蓋範圍內，回傳 True（讓 RAG 處理）。
    使用子字串包含判斷：q_token 是某個 phrase_token 的子字串（或反向），視為已覆蓋。
    """
    phrase_tokens = set(_tokenize(_half(phrase).lower()))
    q_tokens = _tokenize(_half(question).lower())

    def _covered(tok: str) -> bool:
        if tok in _QUERY_STOPS:
            return True
        # 正向：tok 是某 phrase_token 的子字串，或某 phrase_token 是 tok 的子字串
        return any(tok in pt or pt in tok for pt in phrase_tokens)

    extra = [t for t in q_tokens if not _covered(t) and len(t) >= 2]
    return len(extra) > 0


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


_SDG_RE = re.compile(r'^SDG\d+$', re.IGNORECASE)

def _token_in_q(token: str, q_norm: str) -> bool:
    """檢查 token 是否在 q_norm 中，SDG 數字需要 word boundary（SDG1 不能匹配 SDG11）。"""
    if _SDG_RE.match(token):
        return bool(re.search(re.escape(token) + r'(?!\d)', q_norm, re.IGNORECASE))
    return token in q_norm

def _phrase_score(question: str, phrase: str) -> float:
    """計算 phrase 的關鍵詞在 question 中的命中率（0.0 ~ 1.0）。"""
    tokens = _tokenize(phrase)
    if not tokens:
        return 0.0
    # 去除空格，容忍「SDG2 對應的大學」vs「SDG2對應的大學」
    q_norm = question.replace(" ", "").replace("　", "")
    hits = sum(1 for t in tokens if _token_in_q(t, q_norm))
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
