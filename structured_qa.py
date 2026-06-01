"""
結構化 QA 模組：
  從 qa_custom.txt 載入人工撰寫的 Q&A 對（含 SDG 索引與知識型問題）

用法：
    from structured_qa import init_qa, try_structured_answer
    init_qa()
    answer = try_structured_answer("SDG8 有哪些學校？")  # str 或 None
"""
import re
from pathlib import Path

# ── 內部儲存 ──────────────────────────────────────────────
_CUSTOM_QA: list[dict] = []              # [{"keywords": [...], "answer": str}]
_READY = False

# qa_data/ 資料夾預設與 structured_qa.py 同層
_QA_DIR = Path(__file__).parent / "qa_data"
_QA_CUSTOM_PATH = _QA_DIR / "qa_custom.txt"


def init_qa() -> None:
    """啟動時呼叫一次，載入 qa_data/qa_custom.txt。"""
    global _READY
    _load_custom_qa(_QA_CUSTOM_PATH)
    _READY = True
    print(f"[QA] 自訂 QA：{len(_CUSTOM_QA)} 組。")


def try_structured_answer(question: str) -> str | None:
    """
    若問題命中 qa_custom.txt，回傳完整答案字串；否則回傳 None（交給 RAG 處理）。
    """
    if not _READY:
        return None
    return _match_custom_qa(question.strip())


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
            # 去掉尾端空行，但保留中間空行（格式用）
            lines = current_answer_lines[:]
            while lines and not lines[-1].strip():
                lines.pop()
            answer = "\n".join(lines).strip()
            if answer:
                _CUSTOM_QA.append({
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

    if best_score >= 0.6:
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
