"""
Wiki Map Builder
讀取 114md/ 的計畫書 → LLM 提取實體與關係 → NetworkX 有向圖
結果快取到 wiki_graph_cache.json，避免重複呼叫 LLM。
"""
import json
import os
import re
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

MD_DIR     = Path("114md")
CACHE_FILE = Path("wiki_graph_cache.json")

# ── LLM ───────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_MODEL", "gemini-2.5-pro-preview"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.0,
)

# ── Prompt ────────────────────────────────────────────
EXTRACT_PROMPT = """你是知識圖譜專家。請閱讀以下 USR 計畫書，提取重要實體與關係。

【實體類型】
- university  學校名稱
- plan        計畫名稱
- sdg         SDG目標（如 SDG3、SDG8）
- topic       計畫議題（如 高齡照護、地方創生、食農教育）
- region      實踐場域地名（如 臺南、花蓮、屏東）
- keyword     關鍵方法或技術（如 智慧農業、青銀共創、循環經濟）

【關係類型】
- executes       學校執行計畫
- contributes_to 計畫對應SDG目標
- focuses_on     計畫聚焦議題
- located_in     計畫場域所在地
- uses           計畫採用的方法/技術

【輸出規則】
- 只輸出 JSON 陣列，不要任何其他文字、解釋或 markdown 標記
- 每筆格式：{"source":"A","source_type":"university","target":"B","target_type":"plan","relationship":"executes"}
- source 和 target 都必須是具體名詞，不超過 15 個字
- 最多輸出 30 筆關係

【計畫書內容】
{text}
"""


# ── 單份文件提取 ──────────────────────────────────────
def _extract_relations(text: str, filename: str) -> list[dict]:
    """呼叫 LLM 從一份計畫書中提取實體關係，回傳 list of dicts。"""
    # 只取前 3000 字元，避免 token 爆炸
    snippet = text[:3000]
    try:
        res = llm.invoke([HumanMessage(content=EXTRACT_PROMPT.format(text=snippet))])
        raw = res.content
        if isinstance(raw, list):
            raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
        # 取出 JSON 陣列部分
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if not m:
            print(f"  [WARN] {filename}：LLM 未回傳 JSON")
            return []
        relations = json.loads(m.group())
        return [r for r in relations if isinstance(r, dict)
                and "source" in r and "target" in r and "relationship" in r]
    except Exception as e:
        print(f"  [WARN] {filename} 提取失敗：{e}")
        return []


# ── 批次處理所有 MD 檔 ────────────────────────────────
def build_relations(max_files: int = 50) -> list[dict]:
    """
    讀取 114md/ 的 MD 檔，逐一呼叫 LLM 提取關係。
    max_files：限制處理數量（節省 API 費用），設 0 表示全部。
    結果快取到 wiki_graph_cache.json。
    """
    if CACHE_FILE.exists():
        print(f"[WIKI] 載入快取 {CACHE_FILE}")
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    md_files = sorted(MD_DIR.rglob("*.md"))
    if max_files:
        md_files = md_files[:max_files]
    print(f"[WIKI] 處理 {len(md_files)} 份文件...")

    all_relations: list[dict] = []
    for i, path in enumerate(md_files):
        print(f"  [{i+1}/{len(md_files)}] {path.name}")
        text = _safe_read(path)
        if not text:
            continue
        relations = _extract_relations(text, path.name)
        all_relations.extend(relations)

    CACHE_FILE.write_text(json.dumps(all_relations, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"[WIKI] 完成，共 {len(all_relations)} 筆關係，已存入 {CACHE_FILE}")
    return all_relations


def _safe_read(path: Path) -> str | None:
    for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return None


# ── 建立 NetworkX 有向圖 ──────────────────────────────
def build_graph(relations: list[dict]) -> nx.DiGraph:
    """將關係清單轉成 NetworkX 有向圖，節點帶 type 屬性。"""
    G = nx.DiGraph()

    # 節點顏色對應類型
    TYPE_COLOR = {
        "university": "#4f86c6",
        "plan":       "#f4a261",
        "sdg":        "#2ec4b6",
        "topic":      "#e76f51",
        "region":     "#8ecae6",
        "keyword":    "#a8dadc",
    }

    for r in relations:
        src, tgt = r["source"].strip(), r["target"].strip()
        if not src or not tgt or src == tgt:
            continue

        src_type = r.get("source_type", "keyword")
        tgt_type = r.get("target_type", "keyword")

        if src not in G:
            G.add_node(src, type=src_type, color=TYPE_COLOR.get(src_type, "#ccc"))
        if tgt not in G:
            G.add_node(tgt, type=tgt_type, color=TYPE_COLOR.get(tgt_type, "#ccc"))

        G.add_edge(src, tgt, relationship=r.get("relationship", "related_to"))

    print(f"[WIKI] 圖譜：{G.number_of_nodes()} 個節點，{G.number_of_edges()} 條邊")
    return G


# ── 主程式（獨立執行時使用）─────────────────────────
if __name__ == "__main__":
    relations = build_relations(max_files=30)
    G = build_graph(relations)
    print("節點範例：", list(G.nodes(data=True))[:3])
    print("邊範例：",   list(G.edges(data=True))[:3])
