"""
Wiki Map Builder
解析 114md/ 計畫書 → NetworkX 有向圖（學校 → SDG / 議題，不含計畫節點）。
計畫資訊存在大學節點的屬性中，供點擊面板顯示。
"""
import json
import re
from pathlib import Path

import networkx as nx

MD_DIR     = Path("114md")
CACHE_FILE = Path("wiki_graph_cache.json")

TOPICS = [
    "在地關懷", "環境永續", "產業鏈結與經濟永續",
    "健康促進與食品安全", "文化永續", "其他社會實踐",
]

SDG_NAMES = {
    1:"消除貧窮",  2:"零飢餓",          3:"良好健康與福祉",
    4:"優質教育",  5:"性別平等",         6:"乾淨用水及衛生",
    7:"可負擔及乾淨能源", 8:"合宜工作與經濟成長", 9:"產業創新與基礎設施",
    10:"減少不平等", 11:"永續城市及社區",  12:"負責任的消費及生產",
    13:"氣候行動",  14:"水下生物",        15:"陸地生物",
    16:"和平正義與強大機構", 17:"全球夥伴關係",
}


def _parse_md(text: str) -> dict:
    result = {"uni": "", "plan": "", "sdgs": [], "topics": [], "region": ""}
    for line in text.split("\n")[:40]:
        if not result["uni"]:
            m = re.match(r'(?:申請學校|學校名稱)[　\s:：]+(.+)', line)
            if m:
                result["uni"] = m.group(1).strip()
        if not result["plan"]:
            m = re.match(r'計畫名稱[　\s:：]+(.+)', line)
            if m:
                result["plan"] = m.group(1).strip()[:60]
        if not result["sdgs"] and "SDG" in line and ("關聯" in line or "sdg" in line.lower()):
            nums = re.findall(r'(?<!\d)([1-9]|1[0-7])(?!\d)', line)
            result["sdgs"] = sorted({int(n) for n in nums})
        if not result["topics"] and "計畫議題" in line:
            result["topics"] = [t for t in TOPICS if t in line]
        if not result["region"] and "實踐場域" in line:
            m = re.search(r'縣市[：:]\s*([^\s，,。]+[縣市])', line)
            if m:
                result["region"] = m.group(1).strip()
    return result


def _safe_read(path: Path) -> str | None:
    for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return None


def _load_or_parse() -> list[dict]:
    if CACHE_FILE.exists():
        print(f"[WIKI] 載入快取 {CACHE_FILE}")
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    md_files = sorted(MD_DIR.rglob("*.md"))
    print(f"[WIKI] 解析 {len(md_files)} 份文件...")
    items = []

    for path in md_files:
        text = _safe_read(path)
        if not text:
            continue
        d = _parse_md(text)
        if not d["uni"] or not d["plan"]:
            stem = path.stem
            if "_" in stem:
                uni_f, plan_f = stem.split("_", 1)
                plan_f = re.sub(r'\s*\([\w\-]+\)\s*', ' ', plan_f).strip()
                plan_f = re.sub(r'\s*\(\d+\)\s*$', '', plan_f).strip()
                if not d["uni"]:
                    d["uni"] = uni_f
                if not d["plan"]:
                    d["plan"] = plan_f[:60]
        if d["uni"] and d["plan"]:
            items.append(d)

    CACHE_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[WIKI] 解析完成，共 {len(items)} 筆計畫")
    return items


def build_wiki_graph() -> nx.DiGraph:
    """
    建立扁平知識圖：大學 → SDG / 議題（無計畫中間節點）。
    計畫列表儲存在大學節點屬性中，供前端點擊顯示。
    """
    items = _load_or_parse()

    G = nx.DiGraph()

    # SDG 節點
    for n, name in SDG_NAMES.items():
        G.add_node(f"SDG{n}", type="sdg", full_name=name, unis=[], count=0)

    # 議題節點
    for t in TOPICS:
        G.add_node(t, type="topic", unis=[], count=0)

    # 歸納每所學校的所有計畫
    uni_data: dict[str, dict] = {}
    for d in items:
        uni = d["uni"]
        if uni not in uni_data:
            uni_data[uni] = {"plans": [], "sdgs": set(), "topics": set()}
        uni_data[uni]["plans"].append({"plan": d["plan"], "sdgs": d["sdgs"], "topics": d["topics"]})
        uni_data[uni]["sdgs"].update(d["sdgs"])
        uni_data[uni]["topics"].update(d["topics"])

    for uni, info in uni_data.items():
        sdgs   = sorted(info["sdgs"])
        topics = sorted(info["topics"])
        G.add_node(uni, type="university",
                   plans=info["plans"],
                   sdgs=sdgs, topics=topics,
                   count=len(info["plans"]))

        for n in sdgs:
            key = f"SDG{n}"
            G.add_edge(uni, key, relationship="contributes_to")
            G.nodes[key]["count"] += 1
            G.nodes[key]["unis"].append(uni)

        for t in topics:
            G.add_edge(uni, t, relationship="focuses_on")
            G.nodes[t]["count"] += 1
            G.nodes[t]["unis"].append(uni)

    # 移除沒有連結的 SDG / 議題節點
    isolates = [n for n, d in G.nodes(data=True)
                if d.get("type") in ("sdg", "topic") and d.get("count", 0) == 0]
    G.remove_nodes_from(isolates)

    print(f"[WIKI] 圖譜：{G.number_of_nodes()} 個節點，{G.number_of_edges()} 條邊")
    return G


# ── 向下相容（app.py 舊呼叫）──────────────────────────
def build_relations(max_files: int = 0) -> list[dict]:
    return _load_or_parse()

def build_graph(relations: list[dict]) -> nx.DiGraph:
    return build_wiki_graph()


if __name__ == "__main__":
    G = build_wiki_graph()
    print("大學節點範例：", [(n, d["count"]) for n, d in G.nodes(data=True) if d["type"] == "university"][:3])
    print("SDG 節點範例：", [(n, d["count"]) for n, d in G.nodes(data=True) if d["type"] == "sdg"][:3])
