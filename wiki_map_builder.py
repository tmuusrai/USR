"""
Wiki Map Builder
解析 llm_wiki_data/ (優先) 或 114md/ 計畫書 → NetworkX 有向圖。

節點類型：
  university — 學校
  sdg        — SDG 目標節點 (SDG1…SDG17)
  topic      — 標準計畫議題 (6 類)
  concept    — LLM 萃取的細粒度知識概念 (僅 llm_wiki 來源)

邊類型：
  university → sdg      contributes_to
  university → topic    focuses_on
  university → concept  involves       (llm_wiki 來源才有)
  concept    → concept  related_to     (核心概念與連結 section)
"""
import json
import re
from pathlib import Path

import networkx as nx

MD_DIR       = Path("114md")
LLM_WIKI_DIR = Path("llm_wiki_data")
CACHE_FILE   = Path("wiki_graph_cache.json")

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

# 忽略不需加入 concept 節點的通用詞
_SKIP_CONCEPTS = {
    "大學社會責任", "USR", "永續發展", "社會責任", "社區", "計畫",
    "學生", "教師", "大學", "學校", "在地", "台灣", "臺灣",
}
# 也跳過 SDG 節點本身（已由 sdg 類型節點代表）
_SDG_RE = re.compile(r'^SDG\d+$')


def _safe_read(path: Path) -> str | None:
    for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return None


# ── 原始 114md 解析器（舊格式）─────────────────────────────
def _parse_md(text: str) -> dict:
    result = {"uni": "", "plan": "", "sdgs": [], "topics": [],
              "region": "", "concepts": [], "source": "114md"}
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


# ── LLM Wiki 解析器（新格式，含 [[wikilinks]]）──────────────
def _parse_llm_wiki_md(text: str) -> dict:
    result = {"uni": "", "plan": "", "sdgs": [], "topics": [],
              "region": "", "concepts": [], "source": "llm_wiki"}

    # 計畫名稱來自 H1
    m = re.search(r'^#\s+(.+)', text, re.MULTILINE)
    if m:
        result["plan"] = m.group(1).strip()[:60]

    # 解析 Markdown 表格欄位
    for line in text.split("\n")[:25]:
        if "執行學校" in line:
            m = re.search(r'\|\s*執行學校\s*\|\s*(.+?)\s*\|', line)
            if m:
                # 去除可能的 [[link]] 格式
                raw = m.group(1).strip()
                result["uni"] = re.sub(r'\[\[([^\]]+)\]\]', r'\1', raw)

        elif re.search(r'SDG\s*關聯', line):
            m = re.search(r'\|\s*SDG\s*關聯\s*\|\s*(.+?)\s*\|', line)
            if m:
                nums = re.findall(r'\[\[SDG(\d+)\]\]', m.group(1))
                result["sdgs"] = sorted({int(n) for n in nums if 1 <= int(n) <= 17})

        elif "計畫議題" in line:
            m = re.search(r'\|\s*計畫議題\s*\|\s*(.+?)\s*\|', line)
            if m:
                tags = re.findall(r'\[\[([^\]]+)\]\]', m.group(1))
                result["topics"] = [t for t in tags if t in TOPICS]
                # 細粒度概念（不在標準 TOPICS 的 tags 也保留）
                extra = [t for t in tags if t not in TOPICS
                         and t not in _SKIP_CONCEPTS
                         and not _SDG_RE.match(t)]
                result["concepts"].extend(extra)

        elif "實踐場域" in line:
            m = re.search(r'\|\s*實踐場域\s*\|\s*(.+?)\s*\|', line)
            if m:
                locs = re.findall(r'\[\[([^[」\]]*(?:縣|市))\]\]', m.group(1))
                if locs:
                    result["region"] = locs[0]

    # 核心概念與連結 section — 提取所有 [[概念]] 標籤
    core_m = re.search(r'##\s*核心概念與連結(.+?)(?:^##|\Z)',
                       text, re.DOTALL | re.MULTILINE)
    if core_m:
        section = core_m.group(1)
        all_tags = re.findall(r'\[\[([^\]]+)\]\]', section)
        for tag in all_tags:
            if (tag not in _SKIP_CONCEPTS
                    and not _SDG_RE.match(tag)
                    and tag not in TOPICS
                    and tag not in result["concepts"]):
                result["concepts"].append(tag)

    return result


# ── 快取有效性：比較 llm_wiki_data/ 最新修改時間 ──────────
def _cache_valid() -> bool:
    if not CACHE_FILE.exists():
        return False
    cache_mtime = CACHE_FILE.stat().st_mtime
    for p in LLM_WIKI_DIR.rglob("*.md") if LLM_WIKI_DIR.exists() else []:
        if p.stat().st_mtime > cache_mtime:
            return False
    return True


def _load_or_parse() -> list[dict]:
    if _cache_valid():
        print(f"[WIKI] 載入快取 {CACHE_FILE}")
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    # 建立 llm_wiki_data stem → path 查找表
    llm_lookup: dict[str, Path] = {}
    if LLM_WIKI_DIR.exists():
        for p in LLM_WIKI_DIR.rglob("*.md"):
            llm_lookup[p.stem] = p
        print(f"[WIKI] llm_wiki_data/ 共 {len(llm_lookup)} 份已編譯")

    md_files = sorted(MD_DIR.rglob("*.md"))
    print(f"[WIKI] 解析 {len(md_files)} 份文件（llm_wiki 優先）...")
    items = []

    for path in md_files:
        # 優先使用 llm_wiki_data 版本
        if path.stem in llm_lookup:
            text = _safe_read(llm_lookup[path.stem])
            if text:
                d = _parse_llm_wiki_md(text)
            else:
                text = _safe_read(path)
                d = _parse_md(text) if text else {}
        else:
            text = _safe_read(path)
            d = _parse_md(text) if text else {}

        if not d:
            continue

        # 從檔名補充缺失的 uni / plan
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
    llm_count = sum(1 for d in items if d.get("source") == "llm_wiki")
    print(f"[WIKI] 其中 llm_wiki 來源：{llm_count} 筆，114md 來源：{len(items)-llm_count} 筆")
    return items


def build_wiki_graph() -> nx.DiGraph:
    """
    建立多層知識圖譜：
      大學 → SDG / 議題 / 概念
      概念 → 概念（來自 llm_wiki 核心概念與連結 section）
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
            uni_data[uni] = {"plans": [], "sdgs": set(), "topics": set(), "concepts": set()}
        uni_data[uni]["plans"].append({
            "plan": d["plan"],
            "sdgs": d["sdgs"],
            "topics": d["topics"],
            "source": d.get("source", "114md"),
        })
        uni_data[uni]["sdgs"].update(d["sdgs"])
        uni_data[uni]["topics"].update(d["topics"])
        uni_data[uni]["concepts"].update(d.get("concepts", []))

    for uni, info in uni_data.items():
        sdgs     = sorted(info["sdgs"])
        topics   = sorted(info["topics"])
        concepts = sorted(info["concepts"])
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

        for c in concepts:
            if not G.has_node(c):
                G.add_node(c, type="concept", unis=[], count=0)
            node = G.nodes[c]
            # skip if node was already claimed by another type (university, sdg, topic)
            if node.get("type") not in ("concept",):
                continue
            G.add_edge(uni, c, relationship="involves")
            node["count"] = node.get("count", 0) + 1
            node.setdefault("unis", []).append(uni)

    # 移除沒有連結的 SDG / 議題節點
    isolates = [n for n, d in G.nodes(data=True)
                if d.get("type") in ("sdg", "topic") and d.get("count", 0) == 0]
    G.remove_nodes_from(isolates)

    # 移除只有 1 間學校使用的 concept 節點（避免視覺雜訊）
    rare_concepts = [n for n, d in G.nodes(data=True)
                     if d.get("type") == "concept" and d.get("count", 0) < 2]
    G.remove_nodes_from(rare_concepts)

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
    concept_nodes = [(n, d["count"]) for n, d in G.nodes(data=True) if d["type"] == "concept"]
    print(f"Concept 節點：{len(concept_nodes)} 個")
    print("  前10：", concept_nodes[:10])
