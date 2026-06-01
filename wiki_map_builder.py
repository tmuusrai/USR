"""
Wiki Map Builder
直接解析 114md/ 的計畫書結構化欄位（學校、SDG、議題、場域）→ NetworkX 有向圖。
不使用 LLM，速度快（< 10 秒），結果快取到 wiki_graph_cache.json。
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
    1:"消除貧窮", 2:"零飢餓", 3:"良好健康與福祉", 4:"優質教育",
    5:"性別平等", 6:"乾淨用水及衛生", 7:"可負擔及乾淨能源",
    8:"合宜工作與經濟成長", 9:"產業創新與基礎設施", 10:"減少不平等",
    11:"永續城市及社區", 12:"負責任的消費及生產", 13:"氣候行動",
    14:"水下生物", 15:"陸地生物", 16:"和平正義與強大機構", 17:"全球夥伴關係",
}


def _parse_md(text: str) -> dict:
    """從 MD 檔前 40 行解析結構化欄位。"""
    result = {"uni": "", "plan": "", "sdgs": [], "topics": [], "region": ""}
    lines = text.split("\n")[:40]

    for line in lines:
        # 學校名稱
        if not result["uni"]:
            m = re.match(r'(?:申請學校|學校名稱)[　\s:：]+(.+)', line)
            if m:
                result["uni"] = m.group(1).strip()

        # 計畫名稱
        if not result["plan"]:
            m = re.match(r'計畫名稱[　\s:：]+(.+)', line)
            if m:
                result["plan"] = m.group(1).strip()[:40]

        # SDG 號碼（從 SDGs關聯議題 行）
        if not result["sdgs"] and "SDG" in line and ("關聯" in line or "sdg" in line.lower()):
            nums = re.findall(r'(?<!\d)([1-9]|1[0-7])(?!\d)', line)
            result["sdgs"] = sorted({int(n) for n in nums})

        # 計畫議題
        if not result["topics"] and "計畫議題" in line:
            result["topics"] = [t for t in TOPICS if t in line]

        # 實踐場域（縣市）
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


def build_relations(max_files: int = 0) -> list[dict]:
    """
    解析 114md/ 的 MD 檔，回傳關係清單。
    max_files=0 表示全部處理。結果快取到 wiki_graph_cache.json。
    """
    if CACHE_FILE.exists():
        print(f"[WIKI] 載入快取 {CACHE_FILE}")
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    md_files = sorted(MD_DIR.rglob("*.md"))
    if max_files:
        md_files = md_files[:max_files]
    print(f"[WIKI] 解析 {len(md_files)} 份文件...")

    all_relations: list[dict] = []

    for path in md_files:
        text = _safe_read(path)
        if not text:
            continue

        d = _parse_md(text)
        uni  = d["uni"]
        plan = d["plan"]

        # 嘗試從檔名補齊
        if not uni or not plan:
            stem = path.stem
            if "_" in stem:
                uni_f, plan_f = stem.split("_", 1)
                plan_f = re.sub(r'\s*\([\w\-]+\)\s*', ' ', plan_f).strip()
                plan_f = re.sub(r'\s*\(\d+\)\s*$', '', plan_f).strip()
                if not uni:
                    uni = uni_f
                if not plan:
                    plan = plan_f[:40]

        if not uni or not plan:
            continue

        # 學校 → 計畫
        all_relations.append({
            "source": uni, "source_type": "university",
            "target": plan, "target_type": "plan",
            "relationship": "executes",
        })

        # 計畫 → SDG
        for n in d["sdgs"]:
            sdg_label = f"SDG{n}"
            all_relations.append({
                "source": plan, "source_type": "plan",
                "target": sdg_label, "target_type": "sdg",
                "relationship": "contributes_to",
            })

        # 計畫 → 議題
        for topic in d["topics"]:
            all_relations.append({
                "source": plan, "source_type": "plan",
                "target": topic, "target_type": "topic",
                "relationship": "focuses_on",
            })

        # 計畫 → 場域
        if d["region"]:
            all_relations.append({
                "source": plan, "source_type": "plan",
                "target": d["region"], "target_type": "region",
                "relationship": "located_in",
            })

    CACHE_FILE.write_text(
        json.dumps(all_relations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[WIKI] 完成，共 {len(all_relations)} 筆關係，已存入 {CACHE_FILE}")
    return all_relations


def build_graph(relations: list[dict]) -> nx.DiGraph:
    """將關係清單轉成 NetworkX 有向圖。"""
    TYPE_COLOR = {
        "university": "#4f86c6",
        "plan":       "#f4a261",
        "sdg":        "#2ec4b6",
        "topic":      "#e76f51",
        "region":     "#8ecae6",
        "keyword":    "#a8dadc",
    }

    G = nx.DiGraph()
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


if __name__ == "__main__":
    relations = build_relations()
    G = build_graph(relations)
    print("節點範例：", list(G.nodes(data=True))[:3])
    print("邊範例：",   list(G.edges(data=True))[:3])
