"""
將 NetworkX 圖轉成 Pyvis 互動 HTML。
可獨立呼叫，也供 Flask 路由使用。
"""
from pathlib import Path
import networkx as nx
from pyvis.network import Network

OUTPUT_HTML = Path("templates/wiki_map.html")

# 關係標籤中文化
REL_LABEL = {
    "executes":       "執行",
    "contributes_to": "對應",
    "focuses_on":     "聚焦",
    "located_in":     "場域",
    "uses":           "採用",
    "related_to":     "相關",
}

TYPE_COLOR = {
    "university": "#4f86c6",
    "plan":       "#f4a261",
    "sdg":        "#2ec4b6",
    "topic":      "#e76f51",
    "region":     "#8ecae6",
    "keyword":    "#a8dadc",
}

TYPE_SIZE = {
    "university": 28,
    "plan":       20,
    "sdg":        24,
    "topic":      18,
    "region":     16,
    "keyword":    14,
}

TYPE_LABEL_ZH = {
    "university": "學校",
    "plan":       "計畫",
    "sdg":        "SDG",
    "topic":      "議題",
    "region":     "場域",
    "keyword":    "關鍵字",
}


def render_wiki_map(G: nx.DiGraph, output: Path = OUTPUT_HTML) -> Path:
    """
    把 NetworkX 有向圖渲染成 Pyvis 互動 HTML，存到 templates/wiki_map.html。
    回傳輸出路徑。
    """
    net = Network(
        height="90vh",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        notebook=False,
    )

    # 物理引擎設定：讓圖自動排列且不會堆在一起
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.005,
          "springLength": 120,
          "springConstant": 0.1
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 150 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
        "color": { "inherit": false, "color": "#555577" },
        "smooth": { "type": "dynamic" }
      }
    }
    """)

    # 加入節點
    for node, data in G.nodes(data=True):
        ntype = data.get("type", "keyword")
        color = TYPE_COLOR.get(ntype, "#aaa")
        size  = TYPE_SIZE.get(ntype, 14)
        label_zh = TYPE_LABEL_ZH.get(ntype, ntype)
        net.add_node(
            node,
            label=node,
            title=f"【{label_zh}】{node}",
            color=color,
            size=size,
            font={"size": 12, "color": "#ffffff"},
        )

    # 加入邊
    for src, tgt, data in G.edges(data=True):
        rel = data.get("relationship", "related_to")
        net.add_edge(
            src, tgt,
            title=REL_LABEL.get(rel, rel),
            label=REL_LABEL.get(rel, rel),
            font={"size": 9, "color": "#aaaacc"},
        )

    # 產生 HTML
    output.parent.mkdir(exist_ok=True)
    net.save_graph(str(output))

    # 注入圖例
    _inject_legend(output)

    print(f"[WIKI] 地圖已輸出：{output}")
    return output


def _inject_legend(html_path: Path) -> None:
    """在 Pyvis HTML body 開頭注入圖例說明。"""
    legend_html = """
<div style="position:fixed;top:12px;left:12px;z-index:999;
            background:rgba(20,20,40,0.88);border:1px solid #444;
            border-radius:8px;padding:12px 16px;font-size:12px;color:#ddd;">
  <b style="font-size:13px;">節點類型</b><br>
  <span style="color:#4f86c6">■</span> 學校 &nbsp;
  <span style="color:#f4a261">■</span> 計畫 &nbsp;
  <span style="color:#2ec4b6">■</span> SDG &nbsp;
  <span style="color:#e76f51">■</span> 議題<br>
  <span style="color:#8ecae6">■</span> 場域 &nbsp;
  <span style="color:#a8dadc">■</span> 關鍵字
</div>
"""
    content = html_path.read_text(encoding="utf-8")
    content = content.replace("<body>", "<body>\n" + legend_html, 1)
    html_path.write_text(content, encoding="utf-8")
