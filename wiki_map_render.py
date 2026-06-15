"""
將 NetworkX 圖轉成 Pyvis 互動 HTML，注入點擊詳情面板。
"""
import json
import re
from pathlib import Path
import networkx as nx
from pyvis.network import Network

CONCEPTS_DIR = Path("llm_wiki_data/concepts")


def _load_concept_page(concept: str) -> str | None:
    """讀取 AI 合成的概念百科頁，轉成簡易 HTML 供面板顯示。"""
    path = CONCEPTS_DIR / f"{concept}.md"
    if not path.exists():
        return None
    try:
        md = path.read_text(encoding="utf-8")
    except Exception:
        return None

    # 簡易 Markdown → HTML
    lines, html = md.split("\n"), []
    for line in lines:
        line = re.sub(r'\[\[([^\]]+)\]\]',
                      r'<span class="wl">\1</span>', line)
        if line.startswith("# "):
            html.append(f'<h2 class="cp-h1">{line[2:].strip()}</h2>')
        elif line.startswith("## "):
            html.append(f'<h3 class="cp-h2">{line[3:].strip()}</h3>')
        elif line.startswith("- ") or line.startswith("* "):
            html.append(f'<li>{line[2:].strip()}</li>')
        elif line.startswith("**") and line.endswith("**"):
            html.append(f'<b>{line[2:-2]}</b>')
        elif line.strip() == "":
            html.append("<br>")
        else:
            html.append(f'<p>{line.strip()}</p>')
    return "\n".join(html)

OUTPUT_HTML = Path("static/wiki_map.html")

REL_LABEL = {
    "executes":       "執行",
    "contributes_to": "對應 SDG",
    "focuses_on":     "聚焦議題",
    "located_in":     "場域",
    "uses":           "採用",
    "involves":       "涉及概念",
    "related_to":     "概念相關",
}

# 每種關係類型的邊顏色（預設狀態）
EDGE_COLOR = {
    "contributes_to": "#2ec4b6",   # teal   — 對應 SDG
    "focuses_on":     "#e76f51",   # orange — 聚焦議題
    "involves":       "#a78bfa",   # violet — 涉及概念
    "related_to":     "#6b7fd4",   # blue   — 概念相關
    "executes":       "#f4a261",
    "located_in":     "#8ecae6",
    "uses":           "#ffd166",
}

EDGE_COLOR_DEFAULT = "#334466"   # 沒有對應類型時的 fallback

TYPE_COLOR = {
    "university": "#4f86c6",
    "plan":       "#f4a261",
    "sdg":        "#2ec4b6",
    "topic":      "#e76f51",
    "region":     "#8ecae6",
    "keyword":    "#a8dadc",
}

TYPE_SIZE_BASE = {
    "university": 16,
    "sdg":        30,
    "topic":      24,
    "region":     14,
    "keyword":    12,
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
    net = Network(
        height="100vh",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        notebook=False,
    )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.008,
          "springLength": 160,
          "springConstant": 0.08,
          "damping": 0.6
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 300, "updateInterval": 50, "fit": true }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "navigationButtons": true,
        "keyboard": true,
        "multiselect": false
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
        "color": { "inherit": false },
        "smooth": { "type": "continuous" },
        "width": 1.2
      },
      "nodes": {
        "borderWidth": 0,
        "shadow": true
      }
    }
    """)

    # ── 建立 JS 用的節點元資料 ──
    node_meta = {}

    for node, data in G.nodes(data=True):
        ntype = data.get("type", "keyword")
        color = TYPE_COLOR.get(ntype, "#aaa")
        base_size = TYPE_SIZE_BASE.get(ntype, 12)

        if ntype == "university":
            plans = data.get("plans", [])
            sdgs  = data.get("sdgs", [])
            size  = base_size + min(len(plans) * 1.5, 12)
            label = node
            title = f"【學校】{node}｜{len(plans)} 個計畫"
            node_meta[node] = {
                "type": "university", "name": node,
                "count": len(plans),
                "sdgs": sdgs,
                "plans": [p["plan"] for p in plans],
            }

        elif ntype == "sdg":
            cnt  = data.get("count", 0)
            name = data.get("full_name", "")
            size = base_size + min(cnt * 1.2, 20)
            label = node
            title = f"【SDG】{node}　{name}｜{cnt} 所學校"
            node_meta[node] = {
                "type": "sdg", "name": node,
                "full_name": name,
                "count": cnt,
                "unis": data.get("unis", [])[:30],
            }

        elif ntype == "topic":
            cnt  = data.get("count", 0)
            size = base_size + min(cnt * 1.0, 14)
            label = node
            title = f"【議題】{node}｜{cnt} 所學校"
            node_meta[node] = {
                "type": "topic", "name": node,
                "count": cnt,
                "unis": data.get("unis", [])[:30],
            }

        elif ntype == "concept":
            cnt  = data.get("count", 0)
            size = base_size + min(cnt * 0.8, 16)
            label = node
            title = f"【概念】{node}｜{cnt} 所學校"
            page_html = _load_concept_page(node)
            node_meta[node] = {
                "type": "concept", "name": node,
                "count": cnt,
                "unis": data.get("unis", [])[:20],
                "page": page_html,
            }

        else:
            size  = base_size
            label = node
            title = node
            node_meta[node] = {"type": ntype, "name": node}

        net.add_node(
            node,
            label=label,
            title=title,
            color=color,
            size=size,
            font={"size": 11, "color": "#ffffff"},
        )

    # ── 邊：按關係類型上色，並記錄 edge_meta 供 JS 用 ──
    edge_meta = {}   # edge_id → { color, rel, label }
    for idx, (src, tgt, data) in enumerate(G.edges(data=True)):
        rel   = data.get("relationship", "related_to")
        color = EDGE_COLOR.get(rel, EDGE_COLOR_DEFAULT)
        eid   = f"e{idx}"
        edge_meta[eid] = {
            "color": color,
            "rel":   rel,
            "label": REL_LABEL.get(rel, rel),
        }
        net.add_edge(
            src, tgt,
            id=eid,
            title=REL_LABEL.get(rel, rel),
            color={"color": color, "opacity": 0.65},
            width=1.2,
        )

    output.parent.mkdir(exist_ok=True)
    net.save_graph(str(output))

    _inject_ui(output, node_meta, edge_meta)

    print(f"[WIKI] 地圖已輸出：{output}")
    return output


def _inject_ui(html_path: Path, node_meta: dict, edge_meta: dict) -> None:
    """注入圖例、詳情面板與 click 事件（含邊高亮）。"""

    meta_json      = json.dumps(node_meta, ensure_ascii=False)
    edge_meta_json = json.dumps(edge_meta, ensure_ascii=False)

    legend_html = """
<div id="wm-legend" style="position:fixed;top:12px;left:12px;z-index:999;
  background:rgba(10,10,28,.9);border:1px solid #334;border-radius:10px;
  padding:12px 16px;font-size:12px;color:#ccc;line-height:1.9">
  <b style="font-size:13px;color:#fff">節點類型</b><br>
  <span style="color:#4f86c6">■</span> 學校 &nbsp;
  <span style="color:#2ec4b6">■</span> SDG &nbsp;
  <span style="color:#e76f51">■</span> 議題
  <hr style="border-color:#334;margin:6px 0">
  <b style="font-size:12px;color:#fff">線條關係</b><br>
  <span style="color:#2ec4b6">─</span> 對應 SDG &nbsp;
  <span style="color:#e76f51">─</span> 聚焦議題<br>
  <span style="color:#a78bfa">─</span> 涉及概念 &nbsp;
  <span style="color:#6b7fd4">─</span> 概念相關
  <br><span style="font-size:.75rem;color:#666;margin-top:4px;display:block">點擊節點查看詳情</span>
</div>
"""

    panel_html = """
<div id="wm-panel" style="position:fixed;top:0;right:-340px;bottom:0;width:320px;
  background:rgba(12,12,30,.97);border-left:1px solid #334;z-index:998;
  display:flex;flex-direction:column;transition:right .3s ease;font-family:sans-serif">
  <div style="padding:14px 16px;border-bottom:1px solid #334;display:flex;align-items:center;gap:8px">
    <span id="wm-panel-badge" style="font-size:.7rem;padding:2px 8px;border-radius:12px;
      background:#334;color:#aaa">-</span>
    <span id="wm-panel-title" style="flex:1;font-size:.95rem;font-weight:700;color:#e0e0e0">
      點擊任一節點</span>
    <button onclick="closePanel()" style="background:none;border:none;color:#888;
      font-size:1.2rem;cursor:pointer;line-height:1;padding:2px 4px">✕</button>
  </div>
  <div id="wm-panel-body" style="flex:1;overflow-y:auto;padding:14px 16px;
    font-size:.84rem;color:#ccc;line-height:1.8">
    <span style="color:#555">選取節點後顯示詳細資訊</span>
  </div>
</div>
"""

    inject_js = f"""
<script>
var _meta     = {meta_json};
var _edgeMeta = {edge_meta_json};

// ── 邊高亮 ──────────────────────────────────────────────
function highlightEdges(nodeId) {{
  var connected = new Set(network.getConnectedEdges(nodeId));
  var updates = network.body.data.edges.getIds().map(function(eid) {{
    var em = _edgeMeta[eid];
    if (connected.has(eid)) {{
      return {{ id: eid, color: {{ color: em ? em.color : '#aaa', opacity: 1.0 }}, width: 2.8 }};
    }} else {{
      return {{ id: eid, color: {{ color: '#1a1a3a', opacity: 0.2 }}, width: 0.8 }};
    }}
  }});
  network.body.data.edges.update(updates);
}}

function resetEdges() {{
  var updates = network.body.data.edges.getIds().map(function(eid) {{
    var em = _edgeMeta[eid];
    return {{ id: eid, color: {{ color: em ? em.color : '#334466', opacity: 0.65 }}, width: 1.2 }};
  }});
  network.body.data.edges.update(updates);
}}

// ── 詳情面板 ────────────────────────────────────────────
function closePanel() {{
  document.getElementById('wm-panel').style.right = '-340px';
  resetEdges();
}}

function openPanel(nodeId) {{
  var m = _meta[nodeId];
  if (!m) return;
  var badge = document.getElementById('wm-panel-badge');
  var title = document.getElementById('wm-panel-title');
  var body  = document.getElementById('wm-panel-body');
  var typeLabel = {{university:'學校', sdg:'SDG', topic:'議題', region:'場域', concept:'概念'}};
  var typeColor = {{university:'#4f86c6', sdg:'#2ec4b6', topic:'#e76f51', region:'#8ecae6', concept:'#a78bfa'}};
  badge.textContent = typeLabel[m.type] || m.type;
  badge.style.background = typeColor[m.type] || '#334';
  badge.style.color = '#fff';
  title.textContent = m.name;

  var html = '';
  if (m.type === 'university') {{
    html += '<p style="color:#888;margin-bottom:8px">計畫數：<b style="color:#fff">' + m.count + '</b></p>';
    html += '<p style="color:#888;margin-bottom:12px">SDG：<b style="color:#2ec4b6">' +
            (m.sdgs.length ? m.sdgs.map(function(n){{return 'SDG'+n;}}).join('、') : '—') + '</b></p>';
    html += '<div style="border-top:1px solid #223;padding-top:10px">';
    m.plans.forEach(function(p) {{
      html += '<div style="padding:6px 0;border-bottom:1px solid #1a1a30;color:#bbb;font-size:.82rem">▪ ' + p + '</div>';
    }});
    html += '</div>';
  }} else if (m.type === 'sdg') {{
    html += '<p style="color:#aaa;margin-bottom:10px">' + m.full_name + '</p>';
    html += '<p style="color:#888;margin-bottom:12px">參與學校：<b style="color:#fff">' + m.count + '</b> 所</p>';
    html += '<div style="border-top:1px solid #223;padding-top:10px">';
    (m.unis||[]).forEach(function(u) {{
      html += '<div style="padding:5px 0;border-bottom:1px solid #1a1a30;color:#bbb;font-size:.82rem">▪ ' + u + '</div>';
    }});
    html += '</div>';
  }} else if (m.type === 'topic') {{
    html += '<p style="color:#888;margin-bottom:12px">參與學校：<b style="color:#fff">' + m.count + '</b> 所</p>';
    html += '<div style="border-top:1px solid #223;padding-top:10px">';
    (m.unis||[]).forEach(function(u) {{
      html += '<div style="padding:5px 0;border-bottom:1px solid #1a1a30;color:#bbb;font-size:.82rem">▪ ' + u + '</div>';
    }});
    html += '</div>';
  }} else if (m.type === 'concept') {{
    if (m.page) {{
      html += '<style>.cp-h1{{color:#a8dadc;font-size:1rem;margin:8px 0 4px}}.cp-h2{{color:#7ecfd0;font-size:.88rem;margin:10px 0 4px;border-bottom:1px solid #2a2a4a;padding-bottom:3px}}.wl{{color:#4f86c6;font-weight:600}}li{{margin:3px 0 3px 12px;color:#bbb;font-size:.82rem}}p{{color:#ccc;font-size:.82rem;margin:3px 0;line-height:1.6}}</style>';
      html += m.page;
    }} else {{
      html += '<p style="color:#888;margin-bottom:12px">出現在 <b style="color:#fff">' + m.count + '</b> 所學校的計畫</p>';
      html += '<div style="border-top:1px solid #223;padding-top:10px">';
      (m.unis||[]).forEach(function(u) {{
        html += '<div style="padding:5px 0;border-bottom:1px solid #1a1a30;color:#bbb;font-size:.82rem">▪ ' + u + '</div>';
      }});
      html += '</div>';
    }}
  }}
  body.innerHTML = html;
  document.getElementById('wm-panel').style.right = '0';
}}

// ── vis.js 事件 ──────────────────────────────────────────
(function waitForNetwork() {{
  if (typeof network !== 'undefined') {{
    var _freezeTimer = null;

    function freeze() {{
      network.setOptions({{ physics: false }});
    }}

    network.on('stabilizationIterationsDone', function() {{
      freeze();
    }});

    network.on('dragStart', function() {{
      if (_freezeTimer) clearTimeout(_freezeTimer);
      network.setOptions({{ physics: {{
        enabled: true,
        forceAtlas2Based: {{
          gravitationalConstant: -40,
          springLength: 160,
          springConstant: 0.04,
          damping: 0.9
        }}
      }} }});
    }});

    network.on('dragEnd', function() {{
      _freezeTimer = setTimeout(freeze, 1000);
    }});

    network.on('click', function(params) {{
      if (params.nodes.length > 0) {{
        openPanel(params.nodes[0]);
        highlightEdges(params.nodes[0]);
      }} else {{
        closePanel();
      }}
    }});
  }} else {{
    setTimeout(waitForNetwork, 100);
  }}
}})();
</script>
"""

    content = html_path.read_text(encoding="utf-8")
    content = content.replace("<body>", "<body>\n" + legend_html + panel_html, 1)
    content = content.replace("</body>", inject_js + "\n</body>", 1)
    html_path.write_text(content, encoding="utf-8")
