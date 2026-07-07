import os
import sys
import re
import json
import time
import queue as _queue
import threading
from collections import OrderedDict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, Response, stream_with_context
from flask_cors import CORS

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda  # noqa: F401 (kept for rebuild compatibility)
from langchain_core.embeddings import Embeddings

load_dotenv()

from structured_qa import init_qa, try_structured_answer

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("SECRET_KEY", "usr-web-fixed-key-tmuusrai-2024")
app.config["PERMANENT_SESSION_LIFETIME"] = __import__("datetime").timedelta(days=1)

# ── 設定 ──────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
SITE_USERNAME   = os.getenv("SITE_USERNAME", "")
SITE_PASSWORD   = os.getenv("SITE_PASSWORD", "")
VOYAGE_API_KEY  = os.getenv("VOYAGE_API_KEY")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K           = int(os.getenv("TOP_K_RESULTS", 15))

PDF_DIR         = Path("pdfs")
EXTRA_DIR       = Path("extra_docs")
MD_DIR          = Path("114md")
LLM_WIKI_DIR    = Path("llm_wiki_data")
QA_DIR          = Path("qa_data")
INDEX_DIR       = Path("faiss_index")

MD_DIR_113      = Path("113md")
INDEX_DIR_113   = Path("faiss_index_113")

# ── Embedding 模型（全域共用，避免重複初始化）──────────
class _CachedEmbeddings(Embeddings):
    """Query embedding 快取，相同問題不重複呼叫 Voyage AI。上限 200 筆（LRU）。"""
    _MAX = 200

    def __init__(self, base):
        self._base  = base
        self._cache = OrderedDict()

    def embed_query(self, text: str) -> list:
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        vec = self._base.embed_query(text)
        self._cache[text] = vec
        if len(self._cache) > self._MAX:
            self._cache.popitem(last=False)
        return vec

    def embed_documents(self, texts: list) -> list:
        return self._base.embed_documents(texts)

    def __getattr__(self, name):
        return getattr(self._base, name)

_base_embeddings = VoyageAIEmbeddings(
    voyage_api_key=VOYAGE_API_KEY,
    model="voyage-4-large",
)
embeddings = _CachedEmbeddings(_base_embeddings)

# ── RAG Prompt ────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""你是一位熟悉大學 USR（University Social Responsibility）社會責任計畫的專業助理。
請根據以下從計畫書中擷取的內容來回答問題。

【計畫書內容】
{context}

【問題】
{question}

【回答規則】
- 只根據上方提供的計畫書內容回答，不要自行推測或補充計畫書未提及的內容。
- 若計畫書內容不足以回答問題，請誠實說明「計畫書中未找到相關資訊」。
- 回答請使用繁體中文，條理清晰，內容完整，不要省略計畫書中的重要細節。
- 凡提及計畫，格式必須為「學校全名：計畫全名」，例：國立成功大學：城鄉相伴健康永續生活，不得省略學校名稱或只寫縮寫。
- 每次回答開頭，先以一句話說明共有幾件符合條件的計畫（例：「共X件計畫符合條件。」），再逐一列舉。若問題非列舉型，則跳過此步驟。
- 提及地名（縣市、鄉鎮、村里、社區、山川、場域等）或人名時，一律用〔〕標記，例：〔臺南市〕、〔永康區〕、〔萬年溪〕、〔陳明仁〕。學校名稱與計畫名稱不需標記。

回答：""",
)

REVIEWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""你是一位 USR（大學社會責任）計畫的書面審查及實地訪評委員。
請根據以下計畫書內容，提供結構化的審查分析，協助委員進行評核。

【計畫書內容】
{context}

【審查對象／問題】
{question}

【回答規則】
- 只根據上方提供的計畫書內容回答，不自行推測或補充計畫書未提及的內容。
- 回答請使用繁體中文。
- 凡提及計畫，格式必須為「學校全名：計畫全名」。
- 提及地名或人名時，一律用〔〕標記。

請依以下結構輸出：

## 一、執行現況摘要
簡述計畫的核心目標、主要執行內容與目前進度。

## 二、量化指標
列出所有可量化的數據（參與人次、場次、合作單位數、學生人數、經費執行率等）。
若有同類型計畫的比較資料，請標示相對表現（高於／低於同儕）。

## 三、質性亮點與待釐清事項
- **亮點**：值得肯定的創新作法或特色
- **待釐清**：計畫書中不夠具體或需進一步說明的地方

## 四、同儕比較（同類型計畫）
根據提供的同類型計畫參考資料，說明此計畫與同儕的異同與相對表現。
若無比較資料，請直接說明「未提供同類型計畫比較資料」。

## 五、建議訪評問題（3～5題）
根據計畫內容，提出具體、有深度的訪評問題，幫助委員深入了解計畫執行品質與影響力。
問題應針對此計畫的具體細節，而非泛用問題。

回答：""",
)

REVIEWER_PLAN_PROMPT = """你是 USR 計畫書評審委員，正在準備審查特定學校的計畫書。
請針對以下審查問題，列出 3~4 個繁體中文搜尋關鍵字組合，用來從計畫書資料庫中找到：
1. 目標學校的計畫執行詳情與成效
2. 同類型計畫（其他學校）作為同儕比較基準
3. 量化指標與成果數據

問題：{question}

只輸出 JSON 陣列，不要其他文字，例如：
["中正大學 USR 計畫執行", "深耕型計畫 量化指標 成效", "社區參與 大學 影響力", "同類型計畫 比較"]"""

REVIEWER_ANSWER_ONSITE = """你是 USR（大學社會責任）計畫的實地訪評委員。
請根據以下計畫書資料，提供「實地訪評」的專業審查協助。

【計畫書資料】
{context}

【審查問題】
{question}

【輸出規則】
- 只根據提供的資料回答，不自行推測或捏造數據
- 使用繁體中文，提及計畫格式為「學校全名：計畫全名」
- 地名人名使用〔〕標記

## 一、計畫執行現況摘要
簡述核心目標、主要執行內容、目前進度（2~3句）。

## 二、量化指標核查表
列出所有可核實的量化數據，與同儕比較：
| 量化指標 | 本計畫數值 | 同儕參考 | 相對表現 |
|---------|-----------|---------|---------|
（涵蓋：參與人次/場次、服務社區數、學生參與數、合作單位數、經費執行率等。若無比較數據標示「待查核」）

## 三、實地訪評查核清單
- ✅ **可現場驗證的具體成果**（設施、場域、社區、產出物）
- 📋 **建議調閱的文件紀錄**（活動照片、簽到表、合作合約、成果報告）
- 🏫 **建議訪視的現場與訪談對象**（社區、學生、合作夥伴）

## 四、實地訪評關鍵問題（5~7題）
**量化核實：**
1.
2.
**質性探究：**
3.
4.
**永續性與影響力：**
5.

## 五、質性品質評估
- **創新亮點**：值得肯定的特色作法
- **待現場確認**：計畫書描述需進一步核實的事項
- **潛在風險點**：執行面可能的問題

## 六、同儕比較參考
根據同類型計畫資料，說明本計畫相對表現與特色差異。若無同儕資料，說明「未找到足夠同類型計畫比較資料」。

回答："""

REVIEWER_ANSWER_WRITTEN = """你是 USR（大學社會責任）計畫的書面評審委員。
請根據以下計畫書資料，提供「書面審查」的專業評核協助。

【計畫書資料】
{context}

【審查問題】
{question}

【輸出規則】
- 只根據提供的資料回答，不自行推測或捏造數據
- 使用繁體中文，提及計畫格式為「學校全名：計畫全名」
- 地名人名使用〔〕標記

## 一、計畫基本資訊
學校、計畫類型（萌芽/深耕/國際合作/特色永續型）、核心主題、執行場域。

## 二、量化指標比較分析
列出所有量化數據，與同類型計畫進行比較：
| 量化指標 | 本計畫 | 同儕平均/範圍 | 相對表現 |
|---------|--------|-------------|---------|
（涵蓋：人次、場次、社區數、學生數、合作單位數、經費執行率等。若無比較數據標示「待查核」）

## 三、書面審查查核清單
- [ ] 計畫目標明確且可衡量（SMART原則）
- [ ] 執行策略與目標邏輯連貫
- [ ] 量化指標設定合理且具挑戰性
- [ ] 社區需求調查與問題診斷清楚
- [ ] 跨域合作機制具體說明
- [ ] 永續發展與退場機制規劃完整
- [ ] 經費配置與執行項目對應合理
- [ ] SDG對應說明清楚

## 四、書面評核關鍵問題（5~7題）
**量化與成效：**
1.
2.
**質性與品質：**
3.
4.
**可行性與邏輯：**
5.

## 五、質性品質評估
- **論述完整性**：各章節邏輯性與一致性
- **創新性**：特色作法與差異化程度
- **可行性**：目標、策略、資源的匹配程度

## 六、同儕表現比較
與同類型計畫相比，本計畫的相對優勢與不足之處。若無同儕資料，說明「未找到足夠同類型計畫比較資料」。

## 七、書面評審重點觀察
最值得注意的 2~3 個發現（優點或需改進之處）。

回答："""

APPLICANT_ANSWER_PROMPT = """你是 USR（大學社會責任）計畫的申請輔導助理，協助大學了解其他學校的計畫執行狀況，作為規劃與改善的參考。
請根據以下計畫書資料，提供結構化的執行情況分析。

【計畫書資料】
{context}

【問題】
{question}

【輸出規則】
- 只根據提供的資料回答，不自行推測或補充計畫書未提及的內容
- 使用繁體中文，提及計畫格式為「學校全名：計畫全名」
- 地名人名使用〔〕標記

## 一、執行現況概覽
簡述相關計畫的核心目標與主要執行方向（2~3句）。

## 二、特色作法與亮點
列出各計畫的創新做法或特色策略，可供參考借鑑：
- **學校：計畫** — 特色說明

## 三、量化成果參考
列出具體的量化數據，作為目標設定參考：
| 學校 | 指標 | 數值 |
|------|------|------|
（涵蓋：參與人次、場次、合作社區數、學生人數、合作單位等）

## 四、執行步驟與 Process
歸納相關計畫的執行流程與策略，說明「怎麼做」：
1. （前期：社區需求調查、場域盤點）
2. （執行：跨域合作方式、課程/活動設計）
3. （深化：成果評估、永續機制）

## 五、可參考的類似計畫
列出 2~3 個執行方向相近、值得參考的計畫，說明可借鑑之處。

回答："""


_PLAN_TYPE_RE = re.compile(
    r'(大學特色類(?:萌芽型|深耕型)|永續發展類(?:國際合作型|特色永續型))'
)

_THEME_RULES = [
    (re.compile(r'海洋|漁|里海|海岸|海鄉|蔚藍|水產|鯤鯓|澎湖|金門|石滬|濱海'),
     {'g': 'linear-gradient(135deg,#0369a1,#38bdf8)', 'e': '🌊', 'kw': 'ocean,fishing,coastal,sea'}),
    (re.compile(r'農業|農村|食農|農創|農產|農耕|稻田|稻米|有機農|里山|茶葉|咖啡|花卉|柿|竹'),
     {'g': 'linear-gradient(135deg,#15803d,#86efac)', 'e': '🌾', 'kw': 'farming,rice,harvest,agriculture'}),
    (re.compile(r'原住民|部落|泰雅|鄒族|賽夏|馬卡道|原鄉|霧台|南島|布農'),
     {'g': 'linear-gradient(135deg,#7c3aed,#c4b5fd)', 'e': '🏔️', 'kw': 'indigenous,tribe,mountain,traditional'}),
    (re.compile(r'山|森林|生態|綠色|植物|水尾|淺山|茶山|玉山|阿里山|烏來|太魯閣|花蓮|台東|宜蘭|南投'),
     {'g': 'linear-gradient(135deg,#166534,#4ade80)', 'e': '🌿', 'kw': 'forest,nature,ecology,mountain'}),
    (re.compile(r'溪|河|水圳|水域|水資源|頭前溪|曾文溪|埤圳'),
     {'g': 'linear-gradient(135deg,#0891b2,#67e8f9)', 'e': '💧', 'kw': 'river,stream,water,canal'}),
    (re.compile(r'高齡|銀髮|樂齡|長照|失智|照護|養老|長者|超高齡'),
     {'g': 'linear-gradient(135deg,#9333ea,#e879f9)', 'e': '❤️', 'kw': 'elderly,senior,care,aging'}),
    (re.compile(r'偏鄉|學伴|學童|兒童|青少年|早療'),
     {'g': 'linear-gradient(135deg,#ea580c,#fb923c)', 'e': '📚', 'kw': 'education,children,school,learning'}),
    (re.compile(r'醫療|健康|護理|醫學|健促'),
     {'g': 'linear-gradient(135deg,#dc2626,#f87171)', 'e': '🏥', 'kw': 'healthcare,medical,hospital,wellness'}),
    (re.compile(r'AI|數位|科技|智慧|IoT|XR|機器人|大數據'),
     {'g': 'linear-gradient(135deg,#4338ca,#818cf8)', 'e': '💻', 'kw': 'technology,innovation,digital,computer'}),
    (re.compile(r'文化|藝術|創生|工藝|傳統|文創|博物館|影像|陶瓷'),
     {'g': 'linear-gradient(135deg,#b45309,#fcd34d)', 'e': '🎨', 'kw': 'art,culture,craft,museum'}),
    (re.compile(r'觀光|旅遊|旅行|慢城|遊程'),
     {'g': 'linear-gradient(135deg,#0e7490,#22d3ee)', 'e': '🗺️', 'kw': 'travel,tourism,scenic,landscape'}),
    (re.compile(r'食品|餐飲|飲食|食安|料理|惜食'),
     {'g': 'linear-gradient(135deg,#c2410c,#fb923c)', 'e': '🍽️', 'kw': 'food,cooking,restaurant,cuisine'}),
    (re.compile(r'低碳|淨零|碳排|減碳|再生能源|綠電|循環經濟'),
     {'g': 'linear-gradient(135deg,#065f46,#34d399)', 'e': '♻️', 'kw': 'sustainability,solar,renewable,green'}),
    (re.compile(r'都市|城市|城鄉|社區|街道|老街|夜市'),
     {'g': 'linear-gradient(135deg,#334155,#94a3b8)', 'e': '🏙️', 'kw': 'city,urban,community,street'}),
    (re.compile(r'新住民|多元|共融|移民|越南'),
     {'g': 'linear-gradient(135deg,#6d28d9,#a78bfa)', 'e': '🤝', 'kw': 'diversity,multicultural,people,community'}),
    (re.compile(r'身心障礙|漸凍|自閉症'),
     {'g': 'linear-gradient(135deg,#1d4ed8,#93c5fd)', 'e': '💙', 'kw': 'disability,inclusion,support,care'}),
    (re.compile(r'動物|流浪動物|貓|狗'),
     {'g': 'linear-gradient(135deg,#92400e,#d97706)', 'e': '🐾', 'kw': 'animals,pets,cats,dogs'}),
    (re.compile(r'島|離島|小琉球'),
     {'g': 'linear-gradient(135deg,#0c4a6e,#38bdf8)', 'e': '🏝️', 'kw': 'island,beach,tropical,ocean'}),
]
_THEME_DEFAULT = {'g': 'linear-gradient(135deg,#1e40af,#3b82f6)', 'e': '🎓', 'kw': 'university,campus,taiwan'}

def _get_theme(title: str) -> dict:
    for pattern, theme in _THEME_RULES:
        if pattern.search(title):
            return theme
    return _THEME_DEFAULT

# 清理計畫編號與 _formatted 後綴
_PLAN_CODE_RE = re.compile(r'\s*\(114USR-[^)]*\)|_formatted', re.IGNORECASE)

def _clean_plan_code(text: str) -> str:
    """移除 chunk 內容或檔名中的計畫編號與 _formatted。"""
    return _PLAN_CODE_RE.sub('', text).strip()


def _extract_plan_type(content: str) -> str:
    """從 md 內容前 600 字提取計畫類型（萌芽型/深耕型/國際合作型/特色永續型）。"""
    m = _PLAN_TYPE_RE.search(content[:600])
    return m.group(1) if m else ""


def _inject_school_label(content: str, school: str, project: str) -> str:
    """在 Markdown heading（# 到 #######）後面插入（學校　計畫）標籤。"""
    tag = f"（{school}　{project}）"
    lines = []
    for line in content.split("\n"):
        if re.match(r"^#{1,7}\s", line) and tag not in line:
            line = line.rstrip() + tag
        lines.append(line)
    return "\n".join(lines)


def load_or_build_index(year: str = "114") -> FAISS:
    """載入既有索引；若不存在則從 md/ 重新建立。"""
    md_dir    = MD_DIR    if year == "114" else MD_DIR_113
    index_dir = INDEX_DIR if year == "114" else INDEX_DIR_113
    index_file = index_dir / "index.faiss"

    if index_file.exists():
        print(f"[INDEX] 載入既有 FAISS 索引（{year}年）...")
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print(f"[INDEX] 未找到索引（{year}年），開始建立...")
    pdf_files = list(PDF_DIR.rglob("*.pdf")) if (year == "114" and PDF_DIR.exists()) else []
    # 問答索引使用 md_dir/ 原始計畫書；llm_wiki_data/ 僅供知識地圖用
    md_files = list(md_dir.glob("*.md")) if md_dir.exists() else []
    print(f"[INDEX] 使用 {md_dir}/ 原始版（{len(md_files)} 份）")
    overview  = QA_DIR / "計劃總覽.txt"
    if not pdf_files and not md_files and not (year == "114" and overview.exists()):
        raise FileNotFoundError(f"{md_dir}/、qa_data/計劃總覽.txt 都找不到，請先放入 {year} 年度計畫書。")

    docs = []
    for pdf_path in pdf_files:
        print(f"  讀取：{pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    if year == "114" and EXTRA_DIR.exists():
        for txt_path in EXTRA_DIR.rglob("*.txt"):
            print(f"  讀取補充文件：{txt_path.name}")
            loader = TextLoader(str(txt_path), encoding="utf-8")
            docs.extend(loader.load())

    if year == "114" and overview.exists():
        print(f"  讀取：{overview.name}")
        for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
            try:
                loader = TextLoader(str(overview), encoding=enc)
                docs.extend(loader.load())
                break
            except Exception:
                continue

    md_plan_types: dict[str, str] = {}  # md_path str -> 計畫類型

    if md_files:
        for md_path in md_files:
            try:
                loader = TextLoader(str(md_path), encoding="utf-8")
                md_docs = loader.load()
                # 從首段內容提取計畫類型
                plan_type = _extract_plan_type(md_docs[0].page_content) if md_docs else ""
                md_plan_types[str(md_path)] = plan_type
                # 從檔名提取學校與計畫名稱，注入至每個 heading 後
                m_name = re.match(r'^(.+?)_(.+?)(?:\([^)]+\))*$', md_path.stem)
                if m_name:
                    school  = m_name.group(1).strip()
                    project = m_name.group(2).strip()
                    for doc in md_docs:
                        doc.page_content = _inject_school_label(doc.page_content, school, project)
                docs.extend(md_docs)
            except Exception as e:
                print(f"  [WARN] 跳過 {md_path.name}：{e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # 每個 chunk 開頭加學校＋計畫＋計畫類型標籤（確保跨 heading 切割的 chunk 也有來源脈絡）
    for chunk in chunks:
        src = Path(chunk.metadata.get("source", ""))
        if src.suffix.lower() == ".md":
            m_name = re.match(r'^(.+?)_(.+?)(?:\([^)]+\))*$', src.stem)
            if m_name:
                school     = m_name.group(1).strip()
                project    = m_name.group(2).strip()
                plan_type  = md_plan_types.get(str(src), "")
                type_str   = f"（{plan_type}）" if plan_type else ""
                tag        = f"【{school}　{project}{type_str}】"
                if f"【{school}" not in chunk.page_content[:80]:
                    chunk.page_content = f"{tag}\n{chunk.page_content}"
    total = len(chunks)
    print(f"[INDEX] 共切出 {total} 個段落，開始向量化（{year}年）...", flush=True)

    BATCH = 128
    vectorstore = None
    for i in range(0, total, BATCH):
        batch = chunks[i : i + BATCH]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        done = min(i + BATCH, total)
        print(f"  [{done}/{total}] {done*100//total}% 完成", flush=True)

    index_dir.mkdir(exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"[INDEX] {year} 年索引建立完成並已儲存。", flush=True)
    return vectorstore


# ── 對話記憶 ─────────────────────────────────────────
_chat_history: dict[str, list] = {}   # chat_id -> [{q, a}]
_MAX_HISTORY  = 5                      # 每個 session 保留最近幾輪

_FOLLOWUP_RE = re.compile(
    r'此計畫|這個計畫|這計畫|該計畫|這所學校|這間學校|該校|這所|此所'
    r'|它的|其計畫|上述|前述|剛才說|這題|這個問題|繼續|再問|另外'
    r'|它|這間|那個|那所|那間|那個計畫'
    r'|再給|再說|再介紹|再提供|再詳細|更詳細|更多|進一步|詳細說明|展開說'
)

_FULL_PLAN_RE = re.compile(
    r'完整內容|完整計畫|計畫完整|整個計畫|計畫書內容|整體計畫|計畫總概|計畫概要'
    r'|執行計畫.*完整|完整.*執行|計畫全文|計畫全貌|計畫摘要|總攬內容'
    r'|計畫成果(?!.*特定)|執行成果(?!.*特定)|成果報告|整體內容|完整說明'
    r'|整體介紹|全面介紹|詳細介紹|完整介紹|有哪些USR計畫.*介紹|整體計畫概要'
)


def _read_plan_summary(school: str, year: str = "114") -> str | None:
    """讀取 md 檔案中的計畫摘要段落，直接回傳結構化文字（不走 FAISS）。"""
    md_dir = MD_DIR if year == "114" else MD_DIR_113
    matched = []
    for f in sorted(md_dir.glob("*.md")):
        stem_school = f.stem.split("_")[0].strip()
        if school in stem_school or stem_school in school:
            matched.append(f)
    if not matched:
        return None

    blocks = []
    for md_path in matched:
        text = None
        for enc in ("utf-8-sig", "utf-8", "cp950"):
            try:
                text = md_path.read_text(encoding=enc)
                break
            except Exception:
                continue
        if not text:
            continue

        # 擷取計畫名稱
        m_name = re.search(r'計畫名稱[　\s]+(.+)', text)
        plan_name = m_name.group(1).strip() if m_name else md_path.stem

        # 擷取基本資料（SDGs、類別、場域）
        header_lines = []
        for pat in (r'核定類別[　\s]+(.+)', r'SDGs關聯議題[　\s]+(.+)', r'計畫實踐場域[　\s]+(.+)'):
            m2 = re.search(pat, text)
            if m2:
                header_lines.append(m2.group(0).strip())

        # 擷取摘要段落（計畫摘要 → 目錄）
        lines = text.split('\n')
        start = next((i for i, l in enumerate(lines) if '計畫摘要' in l), -1)
        end = next((i for i, l in enumerate(lines) if '目錄' in l and i > (start if start != -1 else 0)), len(lines))
        summary_text = '\n'.join(lines[start:end]).strip() if start != -1 else ''

        block = f"## {school}：{_clean_plan_code(plan_name)}\n"
        if header_lines:
            block += '\n'.join(header_lines) + '\n'
        if summary_text:
            block += '\n' + summary_text
        blocks.append(block)

    return '\n\n---\n\n'.join(blocks) if blocks else None


def _rewrite_question(question: str, history: list) -> str:
    """
    用 LLM 兩步驟判斷追問意圖並改寫為完整獨立問題。
    前驗：舊問題 × 新問題 初步判斷是否為追問
    後驗：舊問題 × 舊回答 × 新問題 確認指涉對象並改寫
    非追問時直接回傳原問題。
    """
    if not history:
        return question
    last = history[-1]
    prompt = (
        "你是對話理解專家。判斷「新問題」是否為追問，並輸出最終搜尋用問題。\n\n"
        f"【前一輪問題】\n{last['q']}\n\n"
        f"【前一輪回答摘要】\n{last['a'][:5000]}\n\n"
        f"【新問題】\n{question}\n\n"
        "判斷步驟：\n"
        "步驟一（前驗）：僅看「前一輪問題」與「新問題」——新問題是否在延伸前一輪的主題、"
        "或含有代名詞／省略指涉而無法獨立理解？\n"
        "步驟二（後驗）：再看「前一輪回答」——回答所提及的具體對象，"
        "是否讓新問題的指涉更明確？\n\n"
        "輸出規則：\n"
        "・若是追問：將代名詞與隱含指涉替換為具體名稱，輸出完整獨立問題\n"
        "・若非追問（全新主題）：原樣輸出新問題，不作任何修改\n\n"
        "只輸出最終問題，不要解釋，不要加引號。"
    )
    try:
        from langchain_core.messages import HumanMessage as _HM
        res = llm.bind(temperature=0).invoke([_HM(content=prompt)])
        rewritten = res.content.strip()
        if rewritten:
            # 若前一輪有明確學校名，確保改寫後仍保留
            prev_school = _extract_school(last['q'])
            if prev_school and prev_school not in rewritten:
                rewritten = f"{prev_school}：{rewritten}"
            print(f"[REWRITE] {question!r} → {rewritten!r}")
            return rewritten
    except Exception as e:
        print(f"[REWRITE] 失敗：{e}")
    return question


# ── 啟動時初始化 ──────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=os.getenv("LLM_MODEL", "gemini-2.5-pro-preview"),
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

vectorstores: dict = {}
retriever = None
for _yr in ("114", "113"):
    try:
        vs = load_or_build_index(_yr)
        vectorstores[_yr] = vs
        if _yr == "114":
            retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
        print(f"[APP] RAG {_yr} 就緒。")
    except FileNotFoundError as e:
        vectorstores[_yr] = None
        print(f"[APP] 警告 {_yr}：{e}")

vectorstore = vectorstores.get("114")  # backwards-compat alias

init_qa()


# ── 路由 ──────────────────────────────────────────────
def _load_plans(year: str = "114"):
    md_dir = MD_DIR if year == "114" else MD_DIR_113
    plans = []
    for i, f in enumerate(sorted(md_dir.glob("*.md"))):
        name = re.sub(r'\([^)]*\)', '', f.stem).replace('_formatted', '').strip('_').strip()
        parts = name.split('_', 1)
        if len(parts) == 2:
            school = parts[0].strip()
            title  = parts[1].strip()
            plans.append({"school": school, "title": title,
                          "theme": _get_theme(title), "idx": i})
    return plans

@app.route("/")
def index():
    authenticated = not SITE_PASSWORD or session.get("authenticated", False)
    return render_template("index.html", authenticated=authenticated,
        plans_114=_load_plans("114"), plans_113=_load_plans("113"))


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username_ok = not SITE_USERNAME or data.get("username") == SITE_USERNAME
    password_ok = not SITE_PASSWORD or data.get("password") == SITE_PASSWORD
    if username_ok and password_ok:
        session.permanent = True
        session["authenticated"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "帳號或密碼錯誤，請再試一次。"}), 401


@app.route("/logout")
def logout():
    session.pop("authenticated", None)
    return redirect(url_for("index"))


@app.route("/ask", methods=["POST"])
def ask():
    if SITE_PASSWORD and not session.get("authenticated"):
        return jsonify({"error": "請先登入。"}), 401

    if retriever is None:
        return jsonify({"error": "索引尚未建立，請先將 PDF 放入 pdfs/ 資料夾後重啟伺服器。"}), 503

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    chat_id   = (data.get("chat_id") or "").strip()
    user_type = (data.get("user_type") or "applicant").strip()
    if user_type not in ("applicant", "reviewer"):
        user_type = "applicant"
    original_question = (data.get("original_question") or "").strip()
    skip_eval = bool(original_question)
    if original_question:
        question = f"{original_question}（請依以下標準評估：{question}）"
    year = (data.get("year") or "114").strip()
    if year not in ("113", "114"):
        year = "114"
    vs = vectorstores.get(year) or vectorstores.get("114")

    if not question:
        return jsonify({"error": "請輸入問題。"}), 400
    if len(question) > 500:
        return jsonify({"error": "問題不得超過 500 字。"}), 400

    def generate():
        try:
            t0 = time.perf_counter()

            # ── 對話記憶：取得 history，必要時改寫問題 ──
            history = _chat_history.get(chat_id, []) if chat_id else []
            search_question = question
            if chat_id and history:
                search_question = _rewrite_question(question, history)

            # ✦ 主觀評量問題攔截：反問使用者定義判斷準則
            if not skip_eval and _is_evaluation_question(question):
                yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': _CLARIFY_MSG}, ensure_ascii=False)}\n\n"
                total_ms = round((time.perf_counter() - t0) * 1000)
                yield f"data: {json.dumps({'type': 'done', 'timing': {'total_ms': total_ms}, 'mode': 'clarify', 'original_question': question}, ensure_ascii=False)}\n\n"
                return

            # ① 結構化 QA 優先（列舉型問題直接回傳，不走 FAISS / LLM）
            structured_ctx = try_structured_answer(question, year=year)
            if structured_ctx:
                yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': structured_ctx}, ensure_ascii=False)}\n\n"
                total_ms = round((time.perf_counter() - t0) * 1000)
                yield f"data: {json.dumps({'type': 'done', 'timing': {'total_ms': total_ms}, 'mode': 'structured'})}\n\n"
                suggestions = _generate_suggestions(question, structured_ctx)
                if suggestions:
                    yield f"data: {json.dumps({'type': 'suggested_questions', 'questions': suggestions}, ensure_ascii=False)}\n\n"
                return

            # ① 完整計畫摘要攔截：直接讀 md 檔，不走 FAISS
            _school_for_full = _extract_school(search_question) or (
                _extract_school(history[-1]['q']) if history else None
            )
            if _FULL_PLAN_RE.search(question) and _school_for_full:
                full_md = _read_plan_summary(_school_for_full, year)
                if full_md:
                    prompt_val = RAG_PROMPT.invoke({"context": full_md, "question": question})
                    yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
                    answer_parts: list[str] = []
                    for chunk in llm.stream(prompt_val.to_messages()):
                        piece = getattr(chunk, 'content', '') or ''
                        if piece:
                            answer_parts.append(piece)
                            yield f"data: {json.dumps({'type': 'chunk', 'text': piece}, ensure_ascii=False)}\n\n"
                    total_ms = round((time.perf_counter() - t0) * 1000)
                    yield f"data: {json.dumps({'type': 'done', 'timing': {'total_ms': total_ms}, 'mode': 'full_plan'})}\n\n"
                    full_ans = "".join(answer_parts)
                    if chat_id:
                        hist = _chat_history.setdefault(chat_id, [])
                        hist.append({"q": question, "a": full_ans[:5000]})
                        if len(hist) > _MAX_HISTORY:
                            hist.pop(0)
                    suggestions = _generate_suggestions(question, full_ans)
                    if suggestions:
                        yield f"data: {json.dumps({'type': 'suggested_questions', 'questions': suggestions}, ensure_ascii=False)}\n\n"
                    return

            # ② Voyage AI：將問題向量化（未提早 return 時執行）
            query_vec = embeddings.embed_query(search_question)
            t_voyage = time.perf_counter()

            # ③ FAISS：第一輪搜尋
            _school    = _extract_school(search_question)
            # 追問且改寫後仍無學校名稱：從歷史記錄補回（不限關鍵字）
            if not _school and history:
                _school = _extract_school(history[-1]['q'])
                if _school:
                    print(f"[ASK] 從歷史補充學校：{_school}")
            _list      = bool(_LIST_INTENT_RE.search(search_question)) and not _school
            _personnel = bool(_PERSONNEL_RE.search(search_question))
            # 人員查詢需要跨多校 chunk，拉高 fetch 量
            _fetch  = TOP_K * 5 if _school else (TOP_K * 4 if _personnel else (TOP_K * 3 if _list else TOP_K))
            docs1 = vs.similarity_search_by_vector(query_vec, k=_fetch)

            # ④ FAISS：第二輪（關鍵詞搜尋，補充第一輪漏掉的 chunk）
            _kw = _extract_keywords(search_question)
            if _kw:
                kw_vec = embeddings.embed_query(_kw)
                docs2  = vs.similarity_search_by_vector(kw_vec, k=TOP_K)
                docs_all = _merge_docs(docs1, docs2)
                print(f"[ASK] 第二輪關鍵詞「{_kw}」→ 合併後 {len(docs_all)} 筆")
            else:
                docs_all = docs1

            # ⑤ 人員查詢：額外以角色關鍵詞再搜一輪（抓「為X人員/X主任」等）
            if _personnel:
                _role = _extract_role_term(question)
                if _role:
                    role_vec = embeddings.embed_query(_role)
                    docs3 = vs.similarity_search_by_vector(role_vec, k=TOP_K * 2)
                    docs_all = _merge_docs(docs_all, docs3)
                    print(f"[ASK] 人員角色輪「{_role}」→ 合併後 {len(docs_all)} 筆")

            if _school:
                # 學校特定查詢：去掉學校名稱後以主題詞再搜一輪
                _topic = question.replace(_school, "").strip()
                _topic = re.sub(r'[的跟與和相關有關請問]+', ' ', _topic).strip()
                if _topic:
                    topic_vec = embeddings.embed_query(_topic)
                    docs_topic = vs.similarity_search_by_vector(topic_vec, k=TOP_K * 3)
                    docs_all = _merge_docs(docs_all, docs_topic)
                    print(f"[ASK] 學校主題輪「{_topic}」→ 合併後 {len(docs_all)} 筆")
                # 額外用學校名稱搜尋，確保該校敘述型 chunk 也被納入
                school_vec = embeddings.embed_query(_school)
                docs_school = vs.similarity_search_by_vector(school_vec, k=TOP_K * 10)
                docs_all = _merge_docs(docs_all, docs_school)
                print(f"[ASK] 學校名稱輪「{_school}」→ 合併後 {len(docs_all)} 筆")
                # 單一學校查詢：取所有該校 chunk，不截斷（一份計畫書約 40 chunk）
                docs = _school_filter_docs(docs_all, _school, k=9999)
                print(f"[ASK] 學校過濾「{_school}」→ {len(docs)} 筆")
            else:
                docs = docs_all[:TOP_K]
            t_faiss = time.perf_counter()

            context = "\n\n".join(_clean_plan_code(doc.page_content) for doc in docs)

            # ── 委員模式：同儕比較（同類型計畫 2~3 所其他學校）──
            if user_type == "reviewer" and _school:
                plan_type = None
                for doc in docs[:10]:
                    m = _PLAN_TYPE_RE.search(doc.page_content)
                    if m:
                        plan_type = m.group(1)
                        break
                if plan_type:
                    peer_vec = embeddings.embed_query(plan_type)
                    peer_raw = vs.similarity_search_by_vector(peer_vec, k=TOP_K * 5)
                    peer_docs = [d for d in peer_raw
                                 if _school not in d.metadata.get("source", "")]
                    peer_docs = _dedup_by_school(peer_docs, k=3)
                    print(f"[ASK] 委員同儕「{plan_type}」→ {len(peer_docs)} 所學校")
                    if peer_docs:
                        peer_ctx = "\n\n".join(_clean_plan_code(d.page_content) for d in peer_docs)
                        context = f"{context}\n\n【同類型計畫參考（{plan_type}）】\n{peer_ctx}"

            prompt_value = (REVIEWER_PROMPT if user_type == "reviewer" else RAG_PROMPT).invoke(
                {"context": context, "question": question}
            )

            sources = []
            seen = set()
            for doc in docs:
                meta = doc.metadata
                raw_src = Path(meta.get("source", "")).stem
                src  = _clean_plan_code(raw_src)
                page = meta.get("page", 0) + 1
                if (src, page) not in seen:
                    seen.add((src, page))
                    sources.append({"source": src, "page": page})

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

            # ③ LLM：串流生成
            answer_chars = 0
            answer_parts = []
            t_first_chunk = None
            t_gemini_start = time.perf_counter()
            for chunk in llm.stream(prompt_value):
                content = chunk.content
                if isinstance(content, list):
                    content = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                if content:
                    if t_first_chunk is None:
                        t_first_chunk = time.perf_counter()
                    answer_chars += len(content)
                    answer_parts.append(content)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': content}, ensure_ascii=False)}\n\n"

            t_end = time.perf_counter()
            if t_first_chunk is None:
                t_first_chunk = t_end

            prompt_chars = len(prompt_value.to_string())
            total_chars = prompt_chars + answer_chars
            print(f"[TOKEN] 輸入={prompt_chars}字元(~{prompt_chars//2}tokens) 輸出={answer_chars}字元(~{answer_chars//2}tokens) 合計~{total_chars//2}tokens")

            timing = {
                "voyage_ms":      round((t_voyage - t0) * 1000),
                "faiss_ms":       round((t_faiss - t_voyage) * 1000),
                "llm_first_ms":   round((t_first_chunk - t_faiss) * 1000),
                "llm_total_ms":   round((t_end - t_faiss) * 1000),
                "total_ms":       round((t_end - t0) * 1000),
            }
            print(
                f"[TIMING] Voyage={timing['voyage_ms']}ms"
                f" | FAISS={timing['faiss_ms']}ms"
                f" | LLM首字={timing['llm_first_ms']}ms"
                f" | LLM完成={timing['llm_total_ms']}ms"
                f" | 總計={timing['total_ms']}ms"
            )

            yield f"data: {json.dumps({'type': 'done', 'timing': timing})}\n\n"

            # ── 儲存對話記憶 ──
            if chat_id:
                full_ans = "".join(answer_parts)
                hist = _chat_history.setdefault(chat_id, [])
                hist.append({"q": question, "a": full_ans[:5000]})
                if len(hist) > _MAX_HISTORY:
                    hist.pop(0)
                if len(_chat_history) > 1000:
                    for k in list(_chat_history.keys())[:200]:
                        del _chat_history[k]

            suggestions = _generate_suggestions(question, "".join(answer_parts))
            if suggestions:
                yield f"data: {json.dumps({'type': 'suggested_questions', 'questions': suggestions}, ensure_ascii=False)}\n\n"

        except Exception as e:
            import traceback
            print(f"[ERROR] /ask stream：{e}")
            print(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'error': '查詢時發生錯誤，請稍後再試。'}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── ReAct Agent ───────────────────────────────────────
AGENT_PLAN_PROMPT = """針對以下問題，請列出 2~3 個最適合的繁體中文搜尋關鍵字，用來從 USR 計畫書資料庫中找到相關內容。

問題：{question}

只輸出 JSON 陣列，不要其他文字，例如：
["高齡照護 USR 計畫", "青銀共創 大學社會責任", "失智症 照護"]"""

AGENT_ANSWER_PROMPT = """你是 USR 計畫書研究助理，請根據以下資料回答問題。

問題：{question}

搜尋到的資料：
{context}

請用繁體中文給出完整、有條理的分析回答，整合所有資料內容。
凡提及任何學校或計畫，必須同時寫出完整的學校名稱與計畫名稱，不得只寫縮寫或簡稱。"""

SUGGEST_PROMPT = """根據以下 USR 計畫書問答，生成 3 個使用者可能想繼續追問的問題。
要求：具體、繁體中文、每題 25 字以內，圍繞原始問題的延伸或深化方向。
只輸出 JSON 陣列，不要任何其他文字。
範例：["這些計畫的具體成效如何評估？", "有哪些相關學校也在推動類似主題？", "這個領域的未來發展趨勢為何？"]

原始問題：{question}
回答摘要：{answer}"""


def _generate_suggestions(question: str, answer: str) -> list[str]:
    try:
        from langchain_core.messages import HumanMessage
        prompt = SUGGEST_PROMPT.format(question=question, answer=answer[:600])
        res = llm.bind(temperature=0.4).invoke([HumanMessage(content=prompt)])
        text = _normalize_content(res.content).strip()
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            items = json.loads(m.group())
            return [s for s in items if isinstance(s, str)][:3]
    except Exception as e:
        print(f"[SUGGEST] 生成失敗：{e}")
    return []


_PERSONNEL_RE = re.compile(
    r'人員|成員|團隊|師資|主任|執行秘書|計畫主持|共同主持|協同主持|老師|教授|誰|姓名|人名'
)

_LIST_INTENT_RE  = re.compile(
    r'哪些|有哪|哪幾|列出'
    r'|所有.{0,6}(?:學校|大學|院校|計畫|計画)'
    r'|各.{0,4}(?:學校|大學|院校|校|間)'
    r'|全部.{0,4}(?:學校|大學|院校|計畫)'
)
_QUESTION_WORDS  = {"哪些", "有哪", "哪幾", "什麼", "怎麼", "如何", "是否", "有沒有",
                    "請問", "告訴我", "介紹", "說明", "哪個", "哪裡", "為何", "為什麼",
                    "幾個", "幾間", "多少", "列出", "有關", "相關", "跟", "和", "與"}

_EVAL_RE = re.compile(
    # 「最」字系列：明確排名意圖
    r'最(?:好|棒|佳|優|優秀|值得|推薦|有效|成功|傑出|具代表|有特色|突出|厲害|強大|重要)'
    r'|哪(?:個|間|所|家|些).{0,8}(?:最|比較好|更好|較好|第一)'
    r'|(?:最好|最佳|最優|最強)的.{0,6}(?:計畫|學校|大學|做法|方案|案例)'
    r'|排名|排行|第一名|冠軍|名次|勝出|較優|更優'
    r'|誰做得比較好|哪個比較好|哪個更好|哪個較好|哪間做得好'
    # 「哪些學校有 + 主觀品質形容詞」：需要定義何謂「完整/完善/良好」
    r'|哪.{0,12}(?:完整|完善|健全|完備|齊全|周全|良好|優良|成熟|豐富|積極|全面|深入|扎實|紮實|有效|到位)(?:的|之).{0,15}(?:制度|機制|措施|政策|規劃|計畫|方案|做法|配套|支持|體系|系統)'
)

_CLARIFY_MSG = """您的問題涉及主觀的優劣評斷，不同人對「最好」的定義可能大不相同，我需要先了解您的判斷標準，才能給出有意義的比較分析。

請問您想依據哪些面向來評估？例如：

• 計畫規模：總經費、執行年期、涵蓋地區
• 社會影響力：直接受惠人數、社區問題改善程度
• 在地連結：與社區的合作深度、長期夥伴關係
• 創新程度：技術應用獨特性、跨域整合方式
• 國際能見度：跨國合作夥伴、獲獎紀錄
• 人才培育：學生實務時數、就業媒合成效

請描述您在意的條件（可以是上面任一項，也可以自行定義），我就能針對您的標準為您做出分析！"""


def _is_evaluation_question(question: str) -> bool:
    return bool(_EVAL_RE.search(question))


def _build_known_schools() -> list[str]:
    """從 114md/ 檔名提取已知學校名稱，按長度降冪排列（優先匹配較長名稱）。"""
    schools = set()
    _name_re = re.compile(r'^([一-鿿]{3,12}(?:大學|學院|科大))')
    for f in MD_DIR.glob("*.md"):
        m = _name_re.match(f.name)
        if m:
            schools.add(m.group(1))
    return sorted(schools, key=len, reverse=True)

_KNOWN_SCHOOLS: list[str] = _build_known_schools()


def _extract_school(text: str) -> str | None:
    """從問題中提取學校名稱（與已知學校清單比對，避免 regex greedy 誤抓前綴詞）。"""
    for school in _KNOWN_SCHOOLS:
        if school in text:
            return school
    return None


def _school_filter_docs(docs, school: str, k: int):
    """過濾含學校名稱的 chunk，無結果則退回全部。"""
    filtered = [d for d in docs if school in d.metadata.get("source", "")]
    return filtered[:k] if filtered else docs[:k]


def _extract_role_term(question: str) -> str | None:
    """從人員查詢中提取角色/職稱關鍵詞，例如「原住民資源中心人員」→「原住民資源中心」。"""
    # 擷取「為X人員/X主任/X中心/X職稱」等後半部分作為角色詞
    m = re.search(r'[為是]([一-鿿A-Za-z0-9]{4,20}?)(?:人員|成員|主任|秘書|教授|老師)?$', question.strip())
    if m:
        return m.group(1).strip()
    # 退回：取問題中最長的非問句 CJK 詞組
    tokens = re.findall(r'[一-鿿]{4,}', question)
    if tokens:
        return max(tokens, key=len)
    return None


def _extract_keywords(question: str) -> str | None:
    """去除疑問詞，留下實詞作為第二輪搜尋 query；若與原問題差異不大則回傳 None。"""
    q = question
    for w in sorted(_QUESTION_WORDS, key=len, reverse=True):
        q = q.replace(w, " ")
    tokens = re.findall(r'[A-Za-z0-9]+|[一-鿿]{2,}', q)
    kw = " ".join(t for t in tokens if len(t) >= 2)
    return kw.strip() if kw.strip() and kw.strip() != question.strip() else None


def _merge_docs(docs1: list, docs2: list) -> list:
    """合併兩輪結果；docs1 優先，docs2 補充未出現的 chunk。"""
    seen: set[tuple] = set()
    result = []
    for doc in docs1 + docs2:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0))
        if key not in seen:
            seen.add(key)
            result.append(doc)
    return result


def _dedup_by_school(docs, k: int):
    """每間學校只保留最相關的一個 chunk，最多取 k 間學校。"""
    seen: set[str] = set()
    result = []
    for doc in docs:
        src = Path(doc.metadata.get("source", "")).name
        school_key = src.split("_")[0]
        if school_key not in seen:
            seen.add(school_key)
            result.append(doc)
            if len(result) >= k:
                break
    return result


def tool_search_rag(query: str, k: int = 5, year: str = "114"):
    """回傳 (觀察文字, sources列表)"""
    _vs = vectorstores.get(year, vectorstores.get("114"))
    print(f"[TOOL] 搜尋（{year}年）：{query[:60]}  vectorstore={'OK' if _vs else 'None'}")
    vec = embeddings.embed_query(query)
    print(f"[TOOL] embed 完成，vec長度={len(vec)}")
    school = _extract_school(query)
    _list  = bool(_LIST_INTENT_RE.search(query)) and not school
    fetch_k = k * 5 if school else (k * 3 if _list else k)
    docs = _vs.similarity_search_by_vector(vec, k=fetch_k)
    if school:
        docs = _school_filter_docs(docs, school, k)
        print(f"[TOOL] 學校過濾「{school}」→ {len(docs)} 筆")
    elif _list:
        docs = _dedup_by_school(docs, k)
        print(f"[TOOL] 列舉去重 → {len(docs)} 間學校")
    print(f"[TOOL] FAISS 找到 {len(docs)} 筆")
    if not docs:
        return "查無相關資料", []
    results = []
    sources = []
    seen = set()
    for i, doc in enumerate(docs):
        src = Path(doc.metadata.get("source", "")).name
        page = doc.metadata.get("page", 0) + 1
        results.append(f"[{i+1}] {src} 第{page}頁\n{doc.page_content[:400]}")
        if (src, page) not in seen:
            seen.add((src, page))
            sources.append({"source": src, "page": page})
    return "\n\n---\n\n".join(results), sources


def _normalize_content(content):
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content


def react_agent_stream(question: str, max_steps: int = 5, year: str = "114"):
    """固定 2 步架構：Gemini 規劃搜尋詞 → 執行搜尋 → Gemini 整合回答。"""
    import re
    from langchain_core.messages import HumanMessage

    # ── 結構化 QA 短路：列舉型問題直接回傳，不呼叫 FAISS ──
    structured_ctx = try_structured_answer(question, year=year)
    if structured_ctx:
        yield "sources", []
        yield "answer", structured_ctx
        return

    all_sources = []
    seen_sources = set()

    # ── 步驟 1：讓 Gemini 決定要搜尋什麼 ──
    yield "heartbeat", None
    try:
        plan_res = llm.bind(temperature=0).invoke([HumanMessage(content=AGENT_PLAN_PROMPT.format(question=question))])
        plan_text = _normalize_content(plan_res.content).strip()
        print(f"[AGENT] 搜尋計畫：{plan_text}")
        match = re.search(r'\[.*?\]', plan_text, re.DOTALL)
        queries = json.loads(match.group()) if match else [question]
        if not isinstance(queries, list) or not queries:
            queries = [question]
    except Exception as e:
        print(f"[AGENT] 搜尋計畫失敗：{e}")
        queries = [question]

    # ── 步驟 2：執行搜尋 ──
    all_context = []
    for i, query in enumerate(queries[:3]):
        yield "step", {"step": i + 1, "preview": f"搜尋：{str(query)[:80]}"}
        try:
            observation, sources = tool_search_rag(str(query), k=10, year=year)
            all_context.append(f"【搜尋 {i+1}：{query}】\n{observation}")
            for s in sources:
                key = (s["source"], s["page"])
                if key not in seen_sources:
                    seen_sources.add(key)
                    all_sources.append(s)
        except Exception as e:
            import traceback
            print(f"[AGENT] 搜尋失敗 {query}：{e}\n{traceback.format_exc()}")

    yield "sources", all_sources

    # ── 步驟 3：Gemini 整合回答 ──
    yield "heartbeat", None
    try:
        context = "\n\n".join(all_context) if all_context else "查無相關資料"
        answer_res = llm.invoke([HumanMessage(content=AGENT_ANSWER_PROMPT.format(
            question=question, context=context
        ))])
        final = _normalize_content(answer_res.content).strip()
        if not final:
            final = "抱歉，無法完成分析，請重新提問。"
    except Exception as e:
        print(f"[AGENT] 整合回答失敗：{e}")
        final = f"分析過程發生錯誤，請重新提問。（{e}）"

    yield "answer", final


@app.route("/agent", methods=["POST"])
def agent_ask():
    if SITE_PASSWORD and not session.get("authenticated"):
        return jsonify({"error": "請先登入。"}), 401

    data = request.get_json(silent=True) or {}
    year = (data.get("year") or "114").strip()
    if year not in ("113", "114"):
        year = "114"
    vs = vectorstores.get(year) or vectorstores.get("114")

    if vs is None:
        return jsonify({"error": "索引尚未建立。"}), 503

    question = (data.get("question") or "").strip()
    original_question = (data.get("original_question") or "").strip()
    skip_eval = bool(original_question)
    if original_question:
        question = f"{original_question}（請依以下標準評估：{question}）"
    if not question:
        return jsonify({"error": "請輸入問題。"}), 400
    if len(question) > 500:
        return jsonify({"error": "問題不得超過 500 字。"}), 400

    def generate():
        try:
            t_agent_start = time.perf_counter()

            # ✦ 主觀評量問題攔截
            if not skip_eval and _is_evaluation_question(question):
                yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': _CLARIFY_MSG}, ensure_ascii=False)}\n\n"
                total_ms = round((time.perf_counter() - t_agent_start) * 1000)
                yield f"data: {json.dumps({'type': 'done', 'timing': {'total_ms': total_ms}, 'mode': 'clarify', 'original_question': question}, ensure_ascii=False)}\n\n"
                return

            yield f"data: {json.dumps({'type': 'status', 'text': '🤖 Agent 模式啟動，第一步：分析問題（約 5-10 秒）...'}, ensure_ascii=False)}\n\n"
            step_count = 0
            for event_type, data in react_agent_stream(question, year=year):
                if event_type == "heartbeat":
                    yield ": keepalive\n\n"  # SSE comment，保持連線不斷
                elif event_type == "step":
                    step_count += 1
                    yield f"data: {json.dumps({'type': 'step', 'step': data['step'], 'preview': data['preview']}, ensure_ascii=False)}\n\n"
                elif event_type == "sources":
                    yield f"data: {json.dumps({'type': 'sources', 'sources': data}, ensure_ascii=False)}\n\n"
                elif event_type == "answer":
                    t_agent_end = time.perf_counter()
                    total_ms = round((t_agent_end - t_agent_start) * 1000)
                    timing = {"total_ms": total_ms, "agent_steps": step_count}
                    yield f"data: {json.dumps({'type': 'chunk', 'text': data}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'timing': timing, 'steps': step_count})}\n\n"
                    suggestions = _generate_suggestions(question, data)
                    if suggestions:
                        yield f"data: {json.dumps({'type': 'suggested_questions', 'questions': suggestions}, ensure_ascii=False)}\n\n"
        except Exception as e:
            import traceback
            print(f"[ERROR] /agent：{e}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'error': '分析時發生錯誤，請稍後再試。'}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/subagent", methods=["POST"])
def subagent_ask():
    if SITE_PASSWORD and not session.get("authenticated"):
        return jsonify({"error": "請先登入。"}), 401

    data = request.get_json(silent=True) or {}
    year = (data.get("year") or "114").strip()
    if year not in ("113", "114"):
        year = "114"
    vs = vectorstores.get(year) or vectorstores.get("114")

    if vs is None:
        return jsonify({"error": "索引尚未建立。"}), 503

    question = (data.get("question") or "").strip()
    original_question = (data.get("original_question") or "").strip()
    skip_eval = bool(original_question)
    if original_question:
        question = f"{original_question}（請依以下標準評估：{question}）"
    if not question:
        return jsonify({"error": "請輸入問題。"}), 400
    if len(question) > 500:
        return jsonify({"error": "問題不得超過 500 字。"}), 400
    user_type = (data.get("user_type") or "applicant").strip()
    if user_type not in ("applicant", "reviewer"):
        user_type = "applicant"
    reviewer_subtype = (data.get("reviewer_subtype") or "written").strip()
    if reviewer_subtype not in ("onsite", "written"):
        reviewer_subtype = "written"

    def generate():
        import re
        from langchain_core.messages import HumanMessage

        try:
            t_start = time.perf_counter()

            # ✦ 主觀評量問題攔截
            if not skip_eval and _is_evaluation_question(question):
                yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': _CLARIFY_MSG}, ensure_ascii=False)}\n\n"
                total_ms = round((time.perf_counter() - t_start) * 1000)
                yield f"data: {json.dumps({'type': 'done', 'timing': {'total_ms': total_ms}, 'mode': 'clarify', 'original_question': question}, ensure_ascii=False)}\n\n"
                return

            # ── 結構化 QA 短路 ──
            structured_ctx = try_structured_answer(question, year=year)
            if structured_ctx:
                yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': structured_ctx}, ensure_ascii=False)}\n\n"
                total_ms = round((time.perf_counter() - t_start) * 1000)
                yield f"data: {json.dumps({'type': 'done', 'timing': {'total_ms': total_ms}, 'mode': 'structured'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'status', 'text': '🔍 分析問題，規劃 subagent 搜尋策略...'}, ensure_ascii=False)}\n\n"

            # ── 步驟 1：規劃搜尋詞（背景執行緒 + keepalive，應對 Gemini 思考階段）──
            plan_q = _queue.Queue()
            _plan_prompt = REVIEWER_PLAN_PROMPT if user_type == "reviewer" else AGENT_PLAN_PROMPT
            _plan_msg = HumanMessage(content=_plan_prompt.format(question=question))
            def _plan_worker():
                try:
                    res = llm.bind(temperature=0).invoke([_plan_msg])
                    plan_q.put(('done', _normalize_content(res.content).strip()))
                except Exception as exc:
                    plan_q.put(('error', str(exc)))
            threading.Thread(target=_plan_worker, daemon=True).start()

            plan_text = ""
            while True:
                try:
                    kind, payload = plan_q.get(timeout=5)
                    if kind == 'done':
                        plan_text = payload
                    break
                except _queue.Empty:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

            try:
                m = re.search(r'\[.*?\]', plan_text, re.DOTALL)
                queries = json.loads(m.group()) if m else [question]
                if not isinstance(queries, list) or not queries:
                    queries = [question]
                queries = [str(q) for q in queries[:4]]
            except Exception as e:
                print(f"[SUBAGENT] 規劃失敗：{e}")
                queries = [question]

            n = len(queries)
            yield f"data: {json.dumps({'type': 'status', 'text': f'⚡ 並行啟動 {n} 個 subagent 搜尋...'}, ensure_ascii=False)}\n\n"

            # ── 步驟 2：並行執行各 subagent ──
            from concurrent.futures import ThreadPoolExecutor, as_completed

            results_map = {}   # index -> (obs, srcs)
            t_search_start = time.perf_counter()

            def _run_subagent(idx_query):
                idx, query = idx_query
                try:
                    obs, srcs = tool_search_rag(query, k=10, year=year)
                    return idx, query, obs, srcs
                except Exception as e:
                    print(f"[SUBAGENT] subagent {idx+1} 失敗：{e}")
                    return idx, query, "查無結果", []

            with ThreadPoolExecutor(max_workers=n) as executor:
                futures = {executor.submit(_run_subagent, (i, q)): i for i, q in enumerate(queries)}
                for future in as_completed(futures):
                    idx, query, obs, srcs = future.result()
                    results_map[idx] = (query, obs, srcs)
                    yield f"data: {json.dumps({'type': 'step', 'step': idx+1, 'preview': query[:70]}, ensure_ascii=False)}\n\n"

            # 依原始順序合併結果
            all_context = []
            all_sources = []
            seen_sources = set()
            for idx in sorted(results_map):
                query, obs, srcs = results_map[idx]
                all_context.append(f"【Subagent {idx+1}：{query}】\n{obs}")
                for s in srcs:
                    key = (s["source"], s["page"])
                    if key not in seen_sources:
                        seen_sources.add(key)
                        all_sources.append(s)

            t_search_ms = round((time.perf_counter() - t_search_start) * 1000)

            # ── 委員模式：從 subagent 結果提取計畫類型，追加同儕比較 ──
            if user_type == "reviewer":
                _school_sa = _extract_school(question)
                if _school_sa:
                    _plan_type_sa = None
                    for idx in sorted(results_map):
                        _, obs, _ = results_map[idx]
                        m = _PLAN_TYPE_RE.search(obs)
                        if m:
                            _plan_type_sa = m.group(1)
                            break
                    if _plan_type_sa:
                        peer_vec = embeddings.embed_query(_plan_type_sa)
                        peer_raw = vs.similarity_search_by_vector(peer_vec, k=TOP_K * 5)
                        peer_docs = [d for d in peer_raw
                                     if _school_sa not in d.metadata.get("source", "")]
                        peer_docs = _dedup_by_school(peer_docs, k=3)
                        print(f"[SUBAGENT] 委員同儕「{_plan_type_sa}」→ {len(peer_docs)} 所學校")
                        if peer_docs:
                            peer_ctx = "\n\n".join(_clean_plan_code(d.page_content) for d in peer_docs)
                            all_context.append(f"【同類型計畫參考（{_plan_type_sa}）】\n{peer_ctx}")

            yield f"data: {json.dumps({'type': 'sources', 'sources': all_sources}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'status', 'text': '🧠 整合所有 subagent 結果，生成回答...'}, ensure_ascii=False)}\n\n"

            # ── 步驟 3：背景執行緒串流（每 5s 發 keepalive，應對 Gemini 思考階段）──
            context = "\n\n".join(all_context)
            if len(context) > 20000:
                context = context[:20000] + "\n\n...(資料已截斷)"
            if user_type == "reviewer":
                _answer_tmpl = REVIEWER_ANSWER_ONSITE if reviewer_subtype == "onsite" else REVIEWER_ANSWER_WRITTEN
            else:
                _answer_tmpl = APPLICANT_ANSWER_PROMPT
            prompt_msg = HumanMessage(content=_answer_tmpl.format(
                question=question, context=context or "查無相關資料"
            ))

            cq = _queue.Queue()

            def _stream_worker():
                try:
                    for chunk in llm.stream([prompt_msg]):
                        text = _normalize_content(chunk.content)
                        if text:
                            cq.put(('chunk', text))
                    cq.put(('done', None))
                except Exception as exc:
                    cq.put(('error', str(exc)))

            threading.Thread(target=_stream_worker, daemon=True).start()

            answer_started = False
            answer_parts_sa = []
            while True:
                try:
                    kind, payload = cq.get(timeout=5)
                    if kind == 'chunk':
                        answer_started = True
                        answer_parts_sa.append(payload)
                        yield f"data: {json.dumps({'type': 'chunk', 'text': payload}, ensure_ascii=False)}\n\n"
                    elif kind == 'done':
                        break
                    else:
                        print(f"[SUBAGENT] 串流失敗：{payload}")
                        if not answer_started:
                            yield f"data: {json.dumps({'type': 'chunk', 'text': '分析過程發生錯誤，請重新提問。'}, ensure_ascii=False)}\n\n"
                        break
                except _queue.Empty:
                    yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"  # Gemini 思考中，每 5s 保持連線

            if not answer_started:
                yield f"data: {json.dumps({'type': 'chunk', 'text': '抱歉，無法完成分析，請重新提問。'}, ensure_ascii=False)}\n\n"

            total_ms = round((time.perf_counter() - t_start) * 1000)
            yield f"data: {json.dumps({'type': 'done', 'timing': {'total_ms': total_ms, 'search_ms': t_search_ms, 'subagent_count': n}, 'mode': 'subagent'})}\n\n"
            if answer_parts_sa:
                suggestions = _generate_suggestions(question, "".join(answer_parts_sa))
                if suggestions:
                    yield f"data: {json.dumps({'type': 'suggested_questions', 'questions': suggestions}, ensure_ascii=False)}\n\n"

        except Exception as e:
            import traceback
            print(f"[ERROR] /subagent stream：{e}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'error': '查詢時發生錯誤，請稍後再試。'}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/rebuild-index", methods=["POST"])
def rebuild_index():
    """重新建立索引（上傳新 PDF 後呼叫）。"""
    global vectorstore, retriever, vectorstores

    req_data = request.get_json(silent=True) or {}
    year = req_data.get("year", "114")
    if year not in ("113", "114"):
        year = "114"

    target_index_dir = INDEX_DIR if year == "114" else INDEX_DIR_113

    # 刪除舊索引
    for f in target_index_dir.glob("*"):
        f.unlink()

    try:
        vs = load_or_build_index(year)
        vectorstores[year] = vs
        if year == "114":
            vectorstore = vs
            retriever   = vs.as_retriever(
                search_type="similarity",
                search_kwargs={"k": TOP_K},
            )
        return jsonify({"message": f"{year} 年索引重建完成。"})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"[ERROR] /rebuild-index：{e}")
        return jsonify({"error": "索引重建失敗。"}), 500


_wiki_building = False


def _build_wiki_background():
    global _wiki_building
    try:
        from wiki_map_builder import build_wiki_graph
        from wiki_map_render import render_wiki_map
        G = build_wiki_graph()
        out = Path("static/wiki_map.html")
        out.parent.mkdir(exist_ok=True)
        render_wiki_map(G, output=out)
        print(f"[WIKI] 背景建立完成：{G.number_of_nodes()} 節點，{G.number_of_edges()} 邊")
    except Exception as e:
        import traceback
        print(f"[WIKI] 背景建立失敗：{e}")
        print(traceback.format_exc())
    finally:
        _wiki_building = False


@app.route("/cards")
def cards_page():
    if SITE_PASSWORD and not session.get("authenticated", False):
        return redirect(url_for("index"))
    plans = []
    for f in sorted(MD_DIR.glob("*.md")):
        name = f.stem
        # 移除 _formatted 後綴與計畫編號括號
        name = re.sub(r'\([^)]*\)', '', name).replace('_formatted', '').strip('_').strip()
        parts = name.split('_', 1)
        if len(parts) == 2:
            plans.append({"school": parts[0].strip(), "title": parts[1].strip(), "filename": f.stem})
    return render_template("cards.html", plans=plans)


@app.route("/api/plans")
def api_plans():
    """回傳各年度計畫清單（供前端 modal 動態載入）。"""
    if SITE_PASSWORD and not session.get("authenticated", False):
        return jsonify({"error": "請先登入"}), 401
    return jsonify({
        "114": _load_plans("114"),
        "113": _load_plans("113"),
    })


@app.route("/wiki-map")
def wiki_map_page():
    global _wiki_building
    authenticated = not SITE_PASSWORD or session.get("authenticated", False)
    if SITE_PASSWORD and not authenticated:
        return redirect(url_for("index"))

    html_file = Path("static/wiki_map.html")

    if html_file.exists():
        return html_file.read_text(encoding="utf-8")

    # 自動在背景觸發建立（只啟動一次）
    if not _wiki_building:
        _wiki_building = True
        t = threading.Thread(target=_build_wiki_background, daemon=True)
        t.start()

    return """<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="UTF-8">
<title>Wiki Map 建置中</title>
<meta http-equiv="refresh" content="4;url=/wiki-map">
<style>body{font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#1a1a2e;color:#e0e0e0;flex-direction:column;gap:16px}
.spinner{width:48px;height:48px;border:5px solid #334;border-top-color:#4f86c6;border-radius:50%;animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}</style></head>
<body><div class="spinner"></div><p>知識地圖建置中，請稍候（首次約需 20–30 秒）…</p>
<p style="font-size:.8rem;color:#666">頁面將自動重新整理</p></body></html>"""


@app.route("/api/wiki-map-build", methods=["POST"])
def wiki_map_build():
    """手動觸發 Wiki Map 重建（會刪除快取後重新產生）。"""
    if SITE_PASSWORD and not session.get("authenticated"):
        return jsonify({"error": "請先登入。"}), 401
    # 刪除快取，強制重建
    cache = Path("wiki_graph_cache.json")
    if cache.exists():
        cache.unlink()
    html = Path("static/wiki_map.html")
    if html.exists():
        html.unlink()
    global _wiki_building
    if not _wiki_building:
        _wiki_building = True
        t = threading.Thread(target=_build_wiki_background, daemon=True)
        t.start()
    return jsonify({"ok": True, "message": "重建已啟動，請稍後重整 /wiki-map"})


@app.route("/warmup")
def warmup():
    """預熱 Voyage AI 連線，減少第一次問答的延遲。"""
    try:
        embeddings.embed_query("warmup")
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"ok": False})


@app.route("/status")
def status():
    """健康檢查：確認系統是否就緒。"""
    pdf_count = len(list(PDF_DIR.rglob("*.pdf"))) if PDF_DIR.exists() else 0
    index_ready = (INDEX_DIR / "index.faiss").exists()
    index_113_ready = (INDEX_DIR_113 / "index.faiss").exists()
    ready = index_ready or index_113_ready
    return jsonify({
        "ready":          ready,
        "pdf_count":      pdf_count,
        "index_ready":    index_ready,
        "index_113_ready": index_113_ready,
        "years_ready":    [y for y, v in vectorstores.items() if v is not None],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", port=port, use_reloader=True)
