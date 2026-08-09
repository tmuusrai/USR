import os
import sys
import re
import jieba
import json
import hashlib
import time
import queue as _queue
import threading
import sqlite3
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
jieba.initialize()

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, Response, stream_with_context
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

load_dotenv()

from structured_qa import init_qa, try_structured_answer

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or os.urandom(32)
app.config["PERMANENT_SESSION_LIFETIME"] = __import__("datetime").timedelta(days=1)

# ── 設定 ──────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
SITE_USERNAME   = os.getenv("SITE_USERNAME", "")
SITE_PASSWORD   = os.getenv("SITE_PASSWORD", "")

def _load_site_users() -> dict:
    raw = os.getenv("SITE_USERS_JSON", "")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    if SITE_USERNAME and SITE_PASSWORD:
        return {SITE_USERNAME: [SITE_PASSWORD]}
    return {}
SITE_USERS = _load_site_users()
ADMIN_USERS = set(u.strip() for u in os.getenv("ADMIN_USERS", "").split(",") if u.strip())

def _is_admin() -> bool:
    return bool(session.get("authenticated") and session.get("username") in ADMIN_USERS)

CONV_DB_PATH    = Path(os.getenv("CONV_DB_PATH", "conversations.db"))

def _get_db():
    conn = sqlite3.connect(str(CONV_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def _init_conv_db():
    with _get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '新對話',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conv_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS site_users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_msg_conv ON conv_messages(conversation_id, timestamp ASC);
            CREATE TABLE IF NOT EXISTS api_costs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                service TEXT NOT NULL,
                call_type TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_api_costs_ts ON api_costs(ts);
        """)
        # 每次啟動都把 SITE_USERS_JSON 裡的新帳號同步進 SQLite（不覆蓋已有的密碼）
        if SITE_USERS:
            now = time.time()
            for uname, passwords in SITE_USERS.items():
                if passwords:
                    conn.execute(
                        "INSERT OR IGNORE INTO site_users (username, password, created_at) VALUES (?,?,?)",
                        (uname, generate_password_hash(passwords[0]), now)
                    )

_init_conv_db()

# ── API 成本追蹤 ─────────────────────────────────────────
_LLM_MODEL_NAME      = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
_LLM_FAST_MODEL_NAME = os.environ.get("LLM_MODEL_FAST", _LLM_MODEL_NAME)
_VOYAGE_PRICE_PER_1M = 0.12  # voyage-4-large, USD per 1M tokens
_LLM_PRICING: dict[str, tuple[float, float]] = {
    # (USD per 1M input tokens, USD per 1M output tokens)
    "gemini-2.5-pro":   (1.25, 10.0),
    "gemini-2.5-flash": (0.15,  0.60),
    "gemini-2.0-flash": (0.075, 0.30),
    "gemini-1.5-pro":   (1.25,  5.00),
    "gemini-1.5-flash": (0.075, 0.30),
}

def _model_price(model_name: str) -> tuple[float, float]:
    for k, v in _LLM_PRICING.items():
        if k in model_name:
            return v
    return (0.075, 0.30)

def _log_api_cost(service: str, call_type: str, model: str, input_tokens: int, output_tokens: int) -> None:
    if service == "voyage":
        cost = (input_tokens / 1_000_000) * _VOYAGE_PRICE_PER_1M
    else:
        in_p, out_p = _model_price(model)
        cost = (input_tokens / 1_000_000) * in_p + (output_tokens / 1_000_000) * out_p
    try:
        with _get_db() as conn:
            conn.execute(
                "INSERT INTO api_costs (ts,service,call_type,model,input_tokens,output_tokens,cost_usd) VALUES (?,?,?,?,?,?,?)",
                (time.time(), service, call_type, model, input_tokens, output_tokens, cost)
            )
    except Exception as _ce:
        print(f"[COST] 記錄失敗: {_ce}")

def _check_login(username: str, password: str) -> bool:
    # 環境變數帳密優先（讓 Render env var 改密碼立即生效）
    if username in SITE_USERS and password in SITE_USERS[username]:
        return True
    with _get_db() as conn:
        row = conn.execute("SELECT password FROM site_users WHERE username=?", (username,)).fetchone()
        if not row:
            return False
        stored = row["password"]
        # 已 hash 的密碼
        if stored.startswith(("pbkdf2:", "scrypt:", "sha256$", "argon2")):
            return check_password_hash(stored, password)
        # 舊的明文密碼：比對成功後自動 migrate 成 hash
        if stored == password:
            conn.execute("UPDATE site_users SET password=? WHERE username=?",
                         (generate_password_hash(password), username))
            return True
        return False

VOYAGE_API_KEY  = os.getenv("VOYAGE_API_KEY")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K           = int(os.getenv("TOP_K_RESULTS", 15))

PDF_DIR         = Path("pdfs")
EXTRA_DIR       = Path("extra_docs")
MD_DIR          = Path("114md")
QA_DIR          = Path("qa_data")
INDEX_DIR       = Path(os.getenv("INDEX_DIR",     "faiss_index"))

MD_DIR_113      = Path("113md")
INDEX_DIR_113   = Path(os.getenv("INDEX_DIR_113", "faiss_index_113"))

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
        _log_api_cost("voyage", "embed_query", "voyage-4-large", max(1, len(text) // 2), 0)
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
    input_variables=["context", "question", "user_profile"],
    template="""你是一位熟悉大學 USR（University Social Responsibility）社會責任計畫的專業助理。
請根據以下從計畫書中擷取的內容來回答問題。{user_profile}

【計畫書內容】
{context}

【問題】
{question}

【回答規則】
- 只根據上方提供的計畫書內容回答，不要自行推測或補充計畫書未提及的內容。
- 請根據現有計畫書內容盡力回答，若部分細節不足，請說明「該部分資訊有限」並以現有資料補充說明，不要直接放棄回答。
- 回答請使用繁體中文，條理清晰，內容完整，不要省略計畫書中的重要細節。
- 凡提及計畫，格式必須為「學校全名：計畫全名」，例：國立成功大學：城鄉相伴健康永續生活，不得省略學校名稱或只寫縮寫。
- 若問題明確在**列舉計畫或學校**（如「有哪些計畫」「哪些學校」「列出大學」）：若 context 開頭有【系統強制指令】與【已篩選計畫清單】，必須**嚴格以該清單為唯一依據**逐條列出，件數與順序均以清單標示為準，不得從【計畫詳細內容】中額外增加或減少條目。輸出格式：以「共X件計畫：」開頭，每條「學校全名：計畫全名」獨立一行，其下空一行寫說明，再空一行接下一條。
- 若問題在**列舉某計畫內部的項目**（如「此計畫有哪些子計畫」「涉及哪些縣市」「列出執行項目」），則從 context 中找出相關條目逐一列舉，有幾項列幾項，不得省略。
- 若問題詢問**策略、方法、做法、影響、成效、特色、原因**等概念，請歸納整理重點，不需逐一列出所有學校，可引用2至3個計畫舉例說明即可。
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

# ── USR 議題關鍵字分類表（來源：USR議題關鍵字調查整合表）──────────────
# 用途：偵測使用者問題屬於哪個議題類別，再用該類別關鍵字展開 FAISS 搜尋
USR_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "在地關懷": [
        "高齡照護", "偏鄉教育", "社區營造", "弱勢扶助", "在地關懷", "社區關懷",
        "社區培力", "青銀共創", "新住民關懷", "原住民族關懷", "偏鄉照顧", "弱勢關懷",
        "場域合作", "MOU", "在地認同", "在地需求", "韌性社區", "地方特色", "區域發展",
        "在地化經濟", "在地產業", "青年就業", "城市轉型", "共學共創", "部落共學", "世代共融",
    ],
    "環境永續": [
        "淨零碳排", "循環經濟", "環境教育", "生態保育", "永續環境", "環境永續",
        "節能減碳", "氣候變遷", "生物多樣性", "水資源", "廢棄物減量", "永續校園",
        "防災韌性", "再生能源", "綠色生活", "臺南濕地", "碳足跡", "碳中和",
        "環境治理", "水資源管理",
    ],
    "健康促進與食品安全": [
        "社區健康促進", "食品安全", "食農教育", "健康識能", "健康促進",
        "身心健康", "社區健康", "營養教育", "高齡健康", "慢性病預防", "心理健康",
        "樸門農法", "永續農業", "運動促進", "在地食材", "食品溯源", "農產品安全",
        "健康飲食", "預防保健", "社區關懷", "社區共餐", "疾病防治",
    ],
    "產業鏈結與經濟永續": [
        "地方創生", "產學合作", "青年返鄉", "產業升級", "產業鏈結", "在地產業",
        "數位轉型", "社會企業", "微型創業", "品牌行銷", "農業加值", "創新創業",
        "永續商業模式", "人才培育", "就業媒合", "區域經濟", "在地經濟", "技術轉移",
        "在地產品", "產業活化", "就業輔導", "產業轉型",
    ],
    "文化永續": [
        "文化保存", "文化傳承", "地方文史", "數位典藏", "文化永續", "文化資產",
        "傳統技藝", "無形文化資產", "原住民族文化", "地方語言", "社區記憶",
        "口述歷史", "文化創意", "文化觀光", "歷史建築活化", "跨世代傳承", "老街活化",
        "宮廟文化", "台灣在地特色", "串聯人文地景", "空間再生", "地方走讀",
        "文創設計", "部落共學",
    ],
    "其他社會實踐": [
        "社會創新", "教育平權", "數位平權", "公民參與", "多元文化", "性別平等",
        "防災教育", "社區安全", "法律扶助", "科技導入", "智慧社區", "新住民支持",
        "身心障礙者支持", "動物保護", "人權教育", "社會共融", "國際合作", "特殊議題",
        "政策倡議", "減少不平等",
    ],
    "計畫行政": [
        "計畫申請", "申請資格", "經費核銷", "績效指標", "成果報告", "配合款",
        "SIG", "SROI", "ESG", "SDGs", "助理經費", "協同主持人", "場域變更",
        "經費變更", "IRB", "計畫執行策略", "團隊架構", "聯繫窗口", "訪視",
        "實地訪視", "委員", "成效評估", "資料填報", "計畫書撰寫", "經費編列",
        "典範轉移", "幼老共學", "人事異動", "社會影響力", "社會影響力評估",
        "USR EXPO", "年報", "計畫推廣", "共培活動", "共培", "PBL", "頂石",
        "學程", "中長程校務", "USR學分學程", "USR亮點案例", "USR影響力評估",
        "USR經費補助", "USR服務學習課程", "USR特色教學", "USR產學永續聯盟",
        "跨國社會實踐計畫", "優良案例", "永續經營",
        "萌芽型", "深耕型", "計畫主持人", "管考機制", "期中報告", "USR推動中心", "專任助理", "經常門", "資本門", "自籌款", "經費流用", "量化指標", "質化指標", "利害關係人", "成果發表會", "場域盤點", "跨校合作", "諮詢委員", "內部管考", "滿意度調查",
    ],
    "轉型正義": [
        "轉型正義", "威權統治", "白色恐怖", "戒嚴時期", "二二八事件", "政治受難者",
        "加害者識別", "加害者處置", "司法不法", "行政不法", "名譽回復", "財產返還",
        "人身自由侵害賠償", "政治暴力創傷", "療癒照顧", "不義遺址", "威權象徵",
        "中正紀念堂轉型", "政治檔案", "檔案解密", "口述歷史", "國家人權記憶庫",
        "臺灣轉型正義資料庫", "原住民族轉型正義", "轉型正義教育", "人權教育",
        "促進轉型正義基金", "黨產", "附隨組織", "黨營機構",
        "歷史記憶", "社會和解", "真相調查", "人權走讀", "空間解嚴", "歷史正義", "世代對話", "政治受難者遺族", "人權地景", "司法平反", "黨外運動", "威權遺緒", "人權策展", "美麗島事件", "在地記憶", "創傷知情",
    ],
    "社會福利": [
        "社會救助", "自立脫貧", "災害救助", "遊民輔導", "社區發展", "福利社區化",
        "社區培力", "在地共生社區", "志願服務", "高齡志工", "長照", "家庭暴力防治",
        "性侵害防治", "性騷擾防制", "兒童少年保護", "兒童少年性剝削防制",
        "目睹家庭暴力", "身心障礙者保護", "老人保護", "原鄉部落服務", "離島",
        "偏遠地區社工", "社會工作", "社工督導", "心理輔導", "創傷療癒",
        "低收入戶", "中低收入戶",
        "新住民服務", "獨居老人", "弱勢家庭", "社會安全網", "青銀共創", "活躍老化", "早期療育", "喘息服務", "食物銀行", "街友關懷", "隔代教養", "社會包容", "無障礙環境", "照顧者支持", "邊緣戶", "社會創新", "婦女培力", "弱勢關懷", "友善社區", "社會支持系統",
    ],
    "健康醫療": [
        "智慧醫療", "智慧照顧", "長照3.0", "醫療人才培育", "醫護工作條件",
        "分級醫療", "垂直整合", "區域聯防", "永續醫療", "社會責任醫療",
        "精準醫療", "臨床AI", "健康台灣", "醫療資源", "偏鄉醫療", "醫院社會責任",
        "高齡友善", "在地老化", "遠距醫療", "健康促進", "健康識能", "預防延緩失能", "社區照護", "失智照護", "醫療平權", "原鄉醫療", "居家醫療", "活躍老化", "數位健康", "心理健康", "跨領域照護", "衛生教育", "慢性病管理", "弱勢照護",
    ],
    "科技研究": [
        "異種器官移植", "再生醫療", "智慧機器人", "智慧製造", "半導體",
        "精準醫學", "綠能", "再生能源", "太空任務", "衛星科學", "癌症轉譯研究",
        "原住民族研究", "族群研究", "氣候韌性", "生物多樣性", "青年學者",
        "科技研發", "國際學術合作",
        "人工智慧", "物聯網", "智慧農業", "智慧醫療", "淨零排放", "循環經濟", "大數據分析", "防災科技", "無人機應用", "產學合作", "跨領域研究", "減碳技術", "輔具科技", "海洋科學", "技術移轉", "永續科技", "智慧城市", "區塊鏈技術",
    ],
    "教育創新": [
        "STEM領域", "女性研發人才", "數位轉型", "教育雲", "校務行政e化",
        "青年職涯輔導", "U-start創新創業", "樂齡學習", "終身教育", "環境教育",
        "綠色學校", "防災教育", "校園安全", "學習扶助", "資訊素養", "數位學習",
        "實驗教育", "素養導向", "跨域學習",
        "偏鄉教育", "青銀共學", "服務學習", "微學程", "PBL教學", "創客教育", "雙語教育", "產學共育", "永續教育", "遠距伴讀", "走讀教育", "翻轉教育", "弱勢賦能", "彈性學分", "業師協同", "地方學", "體驗教育", "程式教育", "EMI課程", "創新教學",
    ],
    "產業經濟": [
        "產業競爭力", "研發轉型", "產業聯盟", "跨域合作", "海外市場", "國際貿易",
        "中小企業", "新創企業", "產業供應鏈", "低碳製造", "綠色轉型", "關稅衝擊",
        "韌性供應鏈", "出口貿易", "產業輔導", "創新研發補助",
        "產業升級", "數位轉型", "產學合作", "循環經濟", "智慧製造", "傳產轉型", "社會企業", "商業模式", "碳盤查", "技術移轉", "創新創業", "地方品牌", "ESG永續", "產業聚落", "青年創業", "價值鏈", "產銷履歷", "育成輔導", "淨零轉型", "地方產業",
    ],
}

USR_TOPIC_QUESTIONS: dict[str, list[str]] = {
    "在地關懷": [
        "如何盤點社區的實際需求",
        "與社區居民建立長期合作關係",
        "高齡照護計畫可以設計哪些活動",
        "偏鄉教育計畫如何訂定成效指標",
        "如何提升居民參與計畫的意願",
        "社區合作夥伴退出時應如何處理",
        "如何與地方連結情況",
        "如何與地方解決什麼問題",
    ],
    "環境永續": [
        "如何結合淨零碳排",
        "如何將環境教育融入正式課程",
        "循環經濟計畫可以採用哪些績效指標",
        "如何評估社區節能減碳的成果",
        "生態保育計畫如何與居民合作",
        "氣候變遷調適可設計哪些社區行動",
    ],
    "健康促進與食品安全": [
        "如何設計社區健康促進活動",
        "食品安全計畫可以結合哪些課程",
        "如何評估居民健康識能是否提升",
        "食農教育計畫可以與哪些單位合作",
        "高齡健康促進計畫應設定哪些KPI",
        "如何建立在地農產品的食品溯源機制",
        "長照具體實踐作法",
        "食品相關產出與設計",
    ],
    "產業鏈結與經濟永續": [
        "如何與在地企業合作",
        "地方創生計畫如何吸引青年返鄉",
        "如何協助地方產業進行數位轉型",
        "在地品牌應如何建立行銷通路",
        "如何評估計畫帶來的經濟效益",
        "補助結束後地方產業如何持續經營",
        "如何與產業連結程度",
        "經濟永續具體實踐方式",
    ],
    "文化永續": [
        "如何進行地方文史調查",
        "如何保存逐漸消失的傳統技藝",
        "文化資產保存計畫如何與居民合作",
        "如何建立地方文化數位典藏",
        "如何培養青年參與文化傳承",
        "文化觀光如何兼顧保存與經濟發展",
    ],
    "其他社會實踐": [
        "如何透過計畫改善數位落差",
        "社會創新計畫應如何進行需求調查",
        "如何提升弱勢族群的社會參與",
        "科技導入社區時應注意哪些倫理問題",
        "如何設計防災教育與社區韌性方案",
        "如何評估社會共融計畫的影響",
    ],
}

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

def _trunc_at_sent(text: str, max_len: int) -> str:
    """在 max_len 字元內，優先在句號/換行處截斷，避免截到字中間。"""
    if len(text) <= max_len:
        return text
    sub = text[:max_len]
    for sep in ('。', '\n', '！', '？', '；'):
        pos = sub.rfind(sep)
        if pos > max_len // 2:
            return sub[:pos + 1]
    return sub


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


def _compute_docs_hash(year: str) -> str:
    """計算所有來源文件的 hash，用來判斷文件是否有變動。"""
    md_dir = MD_DIR if year == "114" else MD_DIR_113
    paths: list[Path] = []
    if md_dir.exists():
        paths += sorted(md_dir.glob("*.md"))
    if year == "114":
        if EXTRA_DIR.exists():
            paths += sorted(EXTRA_DIR.rglob("*.txt"))
        overview = QA_DIR / "計劃總覽_114.txt"
        if overview.exists():
            paths.append(overview)
        qa_custom_path = QA_DIR / f"qa_custom_{year}.txt"
        if qa_custom_path.exists():
            paths.append(qa_custom_path)
    h = hashlib.md5()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _load_qa_custom_as_docs(qa_path: Path) -> list:
    """將 qa_custom 每個 Q/A 對拆成獨立 Document，保留 Q 與 A 的完整配對。"""
    try:
        text = qa_path.read_text(encoding="utf-8")
        raw_pairs = re.split(r"(?=\nQ:)", "\n" + text)
        docs = []
        for pair in raw_pairs:
            pair = pair.strip()
            if pair and "Q:" in pair and "A:" in pair:
                docs.append(Document(
                    page_content=pair,
                    metadata={"source": str(qa_path), "type": "qa_custom"},
                ))
        print(f"[INDEX] qa_custom 共 {len(docs)} 個 Q/A 對")
        return docs
    except Exception as e:
        print(f"[INDEX] qa_custom 載入失敗：{e}")
        return []


def load_or_build_index(year: str = "114") -> FAISS:
    """載入既有索引；文件未變動直接載入，否則重建。"""
    md_dir    = MD_DIR    if year == "114" else MD_DIR_113
    index_dir = INDEX_DIR if year == "114" else INDEX_DIR_113
    index_file = index_dir / "index.faiss"
    hash_file  = index_dir / "docs.hash"

    if index_file.exists():
        current_hash = _compute_docs_hash(year)
        stored_hash  = hash_file.read_text().strip() if hash_file.exists() else ""
        if current_hash == stored_hash:
            print(f"[INDEX] 文件未變動，載入既有索引（{year}年）...")
            return FAISS.load_local(
                str(index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        print(f"[INDEX] 文件已變動，重建索引（{year}年）...")
    else:
        print(f"[INDEX] 未找到索引（{year}年），開始建立...")
    pdf_files = list(PDF_DIR.rglob("*.pdf")) if (year == "114" and PDF_DIR.exists()) else []
    # 問答索引使用 md_dir/ 原始計畫書；llm_wiki_data/ 僅供知識地圖用
    md_files = list(md_dir.glob("*.md")) if md_dir.exists() else []
    print(f"[INDEX] 使用 {md_dir}/ 原始版（{len(md_files)} 份）")
    overview  = QA_DIR / "計劃總覽_114.txt"
    if not pdf_files and not md_files and not (year == "114" and overview.exists()):
        raise FileNotFoundError(f"{md_dir}/、qa_data/計劃總覽_114.txt 都找不到，請先放入 {year} 年度計畫書。")

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

    overview_113 = QA_DIR / "計劃總覽_113.txt"
    _overview_file = overview if year == "114" else (overview_113 if year == "113" else None)
    if _overview_file and _overview_file.exists():
        print(f"  讀取：{_overview_file.name}（按計畫邊界切割）")
        for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
            try:
                text = _overview_file.read_text(encoding=enc)
                blocks = re.split(r'===.+?===', text)
                for block in blocks:
                    block = block.strip()
                    if len(block) > 50:
                        docs.append(Document(
                            page_content=block,
                            metadata={"source": str(_overview_file)},
                        ))
                print(f"  計劃總覽切出 {len([b for b in blocks if b.strip()])} 個計畫 chunk")
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

    qa_custom_path = QA_DIR / f"qa_custom_{year}.txt"
    if qa_custom_path.exists():
        print(f"  讀取：{qa_custom_path.name}（Q/A 對，不切割直接加入）")
        qa_docs = _load_qa_custom_as_docs(qa_custom_path)
        if qa_docs:
            vectorstore.add_documents(qa_docs)
            print(f"[INDEX] qa_custom {len(qa_docs)} 個 Q/A 對已加入索引")

    index_dir.mkdir(exist_ok=True)
    vectorstore.save_local(str(index_dir))
    (index_dir / "docs.hash").write_text(_compute_docs_hash(year))
    print(f"[INDEX] {year} 年索引建立完成並已儲存。", flush=True)
    return vectorstore


# ── 對話記憶 ─────────────────────────────────────────
_chat_history: dict[str, list] = {}   # chat_id -> [{q, a}]
_chat_history_lock = threading.Lock()
_MAX_HISTORY  = 5                      # 每個 session 保留最近幾輪

# ── 使用者 profile cache ──────────────────────────────
_user_profile_cache: dict[str, tuple[str, float]] = {}  # user_id -> (profile, timestamp)
_USER_PROFILE_TTL = 300  # 5 分鐘


def _build_user_profile(user_id: str) -> str:
    """從使用者歷史問題統計常問議題與學校，回傳注入 prompt 的描述字串。"""
    if not user_id or user_id == "user":
        return ""
    try:
        with _get_db() as conn:
            rows = conn.execute("""
                SELECT m.content FROM conv_messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.user_id = ? AND m.role = 'user'
                ORDER BY m.timestamp DESC LIMIT 50
            """, (user_id,)).fetchall()
    except Exception:
        return ""

    if not rows:
        return ""

    topic_counts: dict[str, int] = {}
    school_counts: dict[str, int] = {}
    for (q,) in rows:
        topic, _ = _detect_usr_topic(q)
        if topic:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        school = _extract_school(q)
        if school:
            school_counts[school] = school_counts.get(school, 0) + 1

    if not topic_counts and not school_counts:
        return ""

    parts: list[str] = []
    if topic_counts:
        top = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        parts.append("常詢問議題：" + "、".join(f"{t}（{c}次）" for t, c in top))
    if school_counts:
        top = sorted(school_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        parts.append("常查詢學校：" + "、".join(f"{s}（{c}次）" for s, c in top))

    return "\n【使用者背景參考】此使用者過去" + "；".join(parts) + "，可作為回答時的參考方向。"


def _get_user_profile(user_id: str) -> str:
    now = time.time()
    cached = _user_profile_cache.get(user_id)
    if cached and now - cached[1] < _USER_PROFILE_TTL:
        return cached[0]
    profile = _build_user_profile(user_id)
    _user_profile_cache[user_id] = (profile, now)
    return profile

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


_MULTI_REF_RE = re.compile(
    r'以上\s*\d+\s*間|上述\s*\d+\s*間|這\s*\d+\s*間|那\s*\d+\s*間'
    r'|以上這些|上面這些|前面這些|上述計畫|以上計畫'
    r'|這些\S{0,10}計畫|這些學校|這些大學|上述這些'
)

def _extract_listed_plans(text: str) -> list[str]:
    """從前一輪編號清單提取「學校：計畫名」字串，同校多計畫各自獨立保留。"""
    plans = []
    # 主要格式：1. 學校名：計畫名  或  1. **學校名**：計畫名
    for m in re.finditer(
        r'^\d+\.\s+\*{0,2}([^\*：:\n]{2,50}?)\*{0,2}\s*[：:]\s*([^\n]{4,50})',
        text, re.MULTILINE
    ):
        school = m.group(1).strip()
        # 剝掉括號內的更名說明，例如「經國管理學院(112.08.01更名...)」→「經國管理學院」
        school = re.sub(r'[（(][^）)]*[）)]', '', school).strip()
        plan = m.group(2).strip().rstrip('。，、').split('（')[0].split('(')[0].split(' - ')[0].split(' – ')[0].strip()
        if school and 2 <= len(school) <= 25 and plan:
            plans.append(f"{school}：{plan[:30]}")
    if not plans:
        # 備援：只抓學校名（與舊行為相同）
        for m in re.finditer(r'^\d+\.\s+([^\n：:]{2,25})\n', text, re.MULTILINE):
            school = m.group(1).strip().rstrip('。，、')
            if school and 2 <= len(school) <= 25:
                plans.append(school)
    return list(dict.fromkeys(plans))  # 去重、保順序

def _extract_listed_schools(text: str) -> list[str]:
    """向下相容：回傳不重複學校名稱（從 _extract_listed_plans 取學校部分）。"""
    plans = _extract_listed_plans(text)
    seen, schools = set(), []
    for p in plans:
        s = p.split('：')[0]
        if s not in seen:
            seen.add(s)
            schools.append(s)
    return schools


_KW_THRESHOLD = 15  # 命中學校數超過此值就退回 FAISS

def _keyword_lookup(question: str, year: str = "114") -> list[str]:
    """直接關鍵字比對：問題詞 → keyword_index → 學校清單。"""
    idx = _keyword_index.get(year, {})
    if not idx:
        return []
    matched: dict[str, int] = {}
    q_tokens = set(re.findall(r'[一-鿿]{3,}', question))
    for kw, schools in idx.items():
        hit = kw in question or any(t in kw for t in q_tokens)
        if hit:
            for s in schools:
                matched[s] = matched.get(s, 0) + 1
    ranked = [s for s, _ in sorted(matched.items(), key=lambda x: -x[1])]
    return ranked


def _kw_in_question(kw: str, question: str) -> bool:
    """關鍵字比對：精確子字串 OR 拆 bigram 全命中（處理順序顛倒，如 永續農業 ↔ 農業永續）。"""
    if kw in question:
        return True
    if len(kw) >= 4:
        parts = [kw[i:i+2] for i in range(0, len(kw) - 1, 2)]
        if all(p in question for p in parts):
            return True
    return False


def _detect_usr_topic(question: str) -> tuple[str | None, list[str]]:
    """
    偵測問題是否命中 USR 議題關鍵字清單或常見提問句型。
    回傳 (議題類別, 該類別所有關鍵字)；無命中回傳 (None, [])。
    分數 = 關鍵字命中數×2 + 句型命中數×1，取最高分類別。
    """
    topic_scores: dict[str, int] = {}
    kw_hits: dict[str, list[str]] = {}

    for topic, kws in USR_TOPIC_KEYWORDS.items():
        matched = [kw for kw in kws if _kw_in_question(kw, question)]
        if matched:
            kw_hits[topic] = matched
            topic_scores[topic] = topic_scores.get(topic, 0) + len(matched) * 2

    for topic, patterns in USR_TOPIC_QUESTIONS.items():
        for pattern in patterns:
            for i in range(len(pattern) - 4):
                if pattern[i:i+5] in question:
                    topic_scores[topic] = topic_scores.get(topic, 0) + 1
                    break

    if not topic_scores:
        return None, []
    best = max(topic_scores, key=lambda t: topic_scores[t])
    print(f"[TOPIC] 偵測議題：{best}，分數：{topic_scores[best]}，關鍵字：{kw_hits.get(best, [])}")
    return best, USR_TOPIC_KEYWORDS[best]


def _detect_all_usr_topics(question: str) -> tuple[str | None, list[str]]:
    """列舉型專用：回傳所有有命中的議題類別，合併其關鍵字供 Sequential Query 廣搜。"""
    topic_scores: dict[str, int] = {}
    for topic, kws in USR_TOPIC_KEYWORDS.items():
        matched = [kw for kw in kws if _kw_in_question(kw, question)]
        if matched:
            topic_scores[topic] = len(matched) * 2
    for topic, patterns in USR_TOPIC_QUESTIONS.items():
        for pattern in patterns:
            for i in range(len(pattern) - 4):
                if pattern[i:i+5] in question:
                    topic_scores[topic] = topic_scores.get(topic, 0) + 1
                    break
    if not topic_scores:
        return None, []
    best = max(topic_scores, key=lambda t: topic_scores[t])
    # 合併所有命中類別的關鍵字（去重）
    merged_kws: list[str] = []
    seen: set[str] = set()
    for topic in sorted(topic_scores, key=lambda t: -topic_scores[t]):
        for kw in USR_TOPIC_KEYWORDS[topic]:
            if kw not in seen:
                seen.add(kw)
                merged_kws.append(kw)
    print(f"[TOPIC-ALL] 命中 {len(topic_scores)} 個類別：{list(topic_scores.keys())}，合併 {len(merged_kws)} 個關鍵字")
    return best, merged_kws


def _count_kw_hits(text: str, kws: list[str]) -> dict[str, int]:
    """計算每個關鍵字在文件中的出現次數，只回傳次數 > 0 的。"""
    return {kw: cnt for kw in kws if (cnt := text.count(kw)) > 0}


def _build_sdg_map(vs) -> dict[str, list[str]]:
    """啟動時掃整個 docstore，建立 {SDG號碼: ["學校：計畫", ...]} 對應表（1-17 全部）。"""
    sdg_srcs: dict[str, set[str]] = {}
    seen_src: set[str] = set()
    for doc in vs.docstore._dict.values():
        raw_src = Path(doc.metadata.get("source", "")).stem
        src = re.sub(r'_formatted$', '', raw_src).strip('_')
        if not src or src in seen_src:
            continue
        seen_src.add(src)
        text = doc.page_content
        for m in re.finditer(r'SDG\s*0?(\d{1,2})', text, re.IGNORECASE):
            n = m.group(1).lstrip('0') or '1'
            if 1 <= int(n) <= 17:
                sdg_srcs.setdefault(n, set()).add(src)
    result: dict[str, list[str]] = {}
    for n, srcs in sdg_srcs.items():
        lines = []
        for src in sorted(srcs):
            parts = src.split('_', 1)
            if len(parts) == 2:
                lines.append(f"{parts[0]}：{parts[1]}")
        result[n] = lines
    counts = {n: len(v) for n, v in sorted(result.items(), key=lambda x: int(x[0]))}
    print(f"[SDG-MAP] 建立完成：{counts}")
    return result


def _seq_query_by_index(kws: list[str], topic: str, index: dict,
                        dedup_by_source: int = 0,
                        min_hits: int = 3,
                        condense: bool = False,
                        priority_kws: list[str] | None = None) -> list[str]:
    """用倒排索引快速查詢命中關鍵字的 chunk，不需全表掃描。
    dedup_by_source=True 時每個來源檔案只保留命中最高的一個 chunk（列舉型用）。
    condense=True 時每筆只輸出學校名稱 + 150 字摘要（列舉型用，節省 context 空間）。
    """
    doc_scores: dict[int, list] = {}  # id(doc) → [doc, total, {kw: cnt}]
    for kw in kws:
        for doc, cnt in index.get(kw, []):
            did = id(doc)
            if did not in doc_scores:
                doc_scores[did] = [doc, 0, {}]
            doc_scores[did][1] += cnt
            doc_scores[did][2][kw] = doc_scores[did][2].get(kw, 0) + cnt

    ranked = sorted(doc_scores.values(), key=lambda x: -x[1])

    # 同一來源只取命中最高的前 N 個 chunk（dedup_by_source=N，0 表示不限）
    if dedup_by_source:
        seen_src: dict[str, int] = {}
        deduped = []
        for entry in ranked:
            src = entry[0].metadata.get("source", "")
            cnt = seen_src.get(src, 0)
            if cnt < dedup_by_source:
                seen_src[src] = cnt + 1
                deduped.append(entry)
        ranked = deduped

    high, mid = [], []
    for doc, total, hits in ranked:
        if total < min_hits:
            continue
        text = _clean_plan_code(doc.page_content)
        src_name = _clean_plan_code(Path(doc.metadata.get("source", "")).stem)
        if condense:
            # chunk 開頭有 prepend 的【school　project（type）】標籤，跳過
            bracket_end = text.find('】\n')
            text_body = text[bracket_end + 2:] if 0 <= bracket_end < 120 else text
            sentences = [s.strip() for s in re.split(r'[。！？\n]', text_body) if len(s.strip()) > 15]
            # 排除純標題行（以下任一即視為 heading，不含實質內容）
            _CN_NUM = r'一二三四五六七八九十百壹貳叁肆伍陸柒捌玖拾'
            def _is_heading(s: str) -> bool:
                # markdown heading（#）
                if re.match(r'^#{1,7}\s', s):
                    return True
                # 中文數字大節標題（一、二、三、叁、...）含大寫數字
                if re.match(rf'^[{_CN_NUM}\d]+[、．.]\s', s):
                    return True
                # （一）（二）... 型子節，不論後面接什麼都算標題
                if re.match(rf'^[（(][{_CN_NUM}\d]{{1,3}}[）)]', s):
                    return True
                # 純注入標籤行（school　project），無其他內容
                if re.match(r'^（[^）　]*　[^）]*）\s*$', s):
                    return True
                return False
            content_sents = [s for s in sentences if not _is_heading(s)]
            # 永遠只從 content_sents 選句，不 fallback 回含標題的 sentences
            pri = [s for s in content_sents if priority_kws and any(kw in s for kw in priority_kws)]
            rest = [s for s in content_sents if s not in pri and any(kw in s for kw in hits)]
            if pri or rest:
                selected = (pri[:3] + rest)[:4] if pri else rest[:4]
            else:
                selected = content_sents[:3]  # 沒有命中詞也取前幾句內容，不輸出標題
            snippet = '。'.join(selected) + '。' if selected else text_body[:300]
            entry = f"【{src_name}】\n{snippet}…"
        else:
            # 非列舉型：優先取使用者查詢詞的句子，再補其他命中詞句子，最多 2 句
            sentences = [s.strip() for s in re.split(r'[。！？\n]', text) if len(s.strip()) > 10]
            pri = [s for s in sentences if priority_kws and any(kw in s for kw in priority_kws)]
            rest = [s for s in sentences if s not in pri and any(kw in s for kw in hits)]
            selected = (pri[:2] + rest)[:2] if pri else rest[:2]
            snippet = "。".join(selected) + "。" if selected else text[:120]
            entry = f"【{src_name}】\n{snippet}"
        high.append(entry) if total >= 5 else mid.append(entry)

    print(f"[SEQ] 高度相關 {len(high)} 篇，部分相關 {len(mid)} 篇"
          + ("（已去重複）" if dedup_by_source else "")
          + ("（精簡模式）" if condense else ""))
    return high + mid


def _seq_query_live(kws: list[str], topic: str, vs,
                    dedup_by_source: bool = True,
                    condense: bool = False) -> list[str]:
    """即時掃描所有文件找 LLM 生成的關鍵字（不依賴預建索引）。
    用於預建索引未涵蓋的詞彙（如「流浪動物」、「收容所」）。
    condense=True 時每筆只輸出學校名稱 + 150 字摘要（列舉型用）。
    """
    if not kws:
        return []
    all_docs = list(vs.docstore._dict.values())
    doc_scores: dict[int, list] = {}
    for doc in all_docs:
        text = _clean_plan_code(doc.page_content)
        total = 0
        hits: dict[str, int] = {}
        for kw in kws:
            cnt = text.count(kw)
            if cnt:
                total += cnt
                hits[kw] = cnt
        if total > 0:
            did = id(doc)
            doc_scores[did] = [doc, total, hits]

    ranked = sorted(doc_scores.values(), key=lambda x: -x[1])

    if dedup_by_source:
        seen_src: set[str] = set()
        deduped = []
        for entry in ranked:
            src = entry[0].metadata.get("source", "")
            if src not in seen_src:
                seen_src.add(src)
                deduped.append(entry)
        ranked = deduped

    result = []
    for doc, total, hits in ranked:
        text = _clean_plan_code(doc.page_content)
        hit_summary = "、".join(f"{kw}×{cnt}" for kw, cnt in hits.items())
        if condense:
            src_name = _clean_plan_code(Path(doc.metadata.get("source", "")).stem)
            first_pos = min((text.find(kw) for kw in hits if text.find(kw) >= 0), default=0)
            start = max(0, first_pos - 30)
            snippet = text[start:start + 400]
            result.append(f"【{src_name}】\n{snippet}…")
        else:
            label = "高度相關" if total >= 3 else "部分相關"
            result.append(f"【{label}｜{topic}｜命中：{hit_summary}】\n{text}")

    print(f"[SEQ-LIVE] 即時掃描找到 {len(result)} 篇（關鍵字：{kws[:5]}...）"
          + ("（精簡模式）" if condense else ""))
    return result


def _count_inv_matches(kws: list[str], inv: dict) -> int:
    """用倒排索引快速計算命中任一關鍵字的不重複 doc 數量。"""
    doc_ids: set[int] = set()
    for kw in kws:
        for doc, _ in inv.get(kw, []):
            doc_ids.add(id(doc))
    return len(doc_ids)


def _annotate_docs_by_topic(docs, topic: str, kws: list[str]) -> list[str]:
    """
    Sequential 掃描所有文件，依關鍵字命中次數標記相關程度：
      5+  次 → 【高度相關】完整內容送入 context
      3-4 次 → 【部分相關】加標記說明命中的關鍵字，再送完整內容
      <3  次 → 跳過（不納入 context）

    回傳已排序、已標記的 context 字串清單（高度相關排前面）。
    """
    high, mid = [], []

    for doc in docs:
        text = _clean_plan_code(doc.page_content)
        hits = _count_kw_hits(text, kws)
        total = sum(hits.values())
        if total < 1:
            continue
        hit_summary = "、".join(f"{kw}×{cnt}" for kw, cnt in hits.items())
        if total >= 5:
            label = f"【高度相關｜{topic}｜命中：{hit_summary}】"
            high.append(f"{label}\n{text}")
        elif total >= 3:
            label = f"【部分相關｜{topic}｜命中：{hit_summary}】"
            mid.append(f"{label}\n{text}")
        else:
            label = f"【少量相關｜{topic}｜命中：{hit_summary}】"
            mid.append(f"{label}\n{text}")

    print(f"[SEQ] 高度相關 {len(high)} 篇，部分相關 {len(mid)} 篇")
    return high + mid


def _prepare_search_query(question: str, history: list) -> str:
    """
    有對話歷史時，用 LLM 改寫問題：
    - 若是追問（含代名詞/省略指涉），替換為具體名稱
    - 若是全新主題，原樣回傳
    回傳 search_question
    """
    from langchain_core.messages import HumanMessage as _HM
    last = history[-1]
    prompt = (
        "你是 USR 計畫書搜尋助理。請判斷新問題是否為追問，輸出 JSON。\n\n"
        f"【前一輪問題】\n{last['q']}\n\n"
        f"【前一輪回答摘要】\n{last['a'][:3000]}\n\n"
        f"【新問題】\n{question}\n\n"
        "任務（追問改寫）：\n"
        "・若新問題是追問（含代名詞/省略指涉/延伸前一輪主題），替換為具體名稱，輸出改寫後的完整問題\n"
        "・若非追問（全新主題），原樣輸出新問題\n\n"
        '只輸出 JSON，不要其他文字：{"search_q": "最終問題"}'
    )
    try:
        res = llm_fast.bind(temperature=0, thinking_budget=0).invoke([_HM(content=prompt)])
        text = res.content.strip()
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            search_q = (data.get("search_q") or question).strip() or question
            if search_q != question:
                prev_school = _extract_school(history[-1]['q'])
                if prev_school and prev_school not in search_q:
                    search_q = f"{prev_school}：{search_q}"
            print(f"[PREP] search={search_q!r}")
            return search_q
    except Exception as e:
        print(f"[PREP] 失敗：{e}")
    return question


# ── 啟動時初始化 ──────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=os.environ["LLM_MODEL"],
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    thinking_budget=2048,
)
llm_fast = ChatGoogleGenerativeAI(
    model=os.environ.get("LLM_MODEL_FAST", os.environ["LLM_MODEL"]),
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    thinking_budget=512,
)

vectorstores: dict = {}
try:
    vs = load_or_build_index("114")
    vectorstores["114"] = vs
    print("[APP] RAG 114 就緒。")
except FileNotFoundError as e:
    vectorstores["114"] = None
    print(f"[APP] 警告 114：{e}")

# ── SDG 對應表（優先從 sdg_map.json 讀取，不存在才掃 vectorstore 並存檔）──
_sdg_maps: dict[str, dict[str, list[str]]] = {}
_SDG_MAP_PATH = Path("sdg_map.json")

def _load_or_build_sdg_maps() -> None:
    if _SDG_MAP_PATH.exists():
        try:
            with open(_SDG_MAP_PATH, encoding="utf-8") as _f:
                _sdg_maps.update(json.load(_f))
            total = sum(len(v) for yr in _sdg_maps.values() for v in yr.values())
            print(f"[SDG-MAP] 從 sdg_map.json 載入（{total} 筆）")
            return
        except Exception as _e:
            print(f"[SDG-MAP] 載入失敗，重建：{_e}")
    # 從 vectorstore 掃描建立並存檔
    built: dict[str, dict[str, list[str]]] = {}
    for yr, vs in vectorstores.items():
        if vs:
            built[yr] = _build_sdg_map(vs)
    _sdg_maps.update(built)
    try:
        with open(_SDG_MAP_PATH, "w", encoding="utf-8") as _f:
            json.dump(_sdg_maps, _f, ensure_ascii=False, indent=2)
        print(f"[SDG-MAP] 已存至 sdg_map.json")
    except Exception as _e:
        print(f"[SDG-MAP] 存檔失敗：{_e}")

_load_or_build_sdg_maps()

# ── 懶載入鎖（113 年首次請求時才載）──
_lazy_load_lock = threading.Lock()

def _ensure_year_loaded(year: str) -> None:
    """113 年索引懶載入，只有第一次被請求時才從磁碟載入。"""
    if year == "114" or vectorstores.get(year) is not None:
        return
    with _lazy_load_lock:
        if vectorstores.get(year) is not None:
            return
        print(f"[APP] 懶載入 {year} 年索引...")
        try:
            _vs = load_or_build_index(year)
            vectorstores[year] = _vs
            if year not in _sdg_maps:
                _sdg_maps[year] = _build_sdg_map(_vs)
            # 補建 keyword_index（若此年尚未建議題詞索引）
            _all_topic_kws = {kw for kws in USR_TOPIC_KEYWORDS.values() for kw in kws}
            yr_idx = _keyword_index.setdefault(year, {})
            if _all_topic_kws - set(yr_idx.keys()):
                yr_idx.update(_build_topic_kw_index(year, _vs))
                try:
                    _KW_INDEX_PATH.write_text(
                        json.dumps(_keyword_index, ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )
                except Exception:
                    pass
            print(f"[APP] {year} 年索引就緒。")
        except FileNotFoundError as e:
            vectorstores[year] = None
            print(f"[APP] 警告 {year}：{e}")

init_qa()

# ── 關鍵字索引 ────────────────────────────────────────
_keyword_index: dict[str, dict[str, list[str]]] = {}
_KW_INDEX_PATH = Path("keyword_index.json")

def _build_topic_kw_index(year: str, vs) -> dict[str, list[str]]:
    """掃 vectorstore，建立 USR_TOPIC_KEYWORDS 詞 → 計畫名單。"""
    all_kws: set[str] = {kw for kws in USR_TOPIC_KEYWORDS.values() for kw in kws}
    kw_plans: dict[str, set[str]] = {kw: set() for kw in all_kws}
    t0 = time.perf_counter()
    for doc in vs.docstore._dict.values():
        text = _clean_plan_code(doc.page_content)
        stem = Path(doc.metadata.get("source", "")).stem
        parts = stem.split('_', 1)
        if len(parts) < 2:
            continue
        plan = f"{parts[0]}：{parts[1]}"
        for kw in all_kws:
            if kw in text:
                kw_plans[kw].add(plan)
    elapsed = round((time.perf_counter() - t0) * 1000)
    result = {kw: sorted(plans) for kw, plans in kw_plans.items() if plans}
    print(f"[KW-BUILD] {year} 年：{len(result)} 個詞，耗時 {elapsed}ms")
    return result

def _load_or_build_kw_index() -> None:
    """載入 keyword_index.json；若某年缺少 USR_TOPIC_KEYWORDS 就自動掃 vectorstore 補建。"""
    _all_topic_kws: set[str] = {kw for kws in USR_TOPIC_KEYWORDS.values() for kw in kws}
    if _KW_INDEX_PATH.exists():
        try:
            with open(_KW_INDEX_PATH, encoding="utf-8") as _f:
                _kw_data = json.load(_f)
            for yr in ("114", "113"):
                _keyword_index[yr] = _kw_data.get(yr, {})
            print(f"[KW-IDX] 載入：{sum(len(v) for v in _keyword_index.values())} 個關鍵字")
        except Exception as _e:
            print(f"[KW-IDX] 載入失敗：{_e}")

    # 補建缺少的年份（USR_TOPIC_KEYWORDS 詞不在索引裡）
    _updated = False
    for yr, vs in vectorstores.items():
        if not vs:
            continue
        yr_idx = _keyword_index.setdefault(yr, {})
        _missing = _all_topic_kws - set(yr_idx.keys())
        if _missing:
            print(f"[KW-BUILD] {yr} 年缺少 {len(_missing)} 個議題詞，自動建索引...")
            _new = _build_topic_kw_index(yr, vs)
            yr_idx.update(_new)
            _updated = True

    if _updated:
        try:
            _KW_INDEX_PATH.write_text(
                json.dumps(_keyword_index, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"[KW-IDX] 已更新並存檔")
        except Exception as _e:
            print(f"[KW-IDX] 存檔失敗：{_e}")

_load_or_build_kw_index()


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
    username = session.get("username", "") if authenticated else ""
    return render_template("index.html", authenticated=authenticated,
        username=username, is_admin=_is_admin(),
        plans_114=_load_plans("114"), plans_113=_load_plans("113"))


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    u = (data.get("username") or "").strip()
    p = (data.get("password") or "").strip()
    with _get_db() as _db:
        has_any = _db.execute("SELECT COUNT(*) FROM site_users").fetchone()[0] > 0
    if not has_any and not SITE_USERS:
        return jsonify({"ok": False, "error": "系統尚未設定帳號，請聯絡管理員。"}), 503
    valid = _check_login(u, p)
    if valid:
        session.permanent = True
        session["authenticated"] = True
        session["username"] = u or ""
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "帳號或密碼錯誤，請再試一次。"}), 401


@app.route("/logout")
def logout():
    session.pop("authenticated", None)
    session.pop("username", None)
    return redirect(url_for("index"))


@app.route("/api/conversations", methods=["GET"])
def api_list_conversations():
    if not session.get("authenticated"):
        return jsonify({"error": "unauthorized"}), 401
    user_id = session.get("username") or "user"
    with _get_db() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 100",
            (user_id,)
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/conversations", methods=["POST"])
def api_create_conversation():
    if not session.get("authenticated"):
        return jsonify({"error": "unauthorized"}), 401
    user_id = session.get("username") or "user"
    data = request.get_json() or {}
    conv_id = data.get("id") or str(_uuid.uuid4())
    title = data.get("title", "新對話")
    now = time.time()
    with _get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
            (conv_id, user_id, title, now, now)
        )
    return jsonify({"id": conv_id})

@app.route("/api/conversations/<conv_id>", methods=["GET"])
def api_get_conversation(conv_id):
    if not session.get("authenticated"):
        return jsonify({"error": "unauthorized"}), 401
    user_id = session.get("username") or "user"
    with _get_db() as conn:
        conv = conn.execute("SELECT id, title, user_id FROM conversations WHERE id=?", (conv_id,)).fetchone()
        if not conv or conv["user_id"] != user_id:
            return jsonify({"error": "not found"}), 404
        messages = conn.execute(
            "SELECT role, content, timestamp FROM conv_messages WHERE conversation_id=? ORDER BY timestamp ASC",
            (conv_id,)
        ).fetchall()
    return jsonify({"id": conv_id, "title": conv["title"], "messages": [dict(m) for m in messages]})

@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def api_delete_conversation(conv_id):
    if not session.get("authenticated"):
        return jsonify({"error": "unauthorized"}), 401
    user_id = session.get("username") or "user"
    with _get_db() as conn:
        conv = conn.execute("SELECT user_id FROM conversations WHERE id=?", (conv_id,)).fetchone()
        if not conv or conv["user_id"] != user_id:
            return jsonify({"error": "not found"}), 404
        conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    return jsonify({"ok": True})

@app.route("/api/conversations/<conv_id>/title", methods=["PATCH"])
def api_update_title(conv_id):
    if not session.get("authenticated"):
        return jsonify({"error": "unauthorized"}), 401
    user_id = session.get("username") or "user"
    data = request.get_json() or {}
    title = (data.get("title") or "").strip()[:50] or "新對話"
    with _get_db() as conn:
        conv = conn.execute("SELECT user_id FROM conversations WHERE id=?", (conv_id,)).fetchone()
        if not conv or conv["user_id"] != user_id:
            return jsonify({"error": "not found"}), 404
        conn.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (title, time.time(), conv_id))
    return jsonify({"ok": True})


@app.route("/admin")
def admin_page():
    if not _is_admin():
        return redirect("/")
    return render_template("admin.html")

@app.route("/api/admin/stats")
def api_admin_stats():
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    with _get_db() as conn:
        users = conn.execute("""
            SELECT
                c.user_id,
                COUNT(DISTINCT c.id)                                    AS conv_count,
                SUM(CASE WHEN m.role='user' THEN 1 ELSE 0 END)         AS question_count,
                SUM(CASE WHEN m.role='user'
                         AND m.timestamp > strftime('%s','now','start of day')
                         THEN 1 ELSE 0 END)                             AS today_count,
                MAX(c.updated_at)                                       AS last_active,
                (SELECT lm.content FROM conv_messages lm
                 JOIN conversations lc ON lm.conversation_id = lc.id
                 WHERE lc.user_id = c.user_id AND lm.role = 'user'
                 ORDER BY lm.timestamp DESC LIMIT 1)                    AS last_question
            FROM conversations c
            LEFT JOIN conv_messages m ON m.conversation_id = c.id
            GROUP BY c.user_id
            ORDER BY last_active DESC
        """).fetchall()
        result = [dict(u) for u in users]
    return jsonify(result)

@app.route("/api/admin/user/<username>/messages")
def api_admin_user_messages(username):
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    with _get_db() as conn:
        msgs = conn.execute("""
            SELECT m.content, m.timestamp, c.title
            FROM conv_messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE c.user_id = ? AND m.role = 'user'
            ORDER BY m.timestamp DESC
            LIMIT 200
        """, (username,)).fetchall()
    return jsonify([dict(m) for m in msgs])

@app.route("/api/admin/users_list", methods=["GET"])
def api_admin_users_list():
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    with _get_db() as conn:
        rows = conn.execute("SELECT username, created_at FROM site_users ORDER BY created_at ASC").fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/admin/users_list", methods=["POST"])
def api_admin_add_user():
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    if not username or not password:
        return jsonify({"error": "帳號和密碼不得為空"}), 400
    try:
        with _get_db() as conn:
            conn.execute(
                "INSERT INTO site_users (username, password, created_at) VALUES (?,?,?)",
                (username, generate_password_hash(password), time.time())
            )
        return jsonify({"ok": True})
    except Exception:
        return jsonify({"error": "帳號已存在"}), 409

@app.route("/api/admin/user_account/<username>", methods=["DELETE"])
def api_admin_delete_user(username):
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    if username in ADMIN_USERS:
        return jsonify({"error": "不能刪除管理員帳號"}), 400
    with _get_db() as conn:
        conn.execute("DELETE FROM site_users WHERE username=?", (username,))
    return jsonify({"ok": True})

@app.route("/api/admin/user_account/<username>/password", methods=["PATCH"])
def api_admin_change_password(username):
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    data = request.get_json() or {}
    password = (data.get("password") or "").strip()
    if not password:
        return jsonify({"error": "密碼不得為空"}), 400
    with _get_db() as conn:
        conn.execute("UPDATE site_users SET password=? WHERE username=?",
                     (generate_password_hash(password), username))
    return jsonify({"ok": True})


@app.route("/api/admin/settings", methods=["GET"])
def api_admin_settings_get():
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    with _get_db() as conn:
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
    settings = {r["key"]: r["value"] for r in rows}
    return jsonify({
        "announcement": settings.get("announcement", ""),
        "sysinfo": {
            "LLM_MODEL": os.environ.get("LLM_MODEL", "—"),
            "LLM_MODEL_FAST": os.environ.get("LLM_MODEL_FAST", "—"),
            "TOP_K": str(TOP_K),
            "CHUNK_SIZE": str(CHUNK_SIZE),
            "CHUNK_OVERLAP": str(CHUNK_OVERLAP),
        }
    })

@app.route("/api/admin/settings", methods=["PATCH"])
def api_admin_settings_patch():
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    data = request.get_json() or {}
    key = (data.get("key") or "").strip()
    value = data.get("value", "")
    if key not in ("announcement",):
        return jsonify({"error": "不允許修改此設定"}), 400
    with _get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO app_settings (key, value, updated_at) VALUES (?,?,?)",
            (key, value, time.time())
        )
    return jsonify({"ok": True})

@app.route("/api/admin/costs")
def api_admin_costs():
    if not _is_admin():
        return jsonify({"error": "unauthorized"}), 403
    import datetime
    now = time.time()
    utc_now = datetime.datetime.utcfromtimestamp(now)
    today_start = datetime.datetime(utc_now.year, utc_now.month, utc_now.day).timestamp()
    month_start = datetime.datetime(utc_now.year, utc_now.month, 1).timestamp()
    with _get_db() as conn:
        by_svc = conn.execute("""
            SELECT service,
                   SUM(input_tokens) AS input_tokens,
                   SUM(output_tokens) AS output_tokens,
                   ROUND(SUM(cost_usd),6) AS cost_usd,
                   COUNT(*) AS calls
            FROM api_costs
            GROUP BY service
            ORDER BY cost_usd DESC
        """).fetchall()
        by_type = conn.execute("""
            SELECT service, call_type,
                   SUM(input_tokens) AS input_tokens,
                   SUM(output_tokens) AS output_tokens,
                   ROUND(SUM(cost_usd),6) AS cost_usd,
                   COUNT(*) AS calls
            FROM api_costs
            GROUP BY service, call_type
            ORDER BY service, cost_usd DESC
        """).fetchall()
        daily = conn.execute("""
            SELECT DATE(ts,'unixepoch') AS day,
                   ROUND(SUM(cost_usd),6) AS cost_usd,
                   SUM(input_tokens) AS input_tokens,
                   SUM(output_tokens) AS output_tokens,
                   COUNT(*) AS calls
            FROM api_costs
            WHERE ts >= ?
            GROUP BY day
            ORDER BY day DESC
            LIMIT 14
        """, (now - 14 * 86400,)).fetchall()
        today_row  = conn.execute("SELECT COALESCE(ROUND(SUM(cost_usd),6),0) AS v FROM api_costs WHERE ts>=?", (today_start,)).fetchone()
        month_row  = conn.execute("SELECT COALESCE(ROUND(SUM(cost_usd),6),0) AS v FROM api_costs WHERE ts>=?", (month_start,)).fetchone()
        total_row  = conn.execute("SELECT COALESCE(ROUND(SUM(cost_usd),6),0) AS v FROM api_costs").fetchone()
    return jsonify({
        "by_service": [dict(r) for r in by_svc],
        "by_type":    [dict(r) for r in by_type],
        "daily":      [dict(r) for r in daily],
        "today_usd":  today_row["v"],
        "month_usd":  month_row["v"],
        "total_usd":  total_row["v"],
    })

@app.route("/ask", methods=["POST"])
def ask():
    if SITE_PASSWORD and not session.get("authenticated"):
        return jsonify({"error": "請先登入。"}), 401

    if vectorstores.get("114") is None:
        return jsonify({"error": "索引尚未建立，請先將 PDF 放入 pdfs/ 資料夾後重啟伺服器。"}), 503

    data = request.get_json(silent=True) or {}
    question = _sanitize_prompt_input((data.get("question") or "").strip())
    chat_id    = (data.get("chat_id") or "").strip()
    conv_id    = (data.get("conv_id") or "").strip()
    use_context = bool(data.get("use_context", False))
    user_id    = session.get("username") or "user"
    user_type = (data.get("user_type") or "applicant").strip()
    if user_type not in ("applicant", "reviewer"):
        user_type = "applicant"
    original_question = _sanitize_prompt_input((data.get("original_question") or "").strip())
    skip_eval = bool(original_question)
    if original_question:
        question = f"{original_question}（請依以下標準評估：{question}）"
    year = (data.get("year") or "114").strip()
    if year not in ("113", "114"):
        year = "114"
    _ensure_year_loaded(year)
    vs = vectorstores.get(year) or vectorstores.get("114")

    if not question:
        return jsonify({"error": "請輸入問題。"}), 400
    if len(question) > 500:
        return jsonify({"error": "問題不得超過 500 字。"}), 400

    def generate():
        try:
            t0 = time.perf_counter()

            # ── 送出 highlight 詞彙供前端標記 ──
            _HL_GENERIC = {"USR", "計畫", "學校", "大學"}
            _hl_topic, _hl_topic_kws = _detect_usr_topic(question)
            _hl_topic_terms = [kw for kw in (_hl_topic_kws or []) if len(kw) >= 3 and kw not in _HL_GENERIC][:20]
            _hl_query_terms = [t for t in _extract_query_terms(question) if t not in _HL_GENERIC and t not in _hl_topic_terms]
            _hl_terms = list(dict.fromkeys(_hl_topic_terms + _hl_query_terms))[:25]
            if _hl_terms:
                yield f"data: {json.dumps({'type': 'highlight_terms', 'terms': _hl_terms}, ensure_ascii=False)}\n\n"

            # ── ① qa_custom 短路攔截（優先於所有流程）──
            structured_ctx = try_structured_answer(question, year=year)
            if structured_ctx:
                yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': structured_ctx}, ensure_ascii=False)}\n\n"
                total_ms = round((time.perf_counter() - t0) * 1000)
                yield f"data: {json.dumps({'type': 'done', 'timing': {'total_ms': total_ms}, 'mode': 'qa_custom'}, ensure_ascii=False)}\n\n"
                return

            # ── 對話記憶 + 搜尋問題準備 ──
            history = (_chat_history.get(chat_id, []) if chat_id else []) if use_context else []
            t_prepare_start = time.perf_counter()
            if history:
                search_question = _prepare_search_query(question, history)
            else:
                search_question = question
            t_prepare_end = time.perf_counter()

            # ── USR 議題關鍵字偵測：建立 expand_query 供 FAISS 多輪搜尋 ──
            expand_query: str | None = None
            _list_check = bool(_LIST_INTENT_RE.search(search_question)) and not _LIST_CONCEPT_RE.search(search_question)
            if _list_check:
                # 列舉型：合併所有命中類別的關鍵字，廣度優先
                _usr_topic, _usr_topic_kws = _detect_all_usr_topics(question)
            else:
                _usr_topic, _usr_topic_kws = _detect_usr_topic(question)
            if _usr_topic and _usr_topic_kws:
                expand_query = " ".join(_usr_topic_kws)
                print(f"[TOPIC] 議題展開 query（前60字）：{expand_query[:60]}")

            # ── LLM：議題語意分類 ──
            _explicit_followup = bool(_MULTI_REF_RE.search(question) and history)
            _is_followup: bool = _explicit_followup

            # ③ 主觀評量：含排名/比較/最佳等詞，需釐清評估標準（列舉型問題不觸發）
            if not skip_eval and not _list_check and _is_evaluation_question(question):
                _clarify_msg = _generate_clarify_msg(question)
                yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': _clarify_msg}, ensure_ascii=False)}\n\n"
                total_ms = round((time.perf_counter() - t0) * 1000)
                if conv_id:
                    try:
                        _now = time.time()
                        with _get_db() as _conn:
                            _conn.execute(
                                "INSERT OR IGNORE INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
                                (conv_id, user_id, question[:30] if question else "新對話", _now, _now)
                            )
                            _conn.execute(
                                "INSERT OR IGNORE INTO conv_messages (id, conversation_id, role, content, timestamp) VALUES (?,?,?,?,?)",
                                (str(_uuid.uuid4()), conv_id, "user", question, _now - 0.001)
                            )
                            _conn.execute(
                                "INSERT OR IGNORE INTO conv_messages (id, conversation_id, role, content, timestamp) VALUES (?,?,?,?,?)",
                                (str(_uuid.uuid4()), conv_id, "assistant", _clarify_msg, _now)
                            )
                            _conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now, conv_id))
                    except Exception as _e:
                        print(f"[CONV] 儲存對話失敗: {_e}")
                yield f"data: {json.dumps({'type': 'done', 'timing': {'total_ms': total_ms}, 'mode': 'clarify', 'original_question': question}, ensure_ascii=False)}\n\n"
                return

            # ② 本地預計算（不需 API call）
            _school = _extract_school(search_question)
            if not _school and history:
                _school = _extract_school(history[-1]['q'])
                if _school:
                    print(f"[ASK] 從歷史補充學校：{_school}")
            _list      = bool(_LIST_INTENT_RE.search(search_question)) and not _school and not _LIST_CONCEPT_RE.search(search_question)
            _query_is_or = '或' in question
            _personnel = bool(_PERSONNEL_RE.search(search_question))
            _kw        = _extract_keywords(search_question)
            _role      = _extract_role_term(question) if _personnel else None
            _topic     = None
            if _school:
                _topic = question.replace(_school, "").strip()
                _topic = re.sub(r'[的跟與和相關有關請問]+', ' ', _topic).strip() or None
            _fetch = TOP_K * 5 if _school else (TOP_K * 4 if _personnel else (TOP_K * 3 if _list else TOP_K))

            # ③ 多校清單追問偵測（明確追問詞 or LLM 判斷為追問）
            _listed_schools: list[str] = []
            if _is_followup and history:
                for _h in reversed(history):
                    _candidate = _h.get('plans') or _h.get('schools') or _extract_listed_plans(_h['a'])
                    if _candidate:
                        _listed_schools = _candidate
                        break
                _src = "明確追問詞" if _explicit_followup else "LLM判斷"
                print(f"[ASK] 追問偵測（{_src}），{len(_listed_schools)} 件：{_listed_schools[:3]}")

            # ③-b 關鍵字索引查詢（USR議題詞 + SDG，優先於 TOPIC_LLM）
            _kw_list_hit: str | None = None
            _kw_plan_list: list[str] = []
            _kw_pre_schools: set[str] = set()
            _kw_pre_live_results: list[str] = []
            _kw_pre_extra: list[str] = []
            if not _listed_schools and not _school and _list:
                _kw_idx_pre = _keyword_index.get(year, {})
                _all_topic_kws_set: set[str] = {kw for kws in USR_TOPIC_KEYWORDS.values() for kw in kws}
                _q_terms_pre = _extract_query_terms(search_question)
                _kw_stop_pre = {'計畫', '學校', '大學', '哪些', '相關', '有關', '年度', 'USR'}
                _plan_set_pre: set[str] = set()
                _matched_kws: list[str] = []

                # 1. _usr_topic_kws（_detect_usr_topic 已偵測到的議題關鍵字）
                for _tk in (_usr_topic_kws or []):
                    if _tk in _kw_idx_pre:
                        _plan_set_pre.update(_kw_idx_pre[_tk])
                        if _tk not in _matched_kws:
                            _matched_kws.append(_tk)

                # 2. 問題裡其他 USR_TOPIC_KEYWORDS 詞
                for _qt in _q_terms_pre:
                    if _qt in _all_topic_kws_set and _qt in _kw_idx_pre and _qt not in _matched_kws:
                        _plan_set_pre.update(_kw_idx_pre[_qt])
                        _matched_kws.append(_qt)

                # 3. SDG key 直接出現在問題裡
                for _kw_pre in _kw_idx_pre:
                    if re.match(r'^SDG\d{1,2}$', _kw_pre) and _kw_pre in search_question:
                        if _kw_pre not in _matched_kws:
                            _matched_kws.append(_kw_pre)
                            _plan_set_pre.update(_kw_idx_pre[_kw_pre])

                if _matched_kws:
                    _kw_list_hit = _matched_kws[0]
                    _kw_plan_list = sorted(_plan_set_pre)
                    print(f"[KW-PRE] 命中 {_matched_kws[:3]} → {len(_kw_plan_list)} 件")
                    # 額外未知詞 live scan 過濾
                    _extra_pre = [k for k in _q_terms_pre
                                  if k not in _matched_kws and k not in _kw_stop_pre
                                  and len(k) >= 2 and k not in _all_topic_kws_set]
                    if _extra_pre:
                        print(f"[KW-PRE] 額外詞：{_extra_pre}（OR={_query_is_or}）")
                        _fres = _seq_query_live(_extra_pre, "列舉", vs, condense=True)
                        _kw_pre_live_results = _fres
                        _kw_pre_extra = _extra_pre
                        if _fres:
                            _fschools = {m.group(1) for r in _fres
                                         if (m := re.match(r'【(.+?)_', r))}
                            if _fschools and not _query_is_or:
                                _orig_n = len(_kw_plan_list)
                                _kw_plan_list = [e for e in _kw_plan_list
                                                 if e.split('：', 1)[0] in _fschools]
                                _kw_pre_schools = _fschools
                                print(f"[KW-PRE] AND 篩選後 {len(_kw_plan_list)}/{_orig_n} 件")
                            elif _fschools and _query_is_or:
                                _kw_pre_schools = _fschools
                                print(f"[KW-PRE] OR 模式，額外命中學校 {len(_fschools)} 間")

            # ── LLM：議題語意分類（keyword_index 完全未命中才跑）──
            if _list_check and not _kw_list_hit:
                _llm_topics = _llm_classify_topics(question)
                for _lt in _llm_topics:
                    if _lt not in (_usr_topic or ""):
                        for _tk in USR_TOPIC_KEYWORDS.get(_lt, []):
                            if _tk not in (_usr_topic_kws or []):
                                if _usr_topic_kws is None:
                                    _usr_topic_kws = []
                                _usr_topic_kws.append(_tk)
                        if not _usr_topic:
                            _usr_topic = _lt

            # ③-c 多校追問（>5 間）→ 強制列舉型（壓縮+60k limit），但跳過 Seq Query
            _multi_enumerate = bool(_listed_schools and len(_listed_schools) > 5 and not _list)
            if _multi_enumerate:
                _list = True
                print(f"[ASK] 多校追問({len(_listed_schools)}間) → 強制列舉型（跳過SeqQuery）")

            # ④ Voyage AI 平行 embed
            docs = []
            t_voyage = t_faiss = time.perf_counter()
            if _listed_schools:
                # 多校模式：各計畫 query 同時送出（entry 可為「學校：計畫名」或「學校名」）
                _topic_q = re.sub(r'以上\S*間\S*計劃?|上述\S*計劃?|這些計劃?', '', question).strip()
                with ThreadPoolExecutor() as ex:
                    _sfuts = {s: ex.submit(
                        embeddings.embed_query,
                        f"{s.replace('：', ' ')} {_topic_q}"  # 把計畫名也放進 query 提升精準度
                    ) for s in _listed_schools}
                    _svecs = {s: f.result() for s, f in _sfuts.items()}
                t_voyage = time.perf_counter()
                docs_all = []
                for _s in _listed_schools:
                    _s_docs = vs.similarity_search_by_vector(_svecs[_s], k=TOP_K * 3)
                    _s_filtered = _school_filter_docs(_s_docs, _s, k=10)
                    docs_all = _merge_docs(docs_all, _s_filtered)
                    print(f"[ASK] 多校輪「{_s}」→ {len(_s_filtered)} 筆")
                docs = docs_all
                _school = None
                expand_query = None
                t_faiss = time.perf_counter()
            elif not _list:
                # 一般模式：所有已知 query（含 expand）同時 embed
                embed_tasks: dict[str, str] = {'main': search_question}
                if _kw:
                    embed_tasks['kw'] = _kw
                if _role:
                    embed_tasks['role'] = _role
                if _school:
                    embed_tasks['school'] = _school
                    if _topic:
                        embed_tasks['topic'] = _topic
                if expand_query:
                    embed_tasks['expand'] = expand_query

                with ThreadPoolExecutor() as ex:
                    embed_futs = {k: ex.submit(embeddings.embed_query, v)
                                  for k, v in embed_tasks.items()}
                    vecs = {k: f.result() for k, f in embed_futs.items()}
                t_voyage = time.perf_counter()

                expand_vec = vecs.get('expand')

                # ⑤ FAISS 多輪搜尋
                docs1 = vs.similarity_search_by_vector(vecs['main'], k=_fetch)

                if _kw:
                    docs2 = vs.similarity_search_by_vector(vecs['kw'], k=TOP_K)
                    docs_all = _merge_docs(docs1, docs2)
                    print(f"[ASK] 第二輪關鍵詞「{_kw}」→ 合併後 {len(docs_all)} 筆")
                else:
                    docs_all = docs1

                if _personnel and _role:
                    docs3 = vs.similarity_search_by_vector(vecs['role'], k=TOP_K * 2)
                    docs_all = _merge_docs(docs_all, docs3)
                    print(f"[ASK] 人員角色輪「{_role}」→ 合併後 {len(docs_all)} 筆")

                if expand_vec:
                    docs_expand = vs.similarity_search_by_vector(expand_vec, k=TOP_K * 2)
                    docs_all = _merge_docs(docs_all, docs_expand)
                    print(f"[ASK] 詞義擴充輪 → 合併後 {len(docs_all)} 筆")

                if _school:
                    if _topic:
                        docs_topic = vs.similarity_search_by_vector(vecs['topic'], k=TOP_K * 3)
                        docs_all = _merge_docs(docs_all, docs_topic)
                        print(f"[ASK] 學校主題輪「{_topic}」→ 合併後 {len(docs_all)} 筆")
                    docs_school = vs.similarity_search_by_vector(vecs['school'], k=TOP_K * 10)
                    docs_all = _merge_docs(docs_all, docs_school)
                    print(f"[ASK] 學校名稱輪「{_school}」→ 合併後 {len(docs_all)} 筆")
                    docs = _school_filter_docs(docs_all, _school, k=9999)
                    print(f"[ASK] 學校過濾「{_school}」→ {len(docs)} 筆")
                else:
                    if _list:
                        docs = docs_all[:_fetch]
                    else:
                        # 非列舉非單校：每間學校只保留最相關一個 chunk，避免一間壟斷 context
                        docs = _dedup_by_school(docs_all, k=_fetch)
                t_faiss = time.perf_counter()
                _faiss_srcs = [Path(d.metadata.get("source","")).stem for d in docs]
                print(f"[FAISS-DOCS] {len(docs)} 筆：{_faiss_srcs}")

            # ── 議題關鍵字索引查詢（keyword_index）+ live scan ──────────────
            _q_priority_kws: list[str] = []
            _q_terms: list[str] = []
            _live_scan_kws: list[str] = []
            annotated: list[str] | None = None
            _plan_list_lines: list[str] = []

            if _list and not _multi_enumerate:
                _q_terms = _extract_query_terms(question)
                _q_priority_kws = [k for k in _q_terms if k in search_question]
                _kw_idx = _keyword_index.get(year) or _keyword_index.get("114", {})
                _all_topic_kws: set[str] = {kw for kws in USR_TOPIC_KEYWORDS.values() for kw in kws}

                # ── keyword_index 查詢：_usr_topic_kws + 問題中的已知議題詞 ──
                _topic_plan_set: set[str] = set()
                _lookup_kws = list(dict.fromkeys(
                    list(_usr_topic_kws or []) + [k for k in _q_terms if k in _all_topic_kws]
                ))
                for _lk in _lookup_kws:
                    if _lk in _kw_idx:
                        _topic_plan_set.update(_kw_idx[_lk])
                if _topic_plan_set:
                    print(f"[KW-IDX] 議題詞 {len(_lookup_kws)} 個 → {len(_topic_plan_set)} 件計畫")
                    if not _plan_list_lines:
                        _plan_list_lines = sorted(_topic_plan_set)
                    else:
                        _kw_schools = {e.split('：')[0] for e in _plan_list_lines}
                        _extra = [p for p in sorted(_topic_plan_set)
                                  if p.split('：')[0] not in _kw_schools
                                  and (not _kw_pre_schools or p.split('：')[0] in _kw_pre_schools)]
                        if _extra:
                            _plan_list_lines = _plan_list_lines + _extra
                            print(f"[KW-IDX] 補充 {len(_extra)} 件計畫，共 {len(_plan_list_lines)} 件")

                # ── live scan：KW-IDX 已有結果則跳過；無結果才掃問題詞中不在議題詞典的詞 ──
                _live_scan_kws = (
                    [] if _plan_list_lines
                    else [k for k in _q_terms if k not in _all_topic_kws]
                )
                annotated = []
                if _live_scan_kws:
                    _cached_kws = set(_kw_pre_extra)
                    _need_scan_kws = [k for k in _live_scan_kws if k not in _cached_kws]
                    _cached_results = _kw_pre_live_results if any(k in _cached_kws for k in _live_scan_kws) else []
                    if _cached_results:
                        print(f"[LIVE] 使用 KW-PRE 快取（{len(_cached_results)} 筆）：{[k for k in _live_scan_kws if k in _cached_kws]}")

                    _live_results: list[str] = []
                    if _need_scan_kws and len(_need_scan_kws) >= 2 and not _query_is_or:
                        # 多詞 AND：各自掃一次取學校交集
                        _live_school_sets = []
                        _live_per_kw: dict[str, list[str]] = {}
                        for _lk in _need_scan_kws:
                            _res = _seq_query_live([_lk], "列舉", vs, condense=True)
                            _live_per_kw[_lk] = _res
                            _schools = {re.match(r'【(.+?)(?:_|】)', r).group(1)
                                        for r in _res if re.match(r'【(.+?)(?:_|】)', r)}
                            _live_school_sets.append(_schools)
                            print(f"[LIVE] 「{_lk}」→ {len(_schools)} 間學校")
                        _and_schools = _live_school_sets[0].intersection(*_live_school_sets[1:])
                        print(f"[LIVE] AND 交集 {len(_and_schools)} 間學校")
                        if _and_schools:
                            _live_results = [r for kw_res in _live_per_kw.values()
                                             for r in kw_res if any(s in r for s in _and_schools)]
                        else:
                            _live_results = _seq_query_live(_need_scan_kws, "列舉", vs, condense=True)
                            print(f"[LIVE] AND 交集為空，退回 OR")
                    elif _need_scan_kws:
                        if _query_is_or and len(_need_scan_kws) >= 2:
                            print(f"[LIVE] OR 模式，{len(_need_scan_kws)} 詞取聯集")
                        _live_results = _seq_query_live(_need_scan_kws, "列舉", vs, condense=True)

                    # 合併快取
                    _seen_cache = {r[:80] for r in _live_results}
                    _live_results += [r for r in _cached_results if r[:80] not in _seen_cache]
                    if _live_results:
                        annotated = _live_results
                        print(f"[LIVE] 共 {len(annotated)} 筆")

                # 追問過濾
                if _listed_schools and annotated:
                    annotated = [a for a in annotated if any(s.split('：')[0] in a for s in _listed_schools)]
                    print(f"[LIVE] 追問學校過濾後 {len(annotated)} 筆")

            # 列舉型用 Flash 處理大 context 很快，給更多空間；其他問題截短避免拖慢 Pro
            _CTX_CHAR_LIMIT = 60000 if _list else 30000
            if _list:
                # 列舉型：FAISS 結果也精簡，每筆只保留學校名稱 + 150 字摘要
                faiss_texts = [
                    f"【{_clean_plan_code(Path(doc.metadata.get('source','')).stem)}】\n"
                    f"{_clean_plan_code(doc.page_content)[:150]}…"
                    for doc in docs
                ]
            else:
                faiss_texts = [_clean_plan_code(doc.page_content) for doc in docs]
            print(f"[LIST-GATE] _list={_list} annotated={type(annotated).__name__ if annotated is not None else 'None'}({len(annotated) if annotated else 0}) _usr_topic={_usr_topic} plan_list={len(_plan_list_lines)}")
            _MAX_PLAN_LIST = 150  # LLM 輸出上限（超過會被截斷）

            _plan_to_snippet: dict[str, str] = {}
            if _list:
                # faiss_texts（per-school FAISS 結果）+ annotated（SEQ 結果）合併補 snippet
                _combined_src = list(faiss_texts) + (annotated or [])
                _seen_plan_srcs: set[str] = set()
                _tier1: list[str] = []
                _tier2_scored: list[tuple[int, str]] = []
                _t2_score_kws = [k for k in (_q_priority_kws + _q_terms) if len(k) >= 2]
                for _entry in _combined_src:
                    _m = re.match(r'【(.+)】', _entry.split('\n', 1)[0])
                    if not _m:
                        continue
                    _src = _m.group(1)
                    if _src in _seen_plan_srcs:
                        continue
                    _parts = _src.split('_', 1)
                    if len(_parts) < 2:
                        continue
                    _seen_plan_srcs.add(_src)
                    _line = f"{_parts[0]}：{_parts[1]}"
                    _plan_to_snippet[_line] = _entry[_m.end():].strip()[:500]
                    if _q_priority_kws and any(pk in _entry for pk in _q_priority_kws):
                        _tier1.append(_line)
                    else:
                        _score = sum(1 for k in _t2_score_kws if k in _entry)
                        _tier2_scored.append((_score, _line))
                _tier2_scored.sort(key=lambda x: x[0], reverse=True)
                _T2_CAP = 50
                if _q_priority_kws:
                    _t2_nonzero = [(s, l) for s, l in _tier2_scored if s > 0]
                    _t2_zero    = [(s, l) for s, l in _tier2_scored if s == 0]
                    _tier2 = [l for _, l in (_t2_nonzero + _t2_zero)[:_T2_CAP]]
                else:
                    _tier2 = [l for _, l in _tier2_scored[:_T2_CAP]]
                if not _plan_list_lines:
                    _plan_list_lines = _tier1 + _tier2
                    print(f"[LIST-DEBUG] SEQ 提取 {len(_plan_list_lines)} 件（T1={len(_tier1)}，T2={len(_tier2)}，priority_kws={_q_priority_kws[:3]}）")
                else:
                    # keyword_index 已提供基礎清單，SEQ 補充 keyword_index 沒有的學校
                    _kw_schools = {e.split('：')[0] for e in _plan_list_lines}
                    _seq_extra = [l for l in (_tier1 + _tier2)
                                  if l.split('：')[0] not in _kw_schools
                                  and (not _kw_pre_schools or l.split('：')[0] in _kw_pre_schools)]
                    if _seq_extra:
                        _plan_list_lines = _plan_list_lines + _seq_extra
                        print(f"[LIST-DEBUG] keyword_index {len(_kw_plan_list)} 件 + SEQ 補充 {len(_seq_extra)} 件 = {len(_plan_list_lines)} 件")
                    else:
                        print(f"[LIST-DEBUG] keyword_index {len(_plan_list_lines)} 件，SEQ 無新增學校")

                # 地區過濾：問題含縣市/大區域關鍵字時只保留對應縣市學校
                _question_counties = _detect_question_counties(question)
                if _question_counties:
                    _region_filtered = [l for l in _plan_list_lines if _get_school_county(l.split('：')[0]) in _question_counties]
                    if _region_filtered:
                        _plan_list_lines = _region_filtered
                        print(f"[LIST-REGION] 縣市過濾={_question_counties}，剩餘 {len(_plan_list_lines)} 件")
                    else:
                        print(f"[LIST-REGION] 縣市過濾={_question_counties}，無匹配，不過濾")

                # ── 列舉型 per-school FAISS（確定清單後取各計畫內文）──────────
                if not faiss_texts and _plan_list_lines:
                    _pq = re.sub(r'以上\S*間\S*計劃?|上述\S*計劃?|這些計劃?', '', question).strip()
                    with ThreadPoolExecutor() as ex:
                        _pfuts = {s: ex.submit(embeddings.embed_query,
                                               f"{s.replace('：', ' ')} {_pq}")
                                  for s in _plan_list_lines[:_MAX_PLAN_LIST]}
                        _pvecs = {s: f.result() for s, f in _pfuts.items()}
                    for _s in _plan_list_lines[:_MAX_PLAN_LIST]:
                        if _s in _plan_to_snippet:
                            continue
                        _sd = vs.similarity_search_by_vector(_pvecs[_s], k=TOP_K * 2)
                        _sf = _school_filter_docs(_sd, _s, k=5)
                        if _sf:
                            # re-ranking：含 live scan 關鍵詞的 chunk 優先排前面
                            if _live_scan_kws:
                                _sf.sort(key=lambda d: -sum(
                                    _clean_plan_code(d.page_content).count(kw)
                                    for kw in _live_scan_kws
                                ))
                            _plan_to_snippet[_s] = _trunc_at_sent('\n'.join(
                                _trunc_at_sent(_clean_plan_code(d.page_content), 200) for d in _sf
                            ), 600)
                    print(f"[FAISS-PLAN] per-school FAISS 完成，_plan_to_snippet {len(_plan_to_snippet)} 件")

                # 偵測分析型子問題（多個？分隔），有則限制清單件數
                _extra_sub_qs = [p for p in [p.strip() for p in re.split(r'[？?]', question) if p.strip()][1:]
                                 if re.search(r'什麼|哪些|哪幾|如何|為何|為什麼|怎麼|怎樣|多少|幾個|幾間|幾件|哪', p)]
                _display_lines = _plan_list_lines[:25] if _extra_sub_qs else _plan_list_lines
                _list_display_note = f"（另有更多計畫，以下列出前{len(_display_lines)}件）" if _extra_sub_qs and len(_plan_list_lines) > len(_display_lines) else ""
                print(f"[LIST-SUB] extra_sub_qs={_extra_sub_qs} display={len(_display_lines)} total={len(_plan_list_lines)}")

                # ── 列舉型並行路徑：每個計畫單獨送 LLM，擷取原文關鍵句 ──
                from langchain_core.messages import HumanMessage as _HMList

                def _sum_one_plan(_plan_line: str) -> str:
                    _snip = _plan_to_snippet.get(_plan_line, "")
                    if not _snip:
                        return ""
                    _topic_kws_for_prompt = list(dict.fromkeys(
                        k for k in (_q_priority_kws + list(_usr_topic_kws or []))
                        if len(k) >= 2
                    ))[:20]
                    _kw_hint = (
                        f"- 優先擷取與以下主題關鍵字語意相關的句子：{'、'.join(_topic_kws_for_prompt)}\n"
                        if _topic_kws_for_prompt else ""
                    )
                    _p = (
                        f"根據以下計畫書內容，用1～3句白話中文說明這個計畫**具體在做什麼**。"
                        f"規則：\n"
                        f"{_kw_hint}"
                        f"- 說明重點：做了什麼活動、服務對象是誰、在哪裡執行、達成什麼效果\n"
                        f"- 若有具體合作對象（企業、社區、機構名稱）或數字（場次、人次、件數），必須保留\n"
                        f"- 用流暢白話整理，不要照抄原文，不要條列，不要輸出「此計畫致力於…」「本計畫旨在…」等開頭\n"
                        f"- 將與查詢議題語意相關的詞語（含同義詞、相關概念）用**標記**\n"
                        f"- 若內容僅含章節標題（如「一、」「（一）」「叁、」「## 標題」等）或單位名稱清單、聯絡表格等無具體描述，直接輸出「#RAW」\n"
                        f"只輸出說明句，不要其他文字。\n\n{_snip}"
                    )
                    try:
                        _r = llm_fast.bind(temperature=0, thinking_budget=0).invoke([_HMList(content=_p)])
                        _out = _normalize_content(_r.content).strip()
                        try:
                            _um = getattr(_r, 'usage_metadata', None) or {}
                            _log_api_cost("gemini", "list_para", _LLM_FAST_MODEL_NAME, _um.get('input_tokens', 0), _um.get('output_tokens', 0))
                        except Exception:
                            pass
                        # fallback：LLM 說找不到內容，直接截原文
                        if "#RAW" in _out:
                            return ""
                        _refusal_hints = ("並未包含", "無法擷取", "沒有具體", "僅列出", "不包含描述")
                        if any(h in _out for h in _refusal_hints) or len(_out) < 10:
                            return _snip[:200]
                        return _out
                    except Exception as _pe:
                        print(f"[LIST-PARA] 擷取失敗: {_pe}")
                        return _snip[:200]

                # ── 舊標籤保留（後面程式碼用到）──
                # 補送更完整的 highlight 詞：query 詞 + 所有議題關鍵字（不限 25 個）
                _HL_GENERIC2 = {"USR", "計畫", "學校", "大學", "計畫有"}
                _extra_hl = list(dict.fromkeys(
                    k for k in (_q_priority_kws + _q_terms + list(_hl_topic_kws or []))
                    if len(k) >= 2 and k not in _HL_GENERIC2
                ))
                if _extra_hl:
                    yield f"data: {json.dumps({'type': 'highlight_terms', 'terms': _extra_hl}, ensure_ascii=False)}\n\n"
                _para_sources: list[dict] = []
                _para_seen: set = set()
                for _pd in docs:
                    _ps = _clean_plan_code(Path(_pd.metadata.get("source", "")).stem)
                    _pp = _pd.metadata.get("page", 0) + 1
                    if (_ps, _pp) not in _para_seen:
                        _para_seen.add((_ps, _pp))
                        _para_sources.append({"source": _ps, "page": _pp})
                yield f"data: {json.dumps({'type': 'sources', 'sources': _para_sources}, ensure_ascii=False)}\n\n"

                _t1_label = (
                    f"（前{len(_tier1)}件直接命中查詢詞，後段為同議題相關）"
                    if _tier1 and len(_tier1) < len(_plan_list_lines) else ""
                )
                _para_ans_parts: list[str] = []
                _para_t0 = time.perf_counter()

                # 先收集所有結果，才能在標頭寫正確件數
                _para_collected: list[tuple[str, str]] = []
                _skipped = 0
                with ThreadPoolExecutor(max_workers=25) as _para_ex:
                    _para_futs = [(_pl, _para_ex.submit(_sum_one_plan, _pl)) for _pl in _display_lines]
                    for _pl, _pf in _para_futs:
                        _ps2 = _pf.result()
                        if _ps2 == "":
                            _skipped += 1
                            continue
                        _para_collected.append((_pl, _ps2))

                _out_idx = len(_para_collected)
                _skip_note = f"，另 {_skipped} 件無具體描述略去" if _skipped else ""
                _header_txt = f"找到 {_out_idx} 件相關計畫{_list_display_note}{_t1_label}{_skip_note}\n\n"
                _para_ans_parts.append(_header_txt)
                yield f"data: {json.dumps({'type': 'chunk', 'text': _header_txt}, ensure_ascii=False)}\n\n"
                _para_t_first = time.perf_counter()

                for _i, (_pl, _ps2) in enumerate(_para_collected, 1):
                    _pchunk = f"{_i}. {_pl}\n{_ps2}\n"
                    _para_ans_parts.append(_pchunk)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': _pchunk}, ensure_ascii=False)}\n\n"

                # 子問題分析：列完後再送 LLM 回答
                if _extra_sub_qs:
                    _sub_context = "\n\n".join(
                        f"{i+1}. {pl}\n{_plan_to_snippet.get(pl, '')[:400]}"
                        for i, pl in enumerate(_plan_list_lines[:30])
                    )
                    _sub_q_text = "\n".join(f"- {q}？" for q in _extra_sub_qs)
                    _sub_prompt = (
                        f"根據以下 USR 計畫資料，依序回答問題。每個問題需引用2~3個具體計畫，"
                        f"必須列出學校名稱與計畫名稱，並說明該計畫的具體做法或案例，不得只做概括描述：\n{_sub_q_text}\n\n"
                        f"計畫資料：\n{_sub_context}"
                    )
                    _sub_intro = "\n---\n"
                    _para_ans_parts.append(_sub_intro)
                    yield f"data: {json.dumps({'type': 'chunk', 'text': _sub_intro}, ensure_ascii=False)}\n\n"
                    for _schunk in llm.stream([_HMList(content=_sub_prompt)]):
                        _spiece = _normalize_content(_schunk.content)
                        if _spiece:
                            _para_ans_parts.append(_spiece)
                            yield f"data: {json.dumps({'type': 'chunk', 'text': _spiece}, ensure_ascii=False)}\n\n"

                _para_t_end = time.perf_counter()
                _para_full_ans = "".join(_para_ans_parts)
                _para_timing = {
                    "prepare_ms":  round((t_prepare_end - t_prepare_start) * 1000),
                    "voyage_ms":   round((t_voyage - t_prepare_end) * 1000),
                    "faiss_ms":    round((t_faiss - t_voyage) * 1000),
                    "llm_first_ms": round((_para_t_first - _para_t0) * 1000),
                    "llm_total_ms": round((_para_t_end - _para_t0) * 1000),
                    "total_ms":    round((_para_t_end - t0) * 1000),
                }
                print(f"[LIST-PARA] 並行摘要 {len(_plan_list_lines)} 件，輸出={len(_para_full_ans)}字元，總計={_para_timing['total_ms']}ms")

                _para_hl_extra = _generate_hl_terms(question, _para_full_ans[:800])
                if _para_hl_extra:
                    yield f"data: {json.dumps({'type': 'highlight_terms', 'terms': _para_hl_extra}, ensure_ascii=False)}\n\n"
                _para_sugg = _generate_suggestions(question, _para_full_ans[:600])
                if _para_sugg:
                    yield f"data: {json.dumps({'type': 'suggested_questions', 'questions': _para_sugg}, ensure_ascii=False)}\n\n"

                if conv_id and user_id:
                    try:
                        _now2 = time.time()
                        with _get_db() as _conn2:
                            _conn2.execute(
                                "INSERT OR IGNORE INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
                                (conv_id, user_id, question[:30] if question else "新對話", _now2, _now2)
                            )
                            _ex2 = _conn2.execute("SELECT title FROM conversations WHERE id=?", (conv_id,)).fetchone()
                            if _ex2 and _ex2["title"] == "新對話" and question:
                                _conn2.execute("UPDATE conversations SET title=? WHERE id=?", (question[:30], conv_id))
                            _conn2.execute(
                                "INSERT OR IGNORE INTO conv_messages (id, conversation_id, role, content, timestamp) VALUES (?,?,?,?,?)",
                                (str(_uuid.uuid4()), conv_id, "user", question, _now2 - 0.001)
                            )
                            _conn2.execute(
                                "INSERT OR IGNORE INTO conv_messages (id, conversation_id, role, content, timestamp) VALUES (?,?,?,?,?)",
                                (str(_uuid.uuid4()), conv_id, "assistant", _para_full_ans, _now2)
                            )
                            _conn2.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now2, conv_id))
                    except Exception as _ce:
                        print(f"[CONV] 儲存對話失敗: {_ce}")

                yield f"data: {json.dumps({'type': 'done', 'timing': _para_timing})}\n\n"

                if chat_id:
                    with _chat_history_lock:
                        _ph = _chat_history.setdefault(chat_id, [])
                        _ph.append({"q": question, "a": _para_full_ans[:5000], "plans": _extract_listed_plans(_para_full_ans)})
                        if len(_ph) > _MAX_HISTORY:
                            _ph.pop(0)
                    if len(_chat_history) > 1000:
                        for _hk in list(_chat_history.keys())[:200]:
                            del _chat_history[_hk]
                return  # 跳過原本的大 LLM call

                # 列舉型：SEQ（精確命中）+ FAISS（語意補充）合併，確保廣度
                seen_heads = {a[:80] for a in annotated}
                extra = [t for t in faiss_texts if t[:80] not in seen_heads]
                context = "\n\n".join(annotated + extra)
            elif annotated:
                context = "\n\n".join(annotated)
            else:
                context = "\n\n".join(faiss_texts)
            if len(context) > _CTX_CHAR_LIMIT:
                context = context[:_CTX_CHAR_LIMIT]
            # 列舉型：在 context 前注入預建清單，讓 LLM 直接按清單輸出，不自行過濾
            if _list and _plan_list_lines:
                _t1_count = len([l for l in _plan_list_lines if any(pk in l for pk in _q_priority_kws)]) if _q_priority_kws else 0
                _plans_str = "\n".join(f"{i+1}. {p}" for i, p in enumerate(_plan_list_lines))  # 先建完整版
                _sort_note = (
                    f"清單依相關度排序：前 {_t1_count} 件直接提及查詢關鍵詞，後段為議題相關計畫。"
                    if _q_priority_kws and _t1_count > 0 else ""
                )
                _topic_label = f"（{_usr_topic}議題）" if _usr_topic else ""
                _t1_highlight = f"前 {_t1_count} 件直接命中查詢詞，其餘為同議題相關計畫。" if _t1_count and _t1_count < len(_plan_list_lines) else ""
                # 偵測列舉問題之外的分析型子問題（多個？分隔）
                _extra_sub_qs = [p for p in [p.strip() for p in re.split(r'[？?]', question) if p.strip()][1:]
                                 if re.search(r'什麼|哪些|哪幾|如何|為何|為什麼|怎麼|怎樣|多少|幾個|幾間|幾件|哪', p)]
                # 有分析型子問題時限制清單件數，確保 LLM 有空間回答後續問題
                _display_lines = _plan_list_lines[:25] if _extra_sub_qs else _plan_list_lines
                print(f"[LIST-SUB] extra_sub_qs={_extra_sub_qs} display={len(_display_lines)} total={len(_plan_list_lines)}")
                _extra_q_rule = (
                    f"5. 所有計畫列完後，另起段落依序回答以下子問題（從【計畫詳細內容】歸納，引用2~3個計畫舉例說明）：\n"
                    + "\n".join(f"   - {q}？" for q in _extra_sub_qs) + "\n"
                ) if _extra_sub_qs else ""
                _rule4_suffix = "請接著回答第5點各子問題。" if _extra_sub_qs else f"可另起段落做整體補充。{_sort_note}"
                _list_note = f"（另有更多相關計畫，以下舉例前 {len(_display_lines)} 件）" if _extra_sub_qs and len(_plan_list_lines) > len(_display_lines) else ""
                _sys_directive = (
                    f"【系統強制指令】以下清單已由關鍵字引擎確認，共 {len(_plan_list_lines)} 件計畫，"
                    f"因問題含後續子問題，本次僅列前 {len(_display_lines)} 件作為舉例，列完後必須回答第5點子問題。{_t1_highlight}"
                ) if _extra_sub_qs else (
                    f"【系統強制指令】以下清單已由關鍵字引擎確認，共 {len(_plan_list_lines)} 件計畫，請全數列出。{_t1_highlight}"
                )
                context = (
                    f"{_sys_directive}\n"
                    f"輸出規則：\n"
                    f"1. 第一行必須是「共{len(_plan_list_lines)}件計畫{_list_note}：」\n"
                    f"2. 逐條列出下方【已篩選計畫清單】全部 {len(_display_lines)} 件，格式「N. 學校全名：計畫全名」，計畫名稱必須與清單完全一致逐字照抄，不得增刪或改動任何字符（含標點符號），不得省略任何一件、不得自行更改件數，嚴禁從【計畫詳細內容】自行增加清單以外的計畫。\n"
                    f"3. 每條計畫名稱下方加一句說明（從【計畫詳細內容】中找到對應條目後摘述），說明不限查詢關鍵字，可摘述任何核心工作。若找不到對應內容，直接跳過不寫，絕對不可輸出「資訊有限」「未提及」「僅列出計畫名稱」等佔位語句。\n"
                    f"4. 所有 {len(_display_lines)} 件列完後，{_rule4_suffix}\n"
                    f"{_extra_q_rule}\n"
                    f"【已篩選計畫清單】\n{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(_display_lines))}\n\n"
                    f"【計畫詳細內容（可參考各計畫任何核心內容）】\n"
                ) + context
                print(f"[LIST] 預建清單 {len(_plan_list_lines)} 件（T1={_t1_count}），注入 context 前")
            # 多校追問：提示 LLM 針對每間學校分別回答，不得合併或省略
            if _multi_enumerate:
                context = f"【本問題涉及前一輪列出的 {len(_listed_schools)} 件計畫，請在回答中針對 context 中每件計畫分別說明，不得合併舉例或省略任何一件。】\n\n" + context

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

            _user_profile = "" if user_type == "reviewer" else _get_user_profile(user_id)
            prompt_value = (REVIEWER_PROMPT if user_type == "reviewer" else RAG_PROMPT).invoke(
                {"context": context, "question": question, "user_profile": _user_profile}
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
            if _list:
                _active_llm = llm_fast.bind(thinking_budget=0, temperature=0)  # 列舉型：固定輸出，避免件數飄移
            elif bool(_school) and not _usr_topic:
                _active_llm = llm_fast  # 事實查詢：Flash 有思考
            else:
                _active_llm = llm  # 分析／推理：Pro
            answer_chars = 0
            answer_parts = []
            t_first_chunk = None
            _stream_usage_meta = None
            t_gemini_start = time.perf_counter()
            for chunk in _active_llm.stream(prompt_value):
                content = chunk.content
                if isinstance(content, list):
                    content = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                    _stream_usage_meta = chunk.usage_metadata
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
            try:
                _um = _stream_usage_meta or {}
                _stream_model = _LLM_MODEL_NAME if (_active_llm is llm) else _LLM_FAST_MODEL_NAME
                _log_api_cost("gemini", "rag_stream", _stream_model,
                              _um.get('input_tokens', prompt_chars // 2),
                              _um.get('output_tokens', answer_chars // 2))
            except Exception:
                pass

            timing = {
                "prepare_ms":     round((t_prepare_end - t_prepare_start) * 1000),
                "voyage_ms":      round((t_voyage - t_prepare_end) * 1000),
                "faiss_ms":       round((t_faiss - t_voyage) * 1000),
                "llm_first_ms":   round((t_first_chunk - t_gemini_start) * 1000),
                "llm_total_ms":   round((t_end - t_gemini_start) * 1000),
                "total_ms":       round((t_end - t0) * 1000),
            }
            print(
                f"[TIMING] 查詢改寫={timing['prepare_ms']}ms"
                f" | Voyage嵌入={timing['voyage_ms']}ms"
                f" | FAISS搜尋={timing['faiss_ms']}ms"
                f" | LLM首字={timing['llm_first_ms']}ms"
                f" | LLM完成={timing['llm_total_ms']}ms"
                f" | 總計={timing['total_ms']}ms"
            )

            # ── 評量問題小總結（original_question 存在代表是評量問題流程）──
            if original_question:
                from langchain_core.messages import HumanMessage as _HMSum
                _sum_prompt = (
                    f"根據以下回答內容，用1～2句話直接回答原始問題。\n"
                    f"原始問題：{original_question}\n"
                    f"回答摘要：{''.join(answer_parts)[:800]}\n\n"
                    f"只輸出結論句，不要其他文字。"
                )
                try:
                    _sum_res = llm_fast.bind(temperature=0, thinking_budget=0).invoke([_HMSum(content=_sum_prompt)])
                    _sum_text = _normalize_content(_sum_res.content).strip()
                    if _sum_text:
                        _sum_chunk = f"\n\n---\n**結論**\n{_sum_text}"
                        yield f"data: {json.dumps({'type': 'chunk', 'text': _sum_chunk}, ensure_ascii=False)}\n\n"
                        answer_parts.append(_sum_chunk)
                        try:
                            _um = getattr(_sum_res, 'usage_metadata', None) or {}
                            _log_api_cost("gemini", "eval_summary", _LLM_FAST_MODEL_NAME, _um.get('input_tokens', 0), _um.get('output_tokens', 0))
                        except Exception:
                            pass
                except Exception as _se:
                    print(f"[EVAL-SUM] 小總結失敗: {_se}")

            # ── 建議問題（在 done 之前送，確保串流還開著）──
            suggestions = _generate_suggestions(question, "".join(answer_parts))
            if suggestions:
                yield f"data: {json.dumps({'type': 'suggested_questions', 'questions': suggestions}, ensure_ascii=False)}\n\n"

            # ── 儲存對話到 SQLite ──
            _full_answer = "".join(answer_parts)
            if conv_id and user_id:
                try:
                    _now = time.time()
                    with _get_db() as _conn:
                        _conn.execute(
                            "INSERT OR IGNORE INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
                            (conv_id, user_id, question[:30] if question else "新對話", _now, _now)
                        )
                        _existing = _conn.execute("SELECT title FROM conversations WHERE id=?", (conv_id,)).fetchone()
                        if _existing and _existing["title"] == "新對話" and question:
                            _conn.execute("UPDATE conversations SET title=? WHERE id=?", (question[:30], conv_id))
                        _conn.execute(
                            "INSERT OR IGNORE INTO conv_messages (id, conversation_id, role, content, timestamp) VALUES (?,?,?,?,?)",
                            (str(_uuid.uuid4()), conv_id, "user", question, _now - 0.001)
                        )
                        _conn.execute(
                            "INSERT OR IGNORE INTO conv_messages (id, conversation_id, role, content, timestamp) VALUES (?,?,?,?,?)",
                            (str(_uuid.uuid4()), conv_id, "assistant", _full_answer, _now)
                        )
                        _conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now, conv_id))
                except Exception as _e:
                    print(f"[CONV] 儲存對話失敗: {_e}")

            yield f"data: {json.dumps({'type': 'done', 'timing': timing})}\n\n"

            # ── 儲存對話記憶 ──
            if chat_id:
                full_ans = _full_answer
                saved_plans = _listed_schools if _listed_schools else _extract_listed_plans(full_ans)
                with _chat_history_lock:
                    hist = _chat_history.setdefault(chat_id, [])
                    # 多校追問時保留原始清單（而非從新答案重新提取，避免清單縮水）
                    hist.append({"q": question, "a": full_ans[:5000], "plans": saved_plans})
                    if len(hist) > _MAX_HISTORY:
                        hist.pop(0)
                    if len(_chat_history) > 1000:
                        for k in list(_chat_history.keys())[:200]:
                            del _chat_history[k]

            # 建議問題改由前端另外呼叫 /api/suggest，不在主串流裡生成

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


AGENT_PLAN_PROMPT = """針對以下問題，請列出 2~3 個最適合的繁體中文搜尋關鍵字，用來從 USR 計畫書資料庫中找到相關內容。

問題：{question}

只輸出 JSON 陣列，不要其他文字，例如：
["高齡照護 USR 計畫", "青銀共創 大學社會責任", "失智症 照護"]"""


_USR_TOPIC_NAMES = list(USR_TOPIC_KEYWORDS.keys())

# 將 USR 領域詞加入 jieba 詞庫，避免複合詞被拆開（如「循環經濟」→「循環」+「經濟」）
for _kws in USR_TOPIC_KEYWORDS.values():
    for _kw in _kws:
        if len(_kw) >= 3:
            jieba.add_word(_kw)

TOPIC_CLASSIFY_PROMPT = """你是台灣 USR（大學社會責任）計畫的議題分類專家。
請判斷以下問題**最核心**屬於哪些 USR 議題類別。

規則：
- 只選擇與問題**最核心、最直接**相關的一個類別
- 只能選 1 個，不得複選

可選類別：{topics}

問題：{question}

只輸出 JSON 陣列（類別名稱），不要其他文字。例如：["在地關懷"]
若完全沒有對應類別則輸出：[]"""


def _llm_classify_topics(question: str) -> list[str]:
    """用 LLM 語意判斷問題屬於哪些 USR 議題類別，回傳類別名稱清單。"""
    try:
        from langchain_core.messages import HumanMessage
        prompt = TOPIC_CLASSIFY_PROMPT.format(
            topics="、".join(_USR_TOPIC_NAMES),
            question=question
        )
        res = llm_fast.bind(temperature=0, thinking_budget=0).invoke([HumanMessage(content=prompt)])
        try:
            _um = getattr(res, 'usage_metadata', None) or {}
            _log_api_cost("gemini", "classify", _LLM_FAST_MODEL_NAME, _um.get('input_tokens', 0), _um.get('output_tokens', 0))
        except Exception:
            pass
        text = _normalize_content(res.content).strip()
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            topics = json.loads(m.group())
            valid = [t for t in topics if isinstance(t, str) and t in USR_TOPIC_KEYWORDS]
            print(f"[TOPIC_LLM] 語意分類：{valid}")
            return valid
    except Exception as e:
        print(f"[TOPIC_LLM] 失敗：{e}")
    return []

_JIEBA_STOPWORDS: set[str] = {
    '相關', '有關', '計畫', '哪些', '哪幾', '有幾', '多少', '大學', '學校',
    '告訴', '說明', '介紹', '如何', '列出', '請問', '跟', '與', '和', '或',
    '及', '的', '等', '有', '在', '是', '了', '嗎', '呢', '吧', '啊',
    '哪', '什麼', '為何', '為什麼', '怎麼', '怎樣', '幾個', '幾間', '幾所',
    '幾件', '這些', '那些', '所有', '全部', '各', '每個', '請', '問',
    '想', '知道', '了解',
}

def _extract_query_terms(q: str) -> list[str]:
    """從問題用 jieba 分詞提取內容關鍵字，補充 Sequential Query 掃描詞。"""
    result = []
    for w in jieba.cut(q):
        w = w.strip()
        if len(w) >= 2 and w not in _JIEBA_STOPWORDS:
            result.append(w)
    return list(dict.fromkeys(result))



FOLLOWUP_CHECK_PROMPT = """判斷目前問題是否在追問或延伸前一輪問題的查詢結果，只輸出「是」或「否」。

前一輪問題：{prev_q}
目前問題：{curr_q}"""


def _check_is_followup(question: str, prev_turn: dict) -> bool:
    """用 LLM 判斷目前問題是否追問前一輪結果。失敗時回傳 False（保守視為新問題）。"""
    try:
        from langchain_core.messages import HumanMessage
        prev_q = (prev_turn.get('q') or '')[:120]
        if not prev_q:
            return False
        prompt = FOLLOWUP_CHECK_PROMPT.format(prev_q=prev_q, curr_q=question[:120])
        res = llm_fast.bind(temperature=0, thinking_budget=0).invoke([HumanMessage(content=prompt)])
        try:
            _um = getattr(res, 'usage_metadata', None) or {}
            _log_api_cost("gemini", "followup_check", _LLM_FAST_MODEL_NAME, _um.get('input_tokens', 0), _um.get('output_tokens', 0))
        except Exception:
            pass
        text = _normalize_content(res.content).strip()
        result = text.startswith('是')
        print(f"[FOLLOWUP] {'追問' if result else '新問題'}（LLM）：{question[:40]!r}")
        return result
    except Exception as e:
        print(f"[FOLLOWUP] 檢查失敗：{e}")
        return False


SUGGEST_PROMPT = """根據以下 USR 計畫書問答，生成 3 個使用者可能想繼續追問的問題。
要求：具體、繁體中文、每題 25 字以內，圍繞原始問題的延伸或深化方向。
只輸出 JSON 陣列，不要任何其他文字。
範例：["這些計畫的具體成效如何評估？", "有哪些相關學校也在推動類似主題？", "這個領域的未來發展趨勢為何？"]

原始問題：{question}
回答摘要：{answer}"""


def _generate_hl_terms(question: str, answer: str) -> list[str]:
    """從回答中讓 LLM 提取應標記的相關關鍵詞。"""
    from langchain_core.messages import HumanMessage as _HMHl
    try:
        prompt = (
            f"根據以下問題與回答，列出8~12個回答中最重要、值得標記強調的繁體中文關鍵詞或專有名詞。"
            f"包含：核心概念詞、同義詞、重要專業術語。每行一個詞，不要編號或解釋。\n\n"
            f"問題：{question}\n\n回答摘要：{answer[:800]}"
        )
        resp = llm_fast.bind(temperature=0, thinking_budget=0).invoke([_HMHl(content=prompt)])
        text = _normalize_content(resp.content).strip()
        return [t.strip() for t in text.split('\n') if 2 <= len(t.strip()) <= 15][:15]
    except Exception as e:
        print(f"[HL_TERMS] LLM 呼叫失敗: {e}")
        return []


def _generate_suggestions(question: str, answer: str) -> list[str]:
    print(f"[SUGGEST] 開始生成，問題：{question[:30]!r}，答案長度：{len(answer)}")
    try:
        from langchain_core.messages import HumanMessage
        prompt = SUGGEST_PROMPT.format(question=question, answer=answer[:600])
        res = llm_fast.bind(temperature=0.4, thinking_budget=0).invoke([HumanMessage(content=prompt)])
        try:
            _um = getattr(res, 'usage_metadata', None) or {}
            _log_api_cost("gemini", "suggest", _LLM_FAST_MODEL_NAME, _um.get('input_tokens', 0), _um.get('output_tokens', 0))
        except Exception:
            pass
        text = _normalize_content(res.content).strip()
        print(f"[SUGGEST] LLM 原始回傳：{text[:200]!r}")
        m = re.search(r'\[.*?\]', text, re.DOTALL)
        if m:
            items = json.loads(m.group())
            result = [s for s in items if isinstance(s, str)][:3]
            print(f"[SUGGEST] 成功生成 {len(result)} 個建議")
            return result
        print("[SUGGEST] 找不到 JSON 陣列")
    except Exception as e:
        print(f"[SUGGEST] 生成失敗：{e}")
    return []


@app.route("/api/suggest", methods=["POST"])
def api_suggest():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    answer   = (data.get("answer") or "").strip()
    print(f"[SUGGEST API] 收到請求：{question[:40]!r}")
    if not question:
        return jsonify({"questions": []})
    suggestions = _generate_suggestions(question, answer)
    print(f"[SUGGEST API] 回傳 {len(suggestions)} 個建議")
    return jsonify({"questions": suggestions})


_PERSONNEL_RE = re.compile(
    r'人員|成員|團隊|師資|主任|執行秘書|計畫主持|共同主持|協同主持|老師|教授|誰|姓名|人名'
)

_LIST_INTENT_RE  = re.compile(
    r'哪些|有哪|哪幾|列出'
    r'|所有.{0,6}(?:學校|大學|院校|計畫|計画)'
    r'|全部.{0,4}(?:學校|大學|院校|計畫)'
)
# 哪些後面跟概念詞（策略/方法/影響等）→ 不是列舉學校，不觸發列舉模式
_LIST_CONCEPT_RE = re.compile(
    r'哪些.{0,12}(?:策略|方法|做法|方式|方向|措施|途徑|建議|影響|成效|特色|問題|挑戰|困難|原因|因素|差異|優點|缺點|優缺)'
)
_QUESTION_WORDS  = {"哪些", "有哪", "哪幾", "什麼", "怎麼", "如何", "是否", "有沒有",
                    "請問", "告訴我", "介紹", "說明", "哪個", "哪裡", "為何", "為什麼",
                    "幾個", "幾間", "多少", "列出", "有關", "相關", "跟", "和", "與"}

# ── 學校縣市對照表 ──
_SCHOOL_COUNTY: dict[str, str] = {
    # 台北
    '大同大學': '台北', '中國文化大學': '台北', '中國科技大學': '台北',
    '東吳大學': '台北', '馬偕醫護管理專科學校': '台北', '台北海洋科技大學': '台北',
    '國立臺北科技大學': '台北', '國立臺北護理健康大學': '台北',
    '國立臺灣大學': '台北', '國立臺灣科技大學': '台北', '國立臺灣師範大學': '台北',
    '國立陽明交通大學': '台北', '銘傳大學': '台北', '德明財經科技大學': '台北',
    '臺北市立大學': '台北', '臺北醫學大學': '台北', '實踐大學': '台北',
    # 新北
    '明志科技大學': '新北', '東南科技大學': '新北', '法鼓文理學院': '新北',
    '致理科技大學': '新北', '耕莘健康管理專科學校': '新北', '馬偕醫學大學': '新北',
    '國立臺北大學': '新北', '亞東科技大學': '新北', '淡江大學': '新北',
    '華梵大學': '新北', '輔仁大學': '新北',
    # 基隆
    '國立臺灣海洋大學': '基隆', '經國管理學院': '基隆', '德育護理健康學院': '基隆',
    # 桃園
    '長庚大學': '桃園', '長庚科技大學': '桃園', '健行科技大學': '桃園',
    '國立中央大學': '桃園', '元智大學': '桃園', '中原大學': '桃園',
    '開南大學': '桃園', '龍華科技大學': '桃園', '敏實科技大學': '桃園',
    '萬能科技大學': '桃園',
    # 新竹
    '中華大學': '新竹', '明新科技大學': '新竹', '元培醫事科技大學': '新竹',
    '玄奘大學': '新竹', '國立清華大學': '新竹',
    # 苗栗
    '育達科技大學': '苗栗', '國立聯合大學': '苗栗',
    # 宜蘭
    '國立宜蘭大學': '宜蘭', '佛光大學': '宜蘭',
    # 台中
    '中臺科技大學': '台中', '中國醫藥大學': '台中', '弘光科技大學': '台中',
    '東海大學': '台中', '逢甲大學': '台中', '朝陽科技大學': '台中',
    '靜宜大學': '台中', '嶺東科技大學': '台中', '亞洲大學': '台中',
    '僑光科技大學': '台中', '國立中興大學': '台中', '國立勤益科技大學': '台中',
    '國立臺中科技大學': '台中', '國立臺中教育大學': '台中',
    # 彰化
    '大葉大學': '彰化', '建國科技大學': '彰化', '國立彰化師範大學': '彰化',
    # 南投
    '南開科技大學': '南投', '國立暨南國際大學': '南投',
    # 雲林
    '國立虎尾科技大學': '雲林', '國立雲林科技大學': '雲林',
    # 嘉義
    '南華大學': '嘉義', '國立中正大學': '嘉義', '國立嘉義大學': '嘉義',
    # 台南
    '中華醫事科技大學': '台南', '中信科技大學': '台南', '台南應用科技大學': '台南',
    '長榮大學': '台南', '南臺科技大學': '台南', '國立成功大學': '台南',
    '國立臺南大學': '台南', '國立臺南藝術大學': '台南', '崑山科技大學': '台南',
    '嘉南藥理大學': '台南',
    # 高雄
    '文藻外語大學': '高雄', '正修科技大學': '高雄', '高雄醫學大學': '高雄',
    '國立中山大學': '高雄', '國立高雄大學': '高雄', '國立高雄科技大學': '高雄',
    '國立高雄師範大學': '高雄', '國立高雄餐旅大學': '高雄', '樹德科技大學': '高雄',
    '義守大學': '高雄', '輔英科技大學': '高雄',
    # 屏東
    '大仁科技大學': '屏東', '美和科技大學': '屏東',
    '國立屏東大學': '屏東', '國立屏東科技大學': '屏東',
    # 花蓮
    '國立東華大學': '花蓮', '慈濟大學': '花蓮',
    # 台東
    '國立臺東大學': '台東',
    # 離島
    '國立金門大學': '金門', '國立澎湖科技大學': '澎湖',
}

# 大區域 / 縣市群縮寫 → 縣市集合
_REGION_TO_COUNTIES: dict[str, list[str]] = {
    '北部':   ['台北', '新北', '基隆', '桃園', '新竹', '苗栗', '宜蘭'],
    '北台灣': ['台北', '新北', '基隆', '桃園', '新竹', '苗栗', '宜蘭'],
    '北臺灣': ['台北', '新北', '基隆', '桃園', '新竹', '苗栗', '宜蘭'],
    '中部':   ['台中', '彰化', '南投', '雲林', '苗栗'],
    '中台灣': ['台中', '彰化', '南投', '雲林', '苗栗'],
    '中臺灣': ['台中', '彰化', '南投', '雲林', '苗栗'],
    '南部':   ['嘉義', '台南', '高雄', '屏東'],
    '南台灣': ['嘉義', '台南', '高雄', '屏東'],
    '南臺灣': ['嘉義', '台南', '高雄', '屏東'],
    '東部':   ['花蓮', '台東', '宜蘭'],
    '東台灣': ['花蓮', '台東', '宜蘭'],
    '東臺灣': ['花蓮', '台東', '宜蘭'],
    '離島':   ['金門', '澎湖', '馬祖'],
    # 常見縣市群縮寫
    '北北基': ['台北', '新北', '基隆'],
    '桃竹苗': ['桃園', '新竹', '苗栗'],
    '中彰投': ['台中', '彰化', '南投'],
    '雲嘉南': ['雲林', '嘉義', '台南'],
    '雲嘉':   ['雲林', '嘉義'],
    '高屏':   ['高雄', '屏東'],
    '南高屏': ['台南', '高雄', '屏東'],
    '宜花東': ['宜蘭', '花蓮', '台東'],
    '宜花':   ['宜蘭', '花蓮'],
    '花東':   ['花蓮', '台東'],
    '桃竹苗宜花': ['桃園', '新竹', '苗栗', '宜蘭', '花蓮'],
}

# 個別縣市關鍵字 → 縣市名稱
_COUNTY_KEYWORDS: dict[str, str] = {
    '台北': '台北', '臺北': '台北', '新北': '新北', '基隆': '基隆',
    '桃園': '桃園', '新竹': '新竹', '苗栗': '苗栗', '宜蘭': '宜蘭',
    '台中': '台中', '臺中': '台中', '彰化': '彰化', '南投': '南投', '雲林': '雲林',
    '嘉義': '嘉義', '台南': '台南', '臺南': '台南',
    '高雄': '高雄', '屏東': '屏東',
    '花蓮': '花蓮', '台東': '台東', '臺東': '台東',
    '金門': '金門', '澎湖': '澎湖', '馬祖': '馬祖',
}

def _get_school_county(school_name: str) -> str:
    if school_name in _SCHOOL_COUNTY:
        return _SCHOOL_COUNTY[school_name]
    for k, v in _SCHOOL_COUNTY.items():
        if k in school_name or school_name in k:
            return v
    return ''

def _detect_question_counties(q: str) -> set[str]:
    """從問題偵測地區，回傳縣市集合；空集合表示不過濾。"""
    counties: set[str] = set()
    # 先偵測縣市群縮寫（長的先，避免「宜花」被「宜蘭」吃掉）
    for abbrev in sorted(_REGION_TO_COUNTIES, key=len, reverse=True):
        if abbrev in q:
            counties.update(_REGION_TO_COUNTIES[abbrev])
    # 再偵測個別縣市名稱
    for kw, county in _COUNTY_KEYWORDS.items():
        if kw in q:
            counties.add(county)
    return counties

# 保留舊名供舊程式碼相容
_SCHOOL_REGION = {k: '' for k in _SCHOOL_COUNTY}
def _get_school_region(school_name: str) -> str:
    return _get_school_county(school_name)
def _detect_question_region(q: str) -> str:
    return 'any' if _detect_question_counties(q) else ''

_EVAL_RE = re.compile(
    # 「最」字系列：明確排名意圖
    r'最(?:好|棒|佳|優|優秀|值得|推薦|有效|成功|傑出|具代表|有特色|突出|厲害|強大|重要)'
    r'|哪(?:個|間|所|家|些).{0,8}(?:最|比較好|更好|較好|第一)'
    r'|(?:最好|最佳|最優|最強)的.{0,6}(?:計畫|學校|大學|做法|方案|案例)'
    r'|排名|排行|第一名|冠軍|名次|勝出|較優|更優'
    r'|誰做得比較好|哪個比較好|哪個更好|哪個較好|哪間做得好'
    # 「哪些學校有 + 主觀品質形容詞」：需釐清何謂「完整/完善」
    r'|哪.{0,12}(?:完整|完善|健全|完備|齊全|周全|良好|優良|成熟|豐富|積極|全面|深入|扎實|紮實|有效|到位)(?:的|之).{0,15}(?:制度|機制|措施|政策|規劃|計畫|方案|做法|配套|支持|體系|系統)'
)

_CLARIFY_SYSTEM_PROMPT = """你是 USR（大學社會責任）計畫搜尋助理。使用者的問題含有排名或主觀評量詞，需要先釐清評估標準，才能給出有意義的回答。

常見情況：
含主觀評量詞（如「最好」「完整」「完善」「排名」）→ 不同標準有不同答案

請根據問題的**具體主題**，提出 3～5 個與該主題直接相關的評估面向或操作角度，格式如下：
- 第一句說明為何需要釐清（針對問題主題）
- 面向清單（每項「• 面向：一句說明」）
- 最後一句邀請使用者指定角度，並說明可切換「接續上題」模式繼續提問

禁止使用與問題無關的通用面向。只輸出給使用者看的文字，不加任何前綴或解釋。"""


def _generate_clarify_msg(question: str) -> str:
    """根據問題內容動態生成針對性的澄清訊息。"""
    from langchain_core.messages import HumanMessage as _HMClarify
    try:
        prompt = f"{_CLARIFY_SYSTEM_PROMPT}\n\n問題：{question}"
        print(f"[CLARIFY] 呼叫 LLM 生成澄清訊息，問題：{question[:50]}")
        resp = llm_fast.bind(temperature=0, thinking_budget=0).invoke([_HMClarify(content=prompt)])
        result = _normalize_content(resp.content).strip()
        print(f"[CLARIFY] LLM 回傳：{result[:80]}")
        return result
    except Exception as e:
        print(f"[CLARIFY] LLM 呼叫失敗: {e}")
        return "您的問題涉及主觀判斷，請描述您希望依據哪些面向評估，我就能針對您的標準為您做出分析！"


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


def _school_filter_docs(docs, entry: str, k: int):
    """過濾含學校名稱（及計畫名前綴）的 chunk，無結果則退回全部。
    entry 可以是 '學校名' 或 '學校名：計畫名' 格式。
    """
    parts = entry.split('：', 1)
    school = parts[0]
    plan_prefix = parts[1][:8] if len(parts) > 1 else ""
    src = lambda d: d.metadata.get("source", "")
    if plan_prefix:
        filtered = [d for d in docs if school in src(d) and plan_prefix in src(d)]
        if not filtered:  # plan 前綴比對失敗則退回學校名比對
            filtered = [d for d in docs if school in src(d)]
    else:
        filtered = [d for d in docs if school in src(d)]
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
    _ensure_year_loaded(year)
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


def _sanitize_prompt_input(text: str) -> str:
    """移除控制字元，防止 prompt injection 透過不可見字元操控 LLM。"""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

def _normalize_content(content):
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content


@app.route("/subagent", methods=["POST"])
def subagent_ask():
    if SITE_PASSWORD and not session.get("authenticated"):
        return jsonify({"error": "請先登入。"}), 401

    data = request.get_json(silent=True) or {}
    year = (data.get("year") or "114").strip()
    if year not in ("113", "114"):
        year = "114"
    _ensure_year_loaded(year)
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
                _clarify_msg = _generate_clarify_msg(question)
                yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': _clarify_msg}, ensure_ascii=False)}\n\n"
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
                    res = llm_fast.bind(temperature=0, thinking_budget=0).invoke([_plan_msg])
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
                cut = context[:20000].rfind("\n\n")
                context = context[:cut if cut > 10000 else 20000] + "\n\n...(資料已截斷)"
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
    global vectorstores

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
