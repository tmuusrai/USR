import os
import sys
import json
import time
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

load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(32))

# ── 設定 ──────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
SITE_USERNAME   = os.getenv("SITE_USERNAME", "")
SITE_PASSWORD   = os.getenv("SITE_PASSWORD", "")
VOYAGE_API_KEY  = os.getenv("VOYAGE_API_KEY")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K           = int(os.getenv("TOP_K_RESULTS", 5))

PDF_DIR         = Path("pdfs")
EXTRA_DIR       = Path("extra_docs")
INDEX_DIR       = Path("faiss_index")

# ── Embedding 模型（全域共用，避免重複初始化）──────────
class _CachedEmbeddings:
    """Query embedding 快取，相同問題不重複呼叫 Voyage AI。"""
    def __init__(self, base):
        self._base  = base
        self._cache = {}

    def embed_query(self, text):
        if text not in self._cache:
            self._cache[text] = self._base.embed_query(text)
        return self._cache[text]

    def __call__(self, text):
        return self.embed_query(text)

    def embed_documents(self, texts):
        return self._base.embed_documents(texts)

    def __getattr__(self, name):
        return getattr(self._base, name)

_base_embeddings = VoyageAIEmbeddings(
    voyage_api_key=VOYAGE_API_KEY,
    model="voyage-3",
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
- 回答請使用繁體中文，條理清晰。

回答：""",
)


def load_or_build_index() -> FAISS:
    """載入既有索引；若不存在則從 pdfs/ 重新建立。"""
    index_file = INDEX_DIR / "index.faiss"

    if index_file.exists():
        print("[INDEX] 載入既有 FAISS 索引...")
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("[INDEX] 未找到索引，開始從 PDF 建立...")
    pdf_files = list(PDF_DIR.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"pdfs/ 資料夾中沒有 PDF 檔案，請先放入計畫書。")

    docs = []
    for pdf_path in pdf_files:
        print(f"  讀取：{pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    if EXTRA_DIR.exists():
        for txt_path in EXTRA_DIR.rglob("*.txt"):
            print(f"  讀取補充文件：{txt_path.name}")
            loader = TextLoader(str(txt_path), encoding="utf-8")
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    total = len(chunks)
    print(f"[INDEX] 共切出 {total} 個段落，開始向量化...", flush=True)

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

    INDEX_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print("[INDEX] 索引建立完成並已儲存。", flush=True)
    return vectorstore


# ── 啟動時初始化 ──────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

try:
    vectorstore = load_or_build_index()
    retriever   = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    print("[APP] RAG 系統就緒。")
except FileNotFoundError as e:
    vectorstore = None
    retriever   = None
    print(f"[APP] 警告：{e}")


# ── 路由 ──────────────────────────────────────────────
@app.route("/")
def index():
    authenticated = not SITE_PASSWORD or session.get("authenticated", False)
    return render_template("index.html", authenticated=authenticated)


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username_ok = not SITE_USERNAME or data.get("username") == SITE_USERNAME
    password_ok = not SITE_PASSWORD or data.get("password") == SITE_PASSWORD
    if username_ok and password_ok:
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

    if not question:
        return jsonify({"error": "請輸入問題。"}), 400
    if len(question) > 500:
        return jsonify({"error": "問題不得超過 500 字。"}), 400

    def generate():
        try:
            t0 = time.perf_counter()

            # ① Voyage AI：將問題向量化
            query_vec = embeddings.embed_query(question)
            t_voyage = time.perf_counter()

            # ② FAISS：向量搜尋
            docs = vectorstore.similarity_search_by_vector(query_vec, k=TOP_K)
            t_faiss = time.perf_counter()

            context = "\n\n".join(doc.page_content for doc in docs)
            prompt_value = RAG_PROMPT.invoke({"context": context, "question": question})

            sources = []
            seen = set()
            for doc in docs:
                meta = doc.metadata
                src  = Path(meta.get("source", "")).name
                page = meta.get("page", 0) + 1
                if (src, page) not in seen:
                    seen.add((src, page))
                    sources.append({"source": src, "page": page})

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

            # ③ Gemini：串流生成
            answer_chars = 0
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
                "gemini_first_ms": round((t_first_chunk - t_faiss) * 1000),
                "gemini_total_ms": round((t_end - t_faiss) * 1000),
                "total_ms":        round((t_end - t0) * 1000),
            }
            print(
                f"[TIMING] Voyage={timing['voyage_ms']}ms"
                f" | FAISS={timing['faiss_ms']}ms"
                f" | Gemini首字={timing['gemini_first_ms']}ms"
                f" | Gemini完成={timing['gemini_total_ms']}ms"
                f" | 總計={timing['total_ms']}ms"
            )

            yield f"data: {json.dumps({'type': 'done', 'timing': timing})}\n\n"

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
REACT_SYSTEM_PROMPT = """你是 USR 計畫書自主研究助理，專門處理複雜的跨校比較與趨勢分析。

你可以使用以下工具：
- search_rag(query, k): 搜尋 USR 計畫書資料庫，k 為回傳筆數（預設5）

請嚴格依照以下格式逐步推理：

Thought: 分析問題，決定需要搜尋什麼
Action: search_rag
Action Input: {"query": "具體搜尋詞", "k": 5}

收到 Observation 後繼續推理，直到資料足夠時：
Thought: 我已有足夠資訊
Final Answer: [完整、有條理的分析回答]

規則：
- 複雜的跨校比較或趨勢分析問題，至少搜尋 2-3 次再給答案
- 每次只能執行一個 Action
- Final Answer 必須整合所有 Observation 的資訊，用繁體中文回答
"""


def tool_search_rag(query: str, k: int = 5):
    """回傳 (觀察文字, sources列表)"""
    vec = embeddings.embed_query(query)
    docs = vectorstore.similarity_search_by_vector(vec, k=k)
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


def _force_final_answer(messages):
    """當 Gemini 不按格式時，強制取得最終答案。"""
    from langchain_core.messages import HumanMessage
    msgs = list(messages) + [
        HumanMessage(content="請根據以上所有搜尋結果，直接用繁體中文給出完整的分析回答，不需要再搜尋。")
    ]
    response = llm.invoke(msgs)
    text = _normalize_content(response.content)
    if "Final Answer:" in text:
        return text.split("Final Answer:")[-1].strip()
    return text


def react_agent_stream(question: str, max_steps: int = 5):
    """Generator：每完成一步就 yield，避免 SSE 連線閒置斷線。"""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    messages = [
        SystemMessage(content=REACT_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    all_sources = []
    seen_sources = set()
    searched = False

    for step in range(max_steps):
        response = llm.invoke(messages)
        text = _normalize_content(response.content)

        if "Final Answer:" in text:
            yield "sources", all_sources
            yield "answer", text.split("Final Answer:")[-1].strip()
            return

        if "Action: search_rag" in text and "Action Input:" in text:
            try:
                raw = text.split("Action Input:")[-1].strip().split("\n")[0]
                params = json.loads(raw)
                query = params.get("query", question)
                k = int(params.get("k", 5))
                preview = f"搜尋：{query}"
            except Exception:
                query = question
                k = 5
                preview = f"搜尋：{query[:80]}"
            yield "step", {"step": step + 1, "preview": preview}
            try:
                observation, sources = tool_search_rag(query, k)
                searched = True
                for s in sources:
                    key = (s["source"], s["page"])
                    if key not in seen_sources:
                        seen_sources.add(key)
                        all_sources.append(s)
            except Exception as e:
                observation = f"工具執行失敗：{e}"
            messages.append(AIMessage(content=text))
            messages.append(HumanMessage(content=f"Observation:\n{observation}"))
        else:
            # Gemini 沒照格式，若已有搜尋結果就強制取最終答案
            if searched:
                yield "sources", all_sources
                yield "answer", _force_final_answer(messages)
            else:
                yield "sources", all_sources
                yield "answer", text
            return

    # 達到最大步驟，強制總結
    yield "sources", all_sources
    yield "answer", _force_final_answer(messages)


@app.route("/agent", methods=["POST"])
def agent_ask():
    if SITE_PASSWORD and not session.get("authenticated"):
        return jsonify({"error": "請先登入。"}), 401

    if vectorstore is None:
        return jsonify({"error": "索引尚未建立。"}), 503

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "請輸入問題。"}), 400
    if len(question) > 500:
        return jsonify({"error": "問題不得超過 500 字。"}), 400

    def generate():
        try:
            t_agent_start = time.perf_counter()
            yield f"data: {json.dumps({'type': 'status', 'text': '🤖 Agent 模式啟動，第一步：分析問題（約 5-10 秒）...'}, ensure_ascii=False)}\n\n"
            step_count = 0
            for event_type, data in react_agent_stream(question):
                if event_type == "step":
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
        except Exception as e:
            import traceback
            print(f"[ERROR] /agent：{e}\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'error': '分析時發生錯誤，請稍後再試。'}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/rebuild-index", methods=["POST"])
def rebuild_index():
    """重新建立索引（上傳新 PDF 後呼叫）。"""
    global vectorstore, retriever

    # 刪除舊索引
    for f in INDEX_DIR.glob("*"):
        f.unlink()

    try:
        vectorstore = load_or_build_index()
        retriever   = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )
        return jsonify({"message": "索引重建完成。"})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"[ERROR] /rebuild-index：{e}")
        return jsonify({"error": "索引重建失敗。"}), 500


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
    return jsonify({
        "ready":      retriever is not None,
        "pdf_count":  pdf_count,
        "index_ready": index_ready,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", port=port)
