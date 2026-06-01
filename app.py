import os
import sys
import json
import time
import queue as _queue
import threading
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
app.secret_key = os.getenv("SECRET_KEY", os.urandom(32))

# ── 設定 ──────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
SITE_USERNAME   = os.getenv("SITE_USERNAME", "")
SITE_PASSWORD   = os.getenv("SITE_PASSWORD", "")
VOYAGE_API_KEY  = os.getenv("VOYAGE_API_KEY")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K           = int(os.getenv("TOP_K_RESULTS", 10))

PDF_DIR         = Path("pdfs")
EXTRA_DIR       = Path("extra_docs")
TXT_DIR         = Path("114txt")
INDEX_DIR       = Path("faiss_index")

# ── Embedding 模型（全域共用，避免重複初始化）──────────
class _CachedEmbeddings(Embeddings):
    """Query embedding 快取，相同問題不重複呼叫 Voyage AI。"""
    def __init__(self, base):
        self._base  = base
        self._cache = {}

    def embed_query(self, text: str) -> list:
        if text not in self._cache:
            self._cache[text] = self._base.embed_query(text)
        return self._cache[text]

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

    print("[INDEX] 未找到索引，開始建立...")
    pdf_files = list(PDF_DIR.rglob("*.pdf")) if PDF_DIR.exists() else []
    txt_files_check = list(TXT_DIR.rglob("*.txt")) if TXT_DIR.exists() else []
    if not pdf_files and not txt_files_check:
        raise FileNotFoundError(f"pdfs/ 和 114txt/ 資料夾中都沒有檔案，請先放入計畫書。")

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

    if TXT_DIR.exists():
        txt_files = list(TXT_DIR.rglob("*.txt"))
        print(f"  114txt/ 資料夾：找到 {len(txt_files)} 個 TXT 檔")
        for txt_path in txt_files:
            print(f"  讀取：{txt_path.name}")
            try:
                loader = TextLoader(str(txt_path), encoding="utf-8")
                docs.extend(loader.load())
            except Exception as e:
                print(f"  [WARN] 跳過 {txt_path.name}：{e}")

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
    model=os.getenv("LLM_MODEL", "gemini-2.5-pro-preview"),
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

init_qa(TXT_DIR)


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

            # ① 結構化 QA 優先（列舉型問題不走 FAISS）
            structured_ctx = try_structured_answer(question)
            if structured_ctx:
                t_voyage = t_faiss = time.perf_counter()
                docs = []
                context = structured_ctx
            else:
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

            # ③ LLM：串流生成
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

請用繁體中文給出完整、有條理的分析回答，整合所有資料內容。"""


def tool_search_rag(query: str, k: int = 5):
    """回傳 (觀察文字, sources列表)"""
    print(f"[TOOL] 搜尋：{query[:60]}  vectorstore={'OK' if vectorstore else 'None'}")
    vec = embeddings.embed_query(query)
    print(f"[TOOL] embed 完成，vec長度={len(vec)}")
    docs = vectorstore.similarity_search_by_vector(vec, k=k)
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


def react_agent_stream(question: str, max_steps: int = 5):
    """固定 2 步架構：Gemini 規劃搜尋詞 → 執行搜尋 → Gemini 整合回答。"""
    import re
    from langchain_core.messages import HumanMessage

    # ── 結構化 QA 短路：列舉型問題直接回傳，不呼叫 FAISS ──
    structured_ctx = try_structured_answer(question)
    if structured_ctx:
        yield "sources", []
        yield "answer", structured_ctx
        return

    all_sources = []
    seen_sources = set()

    # ── 步驟 1：讓 Gemini 決定要搜尋什麼 ──
    yield "heartbeat", None
    try:
        plan_res = llm.invoke([HumanMessage(content=AGENT_PLAN_PROMPT.format(question=question))])
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
            observation, sources = tool_search_rag(str(query), k=5)
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
    if vectorstore is None:
        return jsonify({"error": "索引尚未建立。"}), 503

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "請輸入問題。"}), 400
    if len(question) > 500:
        return jsonify({"error": "問題不得超過 500 字。"}), 400

    def generate():
        import re
        from langchain_core.messages import HumanMessage

        try:
            t_start = time.perf_counter()

            # ── 結構化 QA 短路 ──
            structured_ctx = try_structured_answer(question)
            if structured_ctx:
                yield f"data: {json.dumps({'type': 'sources', 'sources': []}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'chunk', 'text': structured_ctx}, ensure_ascii=False)}\n\n"
                total_ms = round((time.perf_counter() - t_start) * 1000)
                yield f"data: {json.dumps({'type': 'done', 'timing': {'total_ms': total_ms}, 'mode': 'structured'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'status', 'text': '🔍 分析問題，規劃 subagent 搜尋策略...'}, ensure_ascii=False)}\n\n"

            # ── 步驟 1：規劃搜尋詞（背景執行緒 + keepalive，應對 Gemini 思考階段）──
            plan_q = _queue.Queue()
            _plan_msg = HumanMessage(content=AGENT_PLAN_PROMPT.format(question=question))
            def _plan_worker():
                try:
                    res = llm.invoke([_plan_msg])
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
            yield f"data: {json.dumps({'type': 'status', 'text': f'⚡ 啟動 {n} 個 subagent，依序執行搜尋...'}, ensure_ascii=False)}\n\n"

            # ── 步驟 2：依序執行各 subagent，每完成一個立刻回報 ──
            all_context = []
            all_sources = []
            seen_sources = set()
            t_search_start = time.perf_counter()

            for i, query in enumerate(queries):
                try:
                    obs, srcs = tool_search_rag(query, k=5)
                    all_context.append(f"【Subagent {i+1}：{query}】\n{obs}")
                    for s in srcs:
                        key = (s["source"], s["page"])
                        if key not in seen_sources:
                            seen_sources.add(key)
                            all_sources.append(s)
                except Exception as e:
                    print(f"[SUBAGENT] subagent {i+1} 失敗：{e}")
                    all_context.append(f"【Subagent {i+1}：{query}】\n查無結果")
                yield f"data: {json.dumps({'type': 'step', 'step': i+1, 'preview': query[:70]}, ensure_ascii=False)}\n\n"

            t_search_ms = round((time.perf_counter() - t_search_start) * 1000)
            yield f"data: {json.dumps({'type': 'sources', 'sources': all_sources}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'status', 'text': '🧠 整合所有 subagent 結果，生成回答...'}, ensure_ascii=False)}\n\n"

            # ── 步驟 3：背景執行緒串流（每 5s 發 keepalive，應對 Gemini 思考階段）──
            context = "\n\n".join(all_context)
            if len(context) > 10000:
                context = context[:10000] + "\n\n...(資料已截斷)"
            prompt_msg = HumanMessage(content=AGENT_ANSWER_PROMPT.format(
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
            while True:
                try:
                    kind, payload = cq.get(timeout=5)
                    if kind == 'chunk':
                        answer_started = True
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


# ── Knowledge Map ─────────────────────────────────────
_km_cache = None

def _build_km_data():
    import re
    SDG_NAMES = {
        1:"消除貧窮",2:"零飢餓",3:"良好健康與福祉",4:"優質教育",
        5:"性別平等",6:"乾淨用水及衛生",7:"可負擔及乾淨能源",
        8:"合宜工作與經濟成長",9:"產業創新與基礎設施",10:"減少不平等",
        11:"永續城市及社區",12:"負責任的消費及生產",13:"氣候行動",
        14:"水下生物",15:"陸地生物",16:"和平正義與強大機構",17:"全球夥伴關係",
    }
    TOPICS = ["在地關懷","環境永續","產業鏈結與經濟永續","健康促進與食品安全","文化永續","其他社會實踐"]
    CATEGORIES = ["大學特色類萌芽型","大學特色類深耕型","永續發展類國際合作型","永續發展類特色永續型"]

    nodes = {}
    links = []

    for num, name in SDG_NAMES.items():
        nid = f"sdg_{num}"
        nodes[nid] = {"id": nid, "type": "sdg", "sdg_num": num, "label": f"SDG{num}", "name": name, "count": 0}
    for t in TOPICS:
        nid = f"topic_{t}"
        nodes[nid] = {"id": nid, "type": "topic", "label": t, "count": 0}

    if not TXT_DIR.exists():
        return {"nodes": list(nodes.values()), "links": []}

    for filepath in sorted(TXT_DIR.glob("*.txt")):
        stem = filepath.stem.strip()
        if "計劃總覽" in stem:
            continue
        if "_" in stem:
            uni, rest = stem.split("_", 1)
            plan = re.sub(r'\s*\([\w\-]+\)\s*', ' ', rest).strip()
            plan = re.sub(r'\s*\(\d+\)\s*$', '', plan).strip()
        else:
            uni, plan = stem, stem

        text = None
        for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
            try:
                text = filepath.read_text(encoding=enc)
                break
            except Exception:
                continue
        if text is None:
            text = filepath.read_text(encoding="utf-8-sig", errors="ignore")

        sdg_nums = set()
        topics = set()
        category = ""
        for line in text.split("\n"):
            if "SDG" in line and "關聯" in line:
                for m in re.finditer(r'(?<!\d)(\d{1,2})(?!\d)', line):
                    n = int(m.group(1))
                    if 1 <= n <= 17:
                        sdg_nums.add(n)
            if "計畫議題" in line:
                for t in TOPICS:
                    if t in line:
                        topics.add(t)
            if "計畫類別" in line and not category:
                for c in CATEGORIES:
                    if c in line:
                        category = c
                        break

        if not sdg_nums and not topics:
            continue

        safe_stem = re.sub(r'[^\w]', '_', stem)
        pid = f"plan_{safe_stem}"
        nodes[pid] = {
            "id": pid, "type": "university",
            "label": uni, "plan": plan,
            "category": category,
            "sdgs": sorted(sdg_nums),
            "topics": list(topics),
        }
        for n in sdg_nums:
            links.append({"source": pid, "target": f"sdg_{n}"})
            nodes[f"sdg_{n}"]["count"] += 1
        for t in topics:
            links.append({"source": pid, "target": f"topic_{t}"})
            nodes[f"topic_{t}"]["count"] += 1

    return {"nodes": list(nodes.values()), "links": links}


@app.route("/knowledge-map")
def knowledge_map():
    authenticated = not SITE_PASSWORD or session.get("authenticated", False)
    if SITE_PASSWORD and not authenticated:
        return redirect(url_for("index"))
    return render_template("knowledge_map.html")


@app.route("/api/knowledge-map-data")
def knowledge_map_data():
    global _km_cache
    if _km_cache is None:
        _km_cache = _build_km_data()
    return jsonify(_km_cache)


_kw_cache = None

KEYWORD_GROUPS = [
    ("高齡長照",   ["高齡", "長照", "失智", "銀髮"]),
    ("兒童青少年", ["青少年", "兒童", "青年", "學童"]),
    ("原住民族",   ["原住民", "部落", "原鄉"]),
    ("食農農業",   ["食農", "農業", "農村", "農產"]),
    ("生態環境",   ["生態", "生物多樣", "自然保育"]),
    ("海洋水資源", ["海洋", "海岸", "漁業", "濕地", "水資源"]),
    ("數位科技",   ["數位", "智慧", "VR", "AR", "資訊科技"]),
    ("文化傳承",   ["文化資產", "傳統技藝", "工藝", "文化保存"]),
    ("社區營造",   ["社區營造", "地方創生", "社造"]),
    ("偏鄉教育",   ["偏鄉", "偏遠地區", "教育資源"]),
    ("健康醫療",   ["健康促進", "醫療", "預防醫學"]),
    ("淨零減碳",   ["淨零", "減碳", "碳中和", "節能", "再生能源"]),
    ("產業創業",   ["產業輔導", "創業", "就業", "職能"]),
    ("新住民移工", ["新住民", "移工", "外籍"]),
    ("身障融合",   ["身障", "障礙", "融合"]),
    ("藝術人文",   ["藝術", "表演藝術", "藝文"]),
    ("防災韌性",   ["防災", "韌性城市", "災害"]),
    ("國際交流",   ["國際合作", "跨國", "海外"]),
]
KW_COLORS = [
    "#f97316","#06b6d4","#8b5cf6","#84cc16","#10b981","#0ea5e9",
    "#6366f1","#f59e0b","#ec4899","#14b8a6","#ef4444","#22c55e",
    "#a855f7","#f43f5e","#475569","#d946ef","#fb923c","#2dd4bf",
]


def _build_kw_data():
    import re
    nodes = {}
    links = []

    for i, (kw_name, _) in enumerate(KEYWORD_GROUPS):
        nid = f"kw_{kw_name}"
        nodes[nid] = {
            "id": nid, "type": "keyword",
            "label": kw_name,
            "color": KW_COLORS[i % len(KW_COLORS)],
            "count": 0,
        }

    if not TXT_DIR.exists():
        return {"nodes": list(nodes.values()), "links": []}

    for filepath in sorted(TXT_DIR.glob("*.txt")):
        stem = filepath.stem.strip()
        if "計劃總覽" in stem:
            continue
        if "_" in stem:
            uni, rest = stem.split("_", 1)
            plan = re.sub(r'\s*\([\w\-]+\)\s*', ' ', rest).strip()
            plan = re.sub(r'\s*\(\d+\)\s*$', '', plan).strip()
        else:
            uni, plan = stem, stem

        text = None
        for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
            try:
                text = filepath.read_text(encoding=enc)
                break
            except Exception:
                continue
        if text is None:
            text = filepath.read_text(encoding="utf-8-sig", errors="ignore")

        matched = []
        for kw_name, keywords in KEYWORD_GROUPS:
            if any(kw in text for kw in keywords):
                matched.append(kw_name)

        if not matched:
            continue

        safe_stem = re.sub(r'[^\w]', '_', stem)
        pid = f"plan_{safe_stem}"
        nodes[pid] = {
            "id": pid, "type": "university",
            "label": uni, "plan": plan,
            "keywords": matched,
        }
        for kw_name in matched:
            nid = f"kw_{kw_name}"
            links.append({"source": pid, "target": nid})
            nodes[nid]["count"] += 1

    return {"nodes": list(nodes.values()), "links": links}


@app.route("/api/keyword-map-data")
def keyword_map_data():
    global _kw_cache
    if _kw_cache is None:
        _kw_cache = _build_kw_data()
    return jsonify(_kw_cache)


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
