import os
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, Response, stream_with_context
from flask_cors import CORS

from langchain_community.document_loaders import PyPDFLoader
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
            docs = retriever.invoke(question)
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

            last_chunk = None
            for chunk in llm.stream(prompt_value):
                content = chunk.content
                if isinstance(content, list):
                    content = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )
                if content:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': content}, ensure_ascii=False)}\n\n"
                last_chunk = chunk

            if last_chunk and hasattr(last_chunk, "usage_metadata") and last_chunk.usage_metadata:
                u = last_chunk.usage_metadata
                print(f"[TOKEN] input={u.get('input_tokens')} output={u.get('output_tokens')} total={u.get('total_tokens')}")

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

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
