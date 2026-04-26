import os
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── 設定 ──────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
VOYAGE_API_KEY  = os.getenv("VOYAGE_API_KEY")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K           = int(os.getenv("TOP_K_RESULTS", 5))

PDF_DIR         = Path("pdfs")
INDEX_DIR       = Path("faiss_index")

# ── Embedding 模型（全域共用，避免重複初始化）──────────
embeddings = VoyageAIEmbeddings(
    voyage_api_key=VOYAGE_API_KEY,
    model="voyage-3",
)

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


def build_qa_chain(vectorstore: FAISS):
    """建立 RAG chain（LCEL）。"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    def _run(input_dict):
        query = input_dict["query"]
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt_value = RAG_PROMPT.invoke({"context": context, "question": query})
        answer = llm.invoke(prompt_value)
        return {"result": answer.content, "source_documents": docs}

    return RunnableLambda(_run)


# ── 啟動時初始化 ──────────────────────────────────────
try:
    vectorstore = load_or_build_index()
    qa_chain    = build_qa_chain(vectorstore)
    print("[APP] RAG 系統就緒。")
except FileNotFoundError as e:
    vectorstore = None
    qa_chain    = None
    print(f"[APP] 警告：{e}")


# ── 路由 ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    if qa_chain is None:
        return jsonify({"error": "索引尚未建立，請先將 PDF 放入 pdfs/ 資料夾後重啟伺服器。"}), 503

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "請輸入問題。"}), 400
    if len(question) > 500:
        return jsonify({"error": "問題不得超過 500 字。"}), 400

    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]

        sources = []
        for doc in result.get("source_documents", []):
            meta = doc.metadata
            sources.append({
                "source": Path(meta.get("source", "")).name,
                "page":   meta.get("page", 0) + 1,
            })
        # 去重（同一頁可能被撈多次）
        seen = set()
        unique_sources = []
        for s in sources:
            key = (s["source"], s["page"])
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return jsonify({"answer": answer, "sources": unique_sources})

    except Exception as e:
        print(f"[ERROR] /ask：{e}")
        return jsonify({"error": "查詢時發生錯誤，請稍後再試。"}), 500


@app.route("/rebuild-index", methods=["POST"])
def rebuild_index():
    """重新建立索引（上傳新 PDF 後呼叫）。"""
    global vectorstore, qa_chain

    # 刪除舊索引
    for f in INDEX_DIR.glob("*"):
        f.unlink()

    try:
        vectorstore = load_or_build_index()
        qa_chain    = build_qa_chain(vectorstore)
        return jsonify({"message": "索引重建完成。"})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"[ERROR] /rebuild-index：{e}")
        return jsonify({"error": "索引重建失敗。"}), 500


@app.route("/status")
def status():
    """健康檢查：確認系統是否就緒。"""
    pdf_count = len(list(PDF_DIR.rglob("*.pdf"))) if PDF_DIR.exists() else 0
    index_ready = (INDEX_DIR / "index.faiss").exists()
    return jsonify({
        "ready":      qa_chain is not None,
        "pdf_count":  pdf_count,
        "index_ready": index_ready,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", port=port)
