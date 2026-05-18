"""一次性腳本：重建 FAISS 索引後結束。"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.embeddings import Embeddings

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 100))
PDF_DIR        = Path("pdfs")
EXTRA_DIR      = Path("extra_docs")
TXT_DIR        = Path("114txt")
INDEX_DIR      = Path("faiss_index")

class _CachedEmbeddings(Embeddings):
    def __init__(self, base):
        self._base  = base
        self._cache = {}
    def embed_query(self, text):
        if text not in self._cache:
            self._cache[text] = self._base.embed_query(text)
        return self._cache[text]
    def embed_documents(self, texts):
        return self._base.embed_documents(texts)
    def __getattr__(self, name):
        return getattr(self._base, name)

print("[BUILD] 初始化 voyage-4-large embedding...")
embeddings = _CachedEmbeddings(VoyageAIEmbeddings(
    voyage_api_key=VOYAGE_API_KEY,
    model="voyage-4-large",
))

pdf_files = list(PDF_DIR.rglob("*.pdf"))
print(f"[BUILD] 找到 {len(pdf_files)} 個 PDF")

docs = []
for pdf_path in pdf_files:
    print(f"  讀取：{pdf_path.name}")
    try:
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    except Exception as e:
        print(f"  [WARN] 跳過 {pdf_path.name}：{e}")

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
print(f"[BUILD] 共切出 {total} 個段落，開始向量化...")

BATCH = 128
vectorstore = None
for i in range(0, total, BATCH):
    batch = chunks[i : i + BATCH]
    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embeddings)
    else:
        vectorstore.add_documents(batch)
    done = min(i + BATCH, total)
    print(f"  [{done}/{total}] {done*100//total}%", flush=True)

INDEX_DIR.mkdir(exist_ok=True)
vectorstore.save_local(str(INDEX_DIR))
print("[BUILD] 索引建立完成！")
