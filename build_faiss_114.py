# -*- coding: utf-8 -*-
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.documents import Document

MD_DIR      = Path("114md")
SUMMARY_DIR = Path("114_output/summary")
INDEX_DIR   = Path("faiss_index")

embeddings = VoyageAIEmbeddings(
    voyage_api_key=os.getenv("VOYAGE_API_KEY"),
    model="voyage-4-large",
    batch_size=32,
)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

docs = []

# 原始 md 文件
for f in sorted(MD_DIR.glob("*.md")):
    text = f.read_text(encoding="utf-8")
    chunks = splitter.split_text(text)
    for chunk in chunks:
        docs.append(Document(page_content=chunk, metadata={"source": str(f), "type": "full"}))

print(f"md 文件：{len(docs)} chunks")

# 摘要 TXT
summary_count = 0
for f in sorted(SUMMARY_DIR.glob("*.txt")):
    text = f.read_text(encoding="utf-8")
    chunks = splitter.split_text(text)
    for chunk in chunks:
        docs.append(Document(page_content=chunk, metadata={"source": str(f), "type": "summary"}))
    summary_count += len(chunks)

print(f"摘要 TXT：{summary_count} chunks")
print(f"總計：{len(docs)} chunks，開始建立向量索引...")

vs = FAISS.from_documents(docs, embeddings)
vs.save_local(str(INDEX_DIR))
print(f"完成！索引儲存至：{INDEX_DIR}")
