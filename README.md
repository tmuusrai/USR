# USR 社會責任計畫書 PDF 問答系統

基於 RAG（Retrieval-Augmented Generation）技術，讓使用者用自然語言查詢大學 USR 計畫書內容。

## 技術棧

| 元件 | 選擇 | 用途 |
|------|------|------|
| Web 框架 | Flask 3.1 | HTTP 路由、模板渲染 |
| RAG 編排 | LangChain | PDF 切塊、chain 串接 |
| 向量資料庫 | FAISS | 相似度搜尋 |
| LLM | Google Gemini 2.5 Flash | 生成回答 |
| Embedding | Voyage AI voyage-3 | 文字向量化 |
| PDF 解析 | PyPDF | 讀取計畫書 |

---

## 專案結構

```
WEB/
├── app.py              # 後端核心（Flask + RAG）
├── requirements.txt    # Python 套件清單
├── .env                # API 金鑰（不進 git）
├── .env.example        # 金鑰範本
├── .gitignore
├── pdfs/               # 放計畫書 PDF（不進 git）
├── templates/
│   └── index.html      # 前端介面
├── static/
│   ├── css/
│   └── js/
└── faiss_index/        # 向量索引（自動生成，不進 git）
```

---

## 快速啟動

### 1. 建立虛擬環境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 設定 API 金鑰

```bash
cp .env.example .env
```

編輯 `.env`，填入以下金鑰：

| 變數 | 取得位置 |
|------|----------|
| `GOOGLE_API_KEY` | https://aistudio.google.com/app/apikey |
| `VOYAGE_API_KEY` | https://dash.voyageai.com/ |
| `FLASK_SECRET_KEY` | 任意隨機字串即可 |

### 4. 放入 PDF

將計畫書 PDF 檔案放入 `pdfs/` 資料夾。

### 5. 啟動伺服器

```bash
python app.py
```

首次啟動會自動讀取 PDF、向量化並建立索引（依 PDF 數量需要幾分鐘）。
之後重啟會直接載入既有索引，幾秒內就緒。

開啟瀏覽器：http://localhost:5000

---

## API 路由

| 方法 | 路由 | 說明 |
|------|------|------|
| GET | `/` | 網頁介面 |
| POST | `/ask` | 問答（JSON：`{"question": "..."}`)` |
| GET | `/status` | 系統健康檢查 |
| POST | `/rebuild-index` | 新增 PDF 後重建索引 |

### `/ask` 回傳格式

```json
{
  "answer": "根據計畫書，...",
  "sources": [
    { "source": "計畫書.pdf", "page": 3 },
    { "source": "計畫書.pdf", "page": 7 }
  ]
}
```

---

## 新增 PDF 流程

1. 將新 PDF 放入 `pdfs/`
2. 呼叫重建索引 API（不需重啟伺服器）：

```bash
curl -X POST http://localhost:5000/rebuild-index
```

---

## 常見問題

**Q：首次啟動很慢？**
正常。Voyage AI 需要對每個文字段落做向量化，索引建好後之後啟動都很快。

**Q：回答說「計畫書中未找到相關資訊」？**
問題可能超出計畫書範圍，或 PDF 解析失敗（掃描圖片型 PDF 無法讀取文字）。

**Q：想調整回答品質？**
修改 `.env` 中的 `CHUNK_SIZE`（切塊大小）和 `TOP_K_RESULTS`（參考段落數），重建索引後生效。

---

## Roadmap

- [x] Phase 1：後端核心 RAG 系統
- [ ] Phase 2：前端 UI 美化
- [ ] Phase 3：部署上線（雲端主機）
