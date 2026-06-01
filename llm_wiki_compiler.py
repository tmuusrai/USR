#!/usr/bin/env python3
"""
llm_wiki_compiler.py — USR 計畫書 LLM 知識編譯腳本
Karpathy LLM Wiki (Knowledge Compilation) 範式實作

流程：
  114md/*.md (原始計畫書)
      │
      ▼  Gemini (wiki 主編角色)
  llm_wiki_data/*.md (結構化 Wiki 頁面，含 [[雙中括號]] 交叉引用)

特性：
  - 完全不修改 114md/ 原始檔案
  - 支援中斷續跑（已輸出的檔案預設略過）
  - --max N 可先用少量測試
  - --dry-run 不呼叫 API，只列印計畫

用法：
    python llm_wiki_compiler.py               # 編譯全部（跳過已完成）
    python llm_wiki_compiler.py --max 5       # 只處理 5 份（先測試）
    python llm_wiki_compiler.py --force       # 強制覆蓋已有輸出
    python llm_wiki_compiler.py --dry-run     # 僅預覽，不呼叫 API
    python llm_wiki_compiler.py --delay 3     # 每次 API 呼叫間隔 3 秒

費用概估（257 份全量）：
    gemini-2.5-pro-preview ≈ $2–4 USD
    gemini-2.0-flash        ≈ $0.10–0.30 USD（建議先測試用 flash）
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── 路徑 ────────────────────────────────────────────────
SRC_DIR  = Path("114md")           # 原始計畫書（唯讀）
OUT_DIR  = Path("llm_wiki_data")   # Wiki 頁面輸出目錄
LOG_PATH = Path("llm_wiki_compiler.log")

# ── 模型（可用環境變數或 --model 覆蓋）──────────────────
DEFAULT_MODEL = os.getenv("WIKI_COMPILER_MODEL",
                           os.getenv("LLM_MODEL", "gemini-2.5-flash"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 每份計畫書取前 N 字元送給 LLM（節省 token）
SNIPPET_CHARS = 4500


# ── Prompt ─────────────────────────────────────────────
WIKI_PROMPT = """\
你是台灣大學社會責任（USR）知識百科的資深主編。

你的任務是將以下一份 USR 計畫書「編譯」成標準的 Wiki 頁面：
- 去除冗長的行政文字（經費明細、日期格式、聯絡資訊、核定金額等）
- 保留並整理：核心目標、主要活動、創新亮點、可量化成果
- 提取所有重要的知識節點，並用 [[雙中括號]] 標記以建立交叉引用：
    • 地名 / 場域：[[屏東縣]]、[[八斗子漁港]]、[[花蓮市]]
    • 計畫議題：[[高齡照護]]、[[食農教育]]、[[地方創生]]、[[循環經濟]]
    • SDG 目標：[[SDG3]]、[[SDG11]]（數字緊接 SDG，無空格）
    • 合作機構：[[屏東市公所]]、[[民和國小]]
    • 關鍵方法：[[青銀共創]]、[[社區陪伴]]、[[智慧農業]]
    • 相關概念：[[在地知識]]、[[永續發展]]

**輸出規則（非常重要）：**
- 直接輸出 Markdown，不要任何前言、解釋或 ```markdown 標記
- 必須有「核心概念與連結」章節，列出本計畫涉及的所有 [[知識節點]]
- 每個章節簡潔，條列式為主

**輸出格式：**

# {計畫名稱}

| 欄位 | 內容 |
|------|------|
| 執行學校 | 學校名稱 |
| 計畫類型 | 萌芽型 / 深耕型 / 國際合作型 / 特色永續型 |
| SDG 關聯 | [[SDG#]]、[[SDG#]] |
| 計畫議題 | [[議題名稱]] |
| 實踐場域 | [[縣市]]、[[鄉鎮/場域名稱]] |

## 執行目標

（核心目標，2–3 句話）

## 主要活動與成果

- 活動或成果（含 [[關鍵概念]] 連結）
- ...

## 核心概念與連結

[[概念1]] — 簡短說明與本計畫的關係
[[概念2]] — ...
...

---

【計畫書原文】
{content}
"""


# ── 工具函式 ───────────────────────────────────────────

def _safe_read(path: Path) -> str | None:
    for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return None


def _safe_stem(stem: str) -> str:
    """將原始 MD 檔名轉成安全的輸出檔名（保留中文，去除不合法字元）。"""
    safe = re.sub(r'[\\/*?:"<>|]', '_', stem)
    return safe[:100] + ".md"


def _build_llm(model: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )


def _compile_one(llm, src_path: Path) -> str | None:
    """呼叫 LLM 編譯單份計畫書，回傳 Wiki Markdown 字串。失敗回傳 None。"""
    from langchain_core.messages import HumanMessage

    text = _safe_read(src_path)
    if not text:
        return None

    snippet = text[:SNIPPET_CHARS]
    prompt  = WIKI_PROMPT.format(content=snippet)

    try:
        res = llm.invoke([HumanMessage(content=prompt)])
        raw = res.content
        # Gemini 有時回傳 list of blocks
        if isinstance(raw, list):
            raw = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in raw
            )
        return raw.strip()
    except Exception as e:
        print(f"    [API ERROR] {e}")
        return None


# ── 主程式 ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="USR LLM Wiki Compiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max",     type=int,   default=0,
                        help="最多處理幾份（0 = 全部）")
    parser.add_argument("--force",   action="store_true",
                        help="強制重新編譯（覆蓋已有輸出）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只列出待處理檔案，不呼叫 API")
    parser.add_argument("--delay",   type=float, default=2.0,
                        help="每次 API 呼叫後的等待秒數（預設 2.0）")
    parser.add_argument("--model",   type=str,   default=DEFAULT_MODEL,
                        help=f"Gemini 模型名稱（預設：{DEFAULT_MODEL}）")
    args = parser.parse_args()

    # ── 前置檢查 ──
    if not SRC_DIR.exists():
        print(f"[ERROR] 找不到來源資料夾：{SRC_DIR}")
        sys.exit(1)

    if not GOOGLE_API_KEY:
        print("[ERROR] 找不到 GOOGLE_API_KEY，請確認 .env 設定")
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)

    md_files = sorted(SRC_DIR.rglob("*.md"))
    if args.max:
        md_files = md_files[: args.max]

    total = len(md_files)
    already_done = sum(
        1 for p in md_files
        if (OUT_DIR / _safe_stem(p.stem)).exists()
    )

    # ── 摘要 ──
    print("=" * 60)
    print("  USR LLM Wiki Compiler")
    print("=" * 60)
    print(f"  來源資料夾  : {SRC_DIR}  ({total} 份)")
    print(f"  輸出資料夾  : {OUT_DIR}")
    print(f"  模型        : {args.model}")
    print(f"  已完成      : {already_done} 份（--force 可覆蓋）")
    print(f"  待處理      : {total - (0 if args.force else already_done)} 份")
    print(f"  API 間隔    : {args.delay} 秒")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry-run] 待處理清單：")
        for i, p in enumerate(md_files, 1):
            out = OUT_DIR / _safe_stem(p.stem)
            status = "✓ 已完成" if out.exists() else "  待編譯"
            print(f"  {i:>3}. [{status}] {p.name[:70]}")
        return

    # ── 編譯迴圈 ──
    llm = _build_llm(args.model)
    done, skipped, failed = 0, 0, 0
    log_lines: list[str] = [
        f"# LLM Wiki Compiler Log — {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"# Model: {args.model}  |  Source: {SRC_DIR}  |  Output: {OUT_DIR}",
        "",
    ]

    for i, src_path in enumerate(md_files, 1):
        out_path = OUT_DIR / _safe_stem(src_path.stem)

        if out_path.exists() and not args.force:
            print(f"[{i:>3}/{total}] SKIP  {src_path.name[:65]}")
            skipped += 1
            log_lines.append(f"SKIP  {src_path.name}")
            continue

        print(f"[{i:>3}/{total}] 編譯  {src_path.name[:65]}")
        wiki_md = _compile_one(llm, src_path)

        if wiki_md:
            out_path.write_text(wiki_md, encoding="utf-8")
            done += 1
            log_lines.append(f"OK    {src_path.name}  →  {out_path.name}")
            print(f"         ✓ {out_path}")
        else:
            failed += 1
            log_lines.append(f"ERR   {src_path.name}")
            print(f"         ✗ 失敗（見 log）")

        # 不是最後一筆才等待
        if i < total:
            time.sleep(args.delay)

    # ── 結果 ──
    log_lines += [
        "",
        f"# 結果：成功 {done}，略過 {skipped}，失敗 {failed}",
    ]
    LOG_PATH.write_text("\n".join(log_lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"  完成：成功 {done}，略過 {skipped}，失敗 {failed}")
    print(f"  輸出目錄：{OUT_DIR.resolve()}")
    print(f"  Log：{LOG_PATH}")
    print("=" * 60)

    if done > 0:
        print("\n下一步建議：")
        print("  1. 檢查 llm_wiki_data/ 的輸出品質")
        print("  2. 滿意後再考慮整合到 FAISS 索引或 wiki-map")


if __name__ == "__main__":
    main()
