"""
從 113md/ 每個 MD 檔提取基本資料表，組成 qa_data/計劃總覽_113.txt
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

MD_DIR   = Path("113md")
OUT_FILE = Path("qa_data/計劃總覽_113.txt")

START_PAT = re.compile(r'教育部推動第三期.{0,60}基本.{0,10}料表', re.DOTALL)
END_PAT   = re.compile(r'教育部推動第三期.{0,60}執行成果簡述', re.DOTALL)

blocks = []
for md_path in sorted(MD_DIR.glob("*.md")):
    text = md_path.read_text(encoding="utf-8")
    m_start = START_PAT.search(text)
    m_end   = END_PAT.search(text)
    if not m_start:
        print(f"[SKIP] 找不到基本資料表：{md_path.name}")
        continue
    start = m_start.start()
    end   = m_end.start() if m_end else len(text)
    block = text[start:end].strip()
    if len(block) > 50:
        blocks.append(f"=== {md_path.stem} ===\n{block}")

OUT_FILE.write_text("\n\n".join(blocks), encoding="utf-8")
print(f"完成：{len(blocks)} 個計畫 → {OUT_FILE}")
