# -*- coding: utf-8 -*-
"""
清理 114md/*.md：
1. 整塊補助款經費表格刪除（計畫經費期程+補助款 → 計畫總經費，最多 25 行）
2. 電話號碼刪除（全文）
3. 金額數字刪除（全文掃描，精確 pattern 排除計數單位）
"""
import sys, re
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

MD_DIR = Path('114md')

# 計數單位（緊接在數字後 → 不是金額）
_CNT = r'[株位人次場件名份冊戶筆個台輛棵顆隻頭條本校縣市鄉鎮區村里班節堂點%％]'

# ── 金額數字 ────────────────────────────────────────────────────
AMOUNT_RE = re.compile(
    r'(?:NT\$|NTD|新臺幣|＄|\$)\s*[\d,]+(?:\.\d+)?\s*(?:元|萬元|億元)?'
    r'|\d{1,3}(?:,\d{3}){2,}(?:\.\d+)?\s*(?:元|萬元)?(?!\s*' + _CNT + r')'   # 7位以上（X,XXX,XXX）
    r'|\d{3},\d{3}(?:\.\d+)?\s*(?:元|萬元)?(?!\s*' + _CNT + r')'             # 6位（XXX,XXX）
    r'|\d{1,2},\d{3}(?:\.\d+)?\s*(?:元|萬元)(?!\s*' + _CNT + r')'           # 4-5位須有元/萬元
    r'|\d+(?:\.\d+)?\s*(?:萬元|億元|億)'
    r'|(?<!\d)\d+(?:\.\d+)?\s*萬(?!\s*' + _CNT + r')'
)

# ── 補助款表格刪除（START → END，最多 25 行）────────────────────
# 只針對補助款表格（款項類別 補助款...），外部資源挹注另外由 AMOUNT_RE 清
BUDGET_BLOCK_START_RE = re.compile(r'計畫經費期程|款項類別.{0,10}補助款')
BUDGET_BLOCK_END_RE   = re.compile(r'計畫總經費.{0,20}(補助款|配合款|核定)|對外募款運用執行率')
MAX_BUDGET_LINES = 25

# ── 電話號碼（全文）──────────────────────────────────────────────
PHONE_MD_RE = re.compile(
    r'電話[：:\s]*\(?\d{2,4}\)?[\d\-\s#]{6,15}'
    r'|(?<!\d)0[2-8]-?\d{4}-?\d{4}(?!\d)'
    r'|\(0[2-8]\)\s*\d{4,8}'
    r'|(?<!\d)09\d{8}(?!\d)'
    r'|09\d{2}[-\s]?\d{3}[-\s]?\d{3}'
)

def _clean_amounts(line: str) -> str:
    cleaned = AMOUNT_RE.sub('', line)
    if cleaned != line:
        cleaned = re.sub(r'\*{2}\s*\*{2}', '', cleaned)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned

def clean_file(text: str) -> str:
    lines = text.split('\n')
    result = []
    in_budget_block = False
    budget_buffer = []

    for line in lines:
        if not in_budget_block and BUDGET_BLOCK_START_RE.search(line):
            in_budget_block = True
            budget_buffer = []   # 不保留 start line
            continue

        if in_budget_block:
            if BUDGET_BLOCK_END_RE.search(line):
                # 結束符合：整塊丟棄（含本行）
                in_budget_block = False
                budget_buffer = []
            elif len(budget_buffer) >= MAX_BUDGET_LINES:
                # 超過上限，沒找到結束：把 buffer 放回去讓 AMOUNT_RE 清金額
                for buf in budget_buffer:
                    buf = PHONE_MD_RE.sub('', buf)
                    result.append(_clean_amounts(buf))
                budget_buffer = []
                in_budget_block = False
                # 目前這行也正常處理
                line = PHONE_MD_RE.sub('', line)
                result.append(_clean_amounts(line))
            else:
                budget_buffer.append(line)
            continue

        # 正常行：電話 + 金額
        line = PHONE_MD_RE.sub('', line)
        result.append(_clean_amounts(line))

    # 若 block 到檔尾仍未結束（應不會，保險用）
    for buf in budget_buffer:
        buf = PHONE_MD_RE.sub('', buf)
        result.append(_clean_amounts(buf))

    return '\n'.join(result)

# ── 執行 ──────────────────────────────────────────────────────
files = sorted(MD_DIR.glob('*.md'))
changed = 0

for f in files:
    original = f.read_text(encoding='utf-8')
    cleaned  = clean_file(original)
    if cleaned != original:
        f.write_text(cleaned, encoding='utf-8')
        changed += 1

print(f'完成：共處理 {len(files)} 件，{changed} 件有變更')
