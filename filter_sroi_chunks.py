# -*- coding: utf-8 -*-
"""
filter_sroi_chunks.py
過濾 kw_chunks 裡的 SROI entries：
  只留「真正在做 SROI 分析」的 chunk（類型1）
  移除「只是去參加培訓/工作坊」和「純數字試算表」
"""
import json, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

KW_CHUNKS_JSON = Path("114_output/kw_chunks_test.json")

# 保留條件：chunk 要含有實際分析相關詞
KEEP_PATTERNS = re.compile(
    r'利害關係人|投入.{0,10}產出|產出.{0,10}成果|影響力因子|貨幣化|量化.{0,10}(成果|影響|效益)|'
    r'SROI.{0,20}(分析|計算|報告|架構|評估|調查問卷)|'
    r'(分析|計算|評估).{0,20}SROI|'
    r'社會投資報酬率.{0,30}(分析|計算|評估)|'
    r'SROI.{0,10}(值|結果|數據|數值).{0,5}\d'
)

# 排除條件：只是培訓/工作坊/課程參加
EXCLUDE_PATTERNS = re.compile(
    r'(SROI|社會投資報酬率).{0,15}(工作坊|培訓班|研習|課程|上課|入門|實作班|講座)|'
    r'(工作坊|培訓班|研習|課程|上課|入門|實作班).{0,15}(SROI|社會投資報酬率)|'
    r'參加.{0,10}(SROI|社會投資報酬率)|'
    r'接受.{0,10}(SROI|社會投資報酬率)'
)

# 排除純數字試算表（chunk 很短且只有 SROI: 數字）
RAW_NUMBER_RE = re.compile(r'^[\s\S]{0,150}SROI[：:]\s*\d[\s\S]{0,100}$')

def is_keep(text: str) -> bool:
    if len(text) < 80:
        return False
    # 排除純試算表
    if RAW_NUMBER_RE.match(text) and len(text) < 200:
        return False
    # 排除只是參加培訓（且沒有分析詞）
    if EXCLUDE_PATTERNS.search(text) and not KEEP_PATTERNS.search(text):
        return False
    # 要有實質分析內容
    return bool(KEEP_PATTERNS.search(text))

# 載入
kw_data = json.loads(KW_CHUNKS_JSON.read_text(encoding='utf-8'))
sroi_entries = kw_data.get("114", {}).get("SROI", [])
print(f"SROI 原始 chunk 數：{len(sroi_entries)}")

kept = [e for e in sroi_entries if is_keep(e.get("text", ""))]
removed = len(sroi_entries) - len(kept)

print(f"保留：{len(kept)}  移除：{removed}")

# 預覽移除的（前5筆）
removed_samples = [e for e in sroi_entries if not is_keep(e.get("text", ""))]
print("\n=== 移除樣本（前5筆）===")
for e in removed_samples[:5]:
    print(f"[{e['plan'][:35]}]")
    print(e['text'][:150])
    print()

print("=== 保留樣本（前5筆）===")
for e in kept[:5]:
    print(f"[{e['plan'][:35]}]")
    print(e['text'][:150])
    print()

ans = input("確認寫入？(y/n): ").strip().lower()
if ans == 'y':
    kw_data["114"]["SROI"] = kept
    KW_CHUNKS_JSON.write_text(json.dumps(kw_data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"✓ 已寫入，SROI chunk 從 {len(sroi_entries)} → {len(kept)}")
else:
    print("取消。")
