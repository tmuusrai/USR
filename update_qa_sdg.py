#!/usr/bin/env python3
"""
update_qa_sdg.py — 根據 114md/ 的 SDG 勾選，自動更新 qa_custom.txt 每個 SDG 區塊的計畫清單與計數。
"""
import re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

# ── 引入 sdg_audit 的解析函式 ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from sdg_audit import parse_all_mds, _norm

QA_FILE = Path("qa_data/qa_custom.txt")

SDG_NAMES = {
    1:"消除貧窮", 2:"零飢餓", 3:"良好健康與福祉", 4:"優質教育",
    5:"性別平等", 6:"乾淨用水及衛生", 7:"可負擔的清潔能源",
    8:"合宜工作與經濟成長", 9:"工業創新及基礎建設", 10:"減少不平等",
    11:"永續城市及社區", 12:"負責任的消費及生產", 13:"氣候行動",
    14:"保育海洋生態", 15:"保育陸域生態", 16:"和平正義及健全制度",
    17:"夥伴關係",
}

def build_sdg_index(md_data: dict) -> dict[int, list[tuple[str, str]]]:
    """從 md_data 建立每個 SDG 對應的 (uni, plan) 清單，依學校名稱排序。"""
    idx: dict[int, list] = {n: [] for n in range(1, 18)}
    for d in md_data.values():
        for sdg in d['sdgs']:
            if 1 <= sdg <= 17:
                idx[sdg].append((d['uni'], d['plan']))
    for sdg in idx:
        idx[sdg].sort(key=lambda x: (x[0], x[1]))
    return idx

def update_qa(qa_text: str, sdg_index: dict[int, list]) -> str:
    lines = qa_text.split('\n')
    out = []
    i = 0

    # pattern: A: **SDG{n}「...」** 共有 ... 個計畫，涉及 ... 間學校：
    header_pat = re.compile(r'^A: \*\*SDG(\d{1,2})「[^」]+」\*\* 共有 \d+ 個計畫，涉及 \d+ 間學校：')
    entry_pat  = re.compile(r'^\d+\. \*\*(.+?)\*\*[　\s]+(.+)')

    while i < len(lines):
        line = lines[i]
        m = header_pat.match(line)
        if m:
            sdg_num = int(m.group(1))
            # 收集現有計畫列表（直到空行後遇到新 Q:）
            existing: list[tuple[str,str]] = []
            j = i + 1
            # 跳過開頭空行
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            # 讀現有 numbered entries
            while j < len(lines):
                em = entry_pat.match(lines[j])
                if em:
                    existing.append((em.group(1).strip(), em.group(2).strip()))
                    j += 1
                elif lines[j].strip() == '':
                    # 空行 → 可能結束，看下一行是否還有 entry
                    if j+1 < len(lines) and entry_pat.match(lines[j+1]):
                        j += 1  # 跳過空行繼續
                    else:
                        break
                else:
                    break  # 非 entry、非空行 → 結束

            # 從 sdg_index 取得完整清單
            all_pairs = sdg_index.get(sdg_num, [])

            # 現有清單的 normalized 集合（用於查重）
            existing_norm = {(_norm(u), _norm(p)) for u, p in existing}

            # 找出缺少的（在 md 有但 QA 沒有）
            missing = [(u, p) for u, p in all_pairs
                       if (_norm(u), _norm(p)) not in existing_norm]

            # 合併：先保留原有順序，再附加缺少的（依學校排序）
            full_list = existing + sorted(missing, key=lambda x: (x[0], x[1]))

            # 統計
            total = len(full_list)
            schools = len({u for u, _ in full_list})

            # 輸出更新後的 header
            sdg_label = SDG_NAMES.get(sdg_num, '')
            out.append(f'A: **SDG{sdg_num}「{sdg_label}」** 共有 {total} 個計畫，涉及 {schools} 間學校：')
            out.append('')
            for k, (u, p) in enumerate(full_list, 1):
                out.append(f'{k}. **{u}**　{p}')
            out.append('')  # 結尾空行

            # i 跳到 j（已消耗的行）
            i = j
            continue

        out.append(line)
        i += 1

    return '\n'.join(out)


def main():
    print("讀取 114md/ …")
    md_data = parse_all_mds()
    sdg_index = build_sdg_index(md_data)

    for sdg, pairs in sdg_index.items():
        print(f"  SDG{sdg:2d}: {len(pairs)} 個計畫")

    qa_text = QA_FILE.read_text(encoding='utf-8')
    updated = update_qa(qa_text, sdg_index)

    # 備份
    bak = QA_FILE.with_suffix('.txt.bak_update')
    bak.write_text(qa_text, encoding='utf-8')
    print(f"\n備份已存至 {bak}")

    QA_FILE.write_text(updated, encoding='utf-8')
    print(f"已更新 {QA_FILE}")


if __name__ == '__main__':
    main()
