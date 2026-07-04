#!/usr/bin/env python3
"""
sdg_audit.py — 只讀 SDGs關聯議題（或 SDGs夥伴關係）那一行，
提取 ☑/■/▓ 後的數字，比對 qa_custom.txt 是否正確。
"""
import re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

MD_DIR  = Path("114md")
QA_FILE = Path("qa_data/qa_custom_114.txt")

SDG_NAMES = {
    1:"消除貧窮", 2:"消除飢餓", 3:"良好健康與福祉", 4:"優質教育",
    5:"性別平等", 6:"乾淨用水及衛生", 7:"可負擔的清潔能源",
    8:"合宜工作與經濟成長", 9:"工業創新及基礎建設", 10:"減少不平等",
    11:"永續城市及社區", 12:"負責任的消費及生產", 13:"氣候行動",
    14:"保育海洋生態", 15:"保育陸域生態", 16:"和平正義及健全制度",
    17:"夥伴關係",
}

# 只匹配 SDGs關聯議題 / SDGs夥伴關係 的行
HEADER_PAT = re.compile(r'SDGs?\s*(?:關聯議題|夥伴關係)', re.IGNORECASE)

# ☑/■/▓ 後面接 1-2 位數（格式：☑3. 或 ☑ 3 ）
CHECKED_NUM = re.compile(r'[☑■▓]\s*(\d{1,2})(?!\d)')

# 純文字格式（無勾號）：數字.文字；
PLAIN_NUM   = re.compile(r'(?<![☑■▓\d])(\d{1,2})[\.、]')


def _safe_read(p: Path) -> str | None:
    for enc in ["utf-8-sig", "utf-8", "cp950", "big5"]:
        try:
            return p.read_text(encoding=enc)
        except Exception:
            continue
    return None


def _stem_to_uni_plan(stem: str) -> tuple[str, str]:
    """
    從檔名拆出 (學校, 計畫名稱)。
    移除 (114USR-RMF-xxx) 形式的代碼，保留 _formatted 等後綴。
    """
    idx = stem.find('_')
    if idx <= 0:
        return '', stem
    uni = stem[:idx]
    plan_raw = stem[idx + 1:]
    # 移除 (114USR-...) 代碼
    plan = re.sub(r'\(114USR-[^\)]+\)', '', plan_raw)
    # 移除結尾的 (1) (2) 等
    plan = re.sub(r'\(\d+\)\s*$', '', plan).strip()
    return uni, plan


def _extract_sdg_line(text: str) -> set[int]:
    """
    只看含 SDGs關聯議題 / SDGs夥伴關係 的那幾行，
    提取勾選的 SDG 號碼（1-17）。
    若無勾號則嘗試純文字格式。
    """
    sdgs: set[int] = set()
    lines = text.split('\n')

    for i, line in enumerate(lines):
        if not HEADER_PAT.search(line):
            continue

        # 收集這行 + 最多 15 行延續（空行跳過，直到出現新 section）
        chunk = line
        for k in range(1, 16):
            if i + k >= len(lines):
                break
            nxt = lines[i + k].strip()
            # 新 section 標記 → 停止
            if nxt.startswith('##'):
                break
            if nxt.startswith('|') and '☑' not in nxt and '■' not in nxt and '▓' not in nxt:
                break
            # 非空的「新欄位」行（不含勾號）→ 停止
            if nxt and not re.search(r'[☑■▓□☐]|\d{1,2}[\.、]', nxt):
                break
            # 空行：跳過但繼續
            if not nxt:
                continue
            chunk += '\n' + nxt

        # 有勾號 → 只取勾號後的數字
        if re.search(r'[☑■▓]', chunk):
            for m in CHECKED_NUM.finditer(chunk):
                n = int(m.group(1))
                if 1 <= n <= 17:
                    sdgs.add(n)
        else:
            # 純文字：取冒號後的數字
            colon = chunk.find('：')
            if colon == -1:
                colon = chunk.find(':')
            rest = chunk[colon+1:] if colon != -1 else chunk
            for m in PLAIN_NUM.finditer(rest):
                n = int(m.group(1))
                if 1 <= n <= 17:
                    sdgs.add(n)

    return sdgs


def parse_all_mds() -> dict[str, dict]:
    results = {}
    for path in sorted(MD_DIR.glob('*.md')):
        text = _safe_read(path)
        uni, plan = _stem_to_uni_plan(path.stem)
        sdgs = _extract_sdg_line(text) if text else set()
        results[path.stem] = {'sdgs': sdgs, 'uni': uni, 'plan': plan}
    return results


def _load_qa_sdg_map(qa_text: str) -> dict[int, list[tuple[str,str]]]:
    sdg_map: dict[int, list] = {}
    for block in re.split(r'\nQ:', qa_text):
        lines = block.split('\n')
        m = re.search(r'SDG\s*(\d{1,2})', lines[0], re.IGNORECASE)
        if not m:
            continue
        sdg_num = int(m.group(1))
        a_idx = block.find('\nA:')
        if a_idx == -1:
            continue
        plans = []
        for line in block[a_idx+3:].split('\n'):
            pm = re.match(r'\d+\.\s*\*\*(.+?)\*\*[　\s]+(.+)', line)
            if pm:
                plans.append((pm.group(1).strip(), pm.group(2).strip()))
        if plans:
            sdg_map[sdg_num] = plans
    return sdg_map


def _norm(s: str) -> str:
    """移除空白方便模糊比對"""
    return re.sub(r'\s+', '', s)


def main():
    print('=' * 70)
    print('  SDG Audit（僅讀 SDGs關聯議題 那行）')
    print('=' * 70)

    md_data = parse_all_mds()
    total   = len(md_data)
    no_sdg  = [(s, d) for s, d in md_data.items() if not d['sdgs']]

    print(f'\n總計 {total} 個檔案，SDGs關聯議題 行偵測不到 SDG 的有 {len(no_sdg)} 個：')
    for s, d in sorted(no_sdg, key=lambda x: x[1]['uni']):
        print(f'  ✗ {d["uni"]}｜{d["plan"]}')

    # Build lookup (normalized)
    md_index: dict[tuple[str,str], set[int]] = {}
    for d in md_data.values():
        md_index[(_norm(d['uni']), _norm(d['plan']))] = d['sdgs']

    def find_sdgs(uni_q: str, plan_q: str) -> set[int] | None:
        u_n = _norm(uni_q)
        p_n = _norm(plan_q)
        # exact
        if (u_n, p_n) in md_index:
            return md_index[(u_n, p_n)]
        # substring match (plan within same uni)
        for (u, p), sdgs in md_index.items():
            if (not u_n or u_n in u or u in u_n) and (p_n in p or p in p_n):
                return sdgs
        # plan only
        for (u, p), sdgs in md_index.items():
            if p_n in p or p in p_n:
                return sdgs
        return None

    qa_text    = QA_FILE.read_text(encoding='utf-8')
    qa_sdg_map = _load_qa_sdg_map(qa_text)

    print('\n' + '=' * 70)
    print('  比對結果')
    print('=' * 70)

    wrong_qa   = []   # QA 聲稱有此 SDG 但 md 不符
    not_found  = []   # QA 裡的計畫在 md 找不到
    missing_qa = []   # md 有此 SDG 但 QA 沒列

    for sdg_num in sorted(qa_sdg_map.keys()):
        qa_plans = qa_sdg_map[sdg_num]
        for uni_q, plan_q in qa_plans:
            actual = find_sdgs(uni_q, plan_q)
            if actual is None:
                not_found.append((sdg_num, uni_q, plan_q))
            elif sdg_num not in actual:
                wrong_qa.append((sdg_num, uni_q, plan_q, sorted(actual)))

        qa_plan_norms = {_norm(p) for _, p in qa_plans}
        for d in md_data.values():
            if sdg_num not in d['sdgs']:
                continue
            p_n = _norm(d['plan'])
            if not any(qn in p_n or p_n in qn for qn in qa_plan_norms):
                missing_qa.append((sdg_num, d['uni'], d['plan']))

    # ── 報告 ────────────────────────────────────────────────────
    if wrong_qa:
        print('\n【QA 答案有誤】QA 聲稱有此 SDG，但 md 實際不符：')
        for sdg, u, p, actual in wrong_qa:
            print(f'  SDG{sdg}  {u}｜{p}')
            print(f'       → md 實際 SDG：{actual if actual else "（找不到 SDGs關聯議題 行）"}')
    else:
        print('\n【QA 答案有誤】無')

    if not_found:
        print(f'\n【QA 裡找不到對應 md 檔】共 {len(not_found)} 筆：')
        for sdg, u, p in not_found:
            print(f'  SDG{sdg}  {u}｜{p}')
    else:
        print('\n【QA 裡找不到對應 md 檔】無')

    if missing_qa:
        print(f'\n【md 有此 SDG 但 QA 未列入】共 {len(missing_qa)} 筆：')
        cur = 0
        for sdg, u, p in sorted(missing_qa):
            if sdg != cur:
                print(f'\n  SDG{sdg} {SDG_NAMES.get(sdg,"")}：')
                cur = sdg
            print(f'    {u}｜{p}')
    else:
        print('\n【md 有此 SDG 但 QA 未列入】無')

    print('\n' + '=' * 70)
    print('[每個檔案偵測到的 SDG]')
    for s in sorted(md_data.keys()):
        d = md_data[s]
        sdg_str = ','.join(str(n) for n in sorted(d['sdgs'])) or '（無）'
        print(f'{d["uni"]:<14} {d["plan"][:40]:<42} {sdg_str}')


if __name__ == '__main__':
    main()
