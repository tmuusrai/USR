# -*- coding: utf-8 -*-
"""
掃描 FAISS，找出有真正國外活動的計畫，更新 location_index 和 label_index。
"""
import os, sys, re, json
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings
from pathlib import Path

LOC_IDX = Path('114_output/location_index.json')
LBL_IDX = Path('114_output/label_index.json')
YEAR = '114'

emb = VoyageAIEmbeddings(voyage_api_key=os.environ['VOYAGE_API_KEY'], model='voyage-4-large')
vs = FAISS.load_local('faiss_index', emb, allow_dangerous_deserialization=True)

COUNTRY_RE = re.compile(
    r'(日本|韓國|越南|泰國|印尼|馬來西亞|菲律賓|新加坡|印度|柬埔寨|緬甸|寮國'
    r'|帛琉|坦尚尼亞|英國|德國|法國|尼泊爾|巴布亞|尚吉巴)'
)
A_RE = re.compile(r'國外實踐場域\s*[：:]?\s*\d*\s*固定實踐場域')
B_RE = re.compile(
    r'前往.{0,10}(日本|韓國|越南|泰國|印尼|馬來西亞|菲律賓|新加坡|印度|柬埔寨|緬甸|寮國|帛琉|坦尚尼亞|英國|德國|法國|尼泊爾|巴布亞)'
    r'|赴.{0,5}(日本|韓國|越南|泰國|印尼|馬來西亞|菲律賓|新加坡|印度|英國|德國|法國|尼泊爾).{0,15}(執行|實踐|工作坊|交流|場域|訪視|服務)'
    r'|(日本|韓國|越南|泰國|印尼|馬來西亞|菲律賓|新加坡|印度|柬埔寨|英國|德國|法國|尼泊爾).{0,10}(場域|工作坊|實踐|執行|合作機構|合作大學|合作夥伴)'
    r'|海外場域.{0,30}(執行|運作|實踐|師生)'
)
BORROW_RE = re.compile(
    r'借鏡|效法|仿照'
    r'|參考.{0,5}(日本|韓國|泰國|新加坡|英國|德國|越南|印尼|馬來西亞)'
    r'|參加.{0,15}(世界賽|國際賽|競賽|大賽|奧林匹亞|博覽會|世博|展覽|研討會|論壇|國際會議)'
    r'|前往.{0,15}(世界賽|國際賽|競賽|博覽會|世博|展覽|研討會)'
    r'|補助.{0,10}前往|青年署補助|學海築夢|海外圓夢'
    r'|邀請.{0,15}(來台|來臺|到台灣|蒞臨|擔任講者|擔任講師|授課)'
    r'|學術發表|發表論文|發表研究'
    r'|在.{0,3}(台灣|臺灣|國內).{0,10}(工作坊|活動|課程)'
)

# 人工排除的誤判計畫（計畫名部分字串即可）
MANUAL_EXCLUDE = {
    '從認識到選擇',       # 元培：培英國中誤判為英國
    '開江．破竹',         # 崑山：英國建築師來台
    '甘丹',               # 德明：未前往法國
    '農安紮根',           # 朝陽：海外生來台
    '里山團結碳經濟',     # 慈濟：德國講師來台
    '燕巢樂齡',           # 義守：台灣舉辦活動名含日本
    '石碇崩山',           # 華梵：日本學者來台
    '梓想與你鄉遇',       # 台南應用：盤點越南商機
    '北臺首學帶狀',       # 明志：活動名稱誤判
    '新北社區國際文化',   # 明志：海外生來台
    '原鄉兒少肥胖',       # 長庚大學：計畫主持人個人研修赴日，非師生場域執行
}

def _is_B(text):
    for line in re.split(r'[。\n]', text):
        if B_RE.search(line) and not BORROW_RE.search(line):
            return True
    return False

def norm(s):
    return re.sub(r'[\s　\-_–—·\(（）\)Ⅱ+×xX]+', '', s).lower()

# 掃描
found = {}
for doc in vs.docstore._dict.values():
    src = doc.metadata.get('plan_name', '') or doc.metadata.get('source', '')
    if not src or '114md' not in src:
        continue
    text = doc.page_content
    is_A = bool(A_RE.search(text))
    is_B = _is_B(text)
    if not (is_A or is_B):
        continue
    countries = set(COUNTRY_RE.findall(text))
    if not countries:
        continue
    fname = src.replace('/', os.sep).replace('\\', os.sep)
    fname = fname.split('114md' + os.sep)[-1] if ('114md' + os.sep) in fname else fname.split('114md')[-1].lstrip(os.sep)
    parts = fname.split('_', 1)
    school = parts[0] if len(parts) > 1 else ''
    plan_raw = parts[1] if len(parts) > 1 else fname
    plan_raw = plan_raw.replace('.md', '')
    plan_raw = re.sub(r'\s*\(\d{3}USR-[^)]*\)', '', plan_raw).strip()

    # 人工排除
    if any(ex in plan_raw for ex in MANUAL_EXCLUDE):
        continue

    key = f'{school}：{plan_raw}'
    found.setdefault(key, {'countries': set(), 'school': school, 'plan': plan_raw})
    found[key]['countries'].update(countries)

# 載入現有索引
loc_data = json.loads(LOC_IDX.read_text(encoding='utf-8'))
lbl_data = json.loads(LBL_IDX.read_text(encoding='utf-8'))
loc_plans = loc_data.get(YEAR, {}).get('plans', {})
existing_overseas = {k for k, v in loc_plans.items() if v.get('overseas_countries')}
existing_norm = {norm(k.split('：', 1)[-1]): k for k in existing_overseas}

# 比對並更新
added = []
for key, info in sorted(found.items()):
    n = norm(info['plan'])
    # 已有 overseas_countries 的跳過
    if any(n == ek or (len(n) > 5 and (n in ek or ek in n)) for ek in existing_norm):
        continue
    # 找 location_index 對應的 key
    matched_loc_key = None
    for loc_key in loc_plans:
        loc_plan_norm = norm(loc_key.split('：', 1)[-1])
        if n == loc_plan_norm or (len(n) > 5 and (n in loc_plan_norm or loc_plan_norm in n)):
            matched_loc_key = loc_key
            break
    if not matched_loc_key:
        print(f'  ! 找不到 location_index key：{key[:60]}')
        continue

    countries = sorted(info['countries'])
    loc_plans[matched_loc_key]['overseas_countries'] = countries
    added.append(matched_loc_key)
    print(f'  + {matched_loc_key}')
    print(f'    → {countries}')

# 更新 label_index 的 國外 key
existing_lbl = set(lbl_data.get('國外', []))
merged = list(existing_lbl | set(added))
lbl_data['國外'] = merged

# 寫回
loc_data[YEAR]['plans'] = loc_plans
LOC_IDX.write_text(json.dumps(loc_data, ensure_ascii=False, indent=2), encoding='utf-8')
LBL_IDX.write_text(json.dumps(lbl_data, ensure_ascii=False, indent=2), encoding='utf-8')

print(f'\n新增 {len(added)} 件，location_index 和 label_index 已更新')
print(f'label_index 國外 key 共 {len(merged)} 件')
