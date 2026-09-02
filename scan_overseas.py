# -*- coding: utf-8 -*-
import os, sys, re, json
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_voyageai import VoyageAIEmbeddings

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
# 排除不算國外場域的句子
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

def _is_B(text: str) -> bool:
    """逐句判斷：有 B_RE 命中，且該句不是借鏡句。"""
    for line in re.split(r'[。\n]', text):
        if B_RE.search(line) and not BORROW_RE.search(line):
            return True
    return False

loc = json.loads(open('114_output/location_index.json', encoding='utf-8').read())
existing_overseas = {k for k, v in loc.get('114', {}).get('plans', {}).items() if v.get('overseas_countries')}

def norm(s):
    return re.sub(r'[\s　\-_–—·\(（）\)II+\xd7xX]+', '', s).lower()

existing_norm = {norm(k.split('：', 1)[-1]) for k in existing_overseas}

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

    # 取學校名和計畫名
    fname = src.replace('/', os.sep).replace('\\', os.sep)
    fname = fname.split('114md' + os.sep)[-1] if ('114md' + os.sep) in fname else fname.split('114md')[-1].lstrip(os.sep)
    parts = fname.split('_', 1)
    school = parts[0] if len(parts) > 1 else ''
    plan_raw = parts[1] if len(parts) > 1 else fname
    plan_raw = plan_raw.replace('.md', '')
    plan_raw = re.sub(r'\s*\(\d{3}USR-[^)]*\)', '', plan_raw).strip()

    key = f'{school}：{plan_raw}'
    tag = []
    if is_A: tag.append('A')
    if is_B: tag.append('B')
    # 收集觸發 B 條件的句子作為證據
    evidence = []
    for line in re.split(r'[。\n]', text):
        if B_RE.search(line) and not BORROW_RE.search(line) and line.strip():
            evidence.append(line.strip())
    found.setdefault(key, {'countries': set(), 'tag': set(), 'school': school, 'plan': plan_raw, 'evidence': []})
    found[key]['countries'].update(countries)
    found[key]['tag'].update(tag)
    for ev in evidence:
        if ev not in found[key]['evidence']:
            found[key]['evidence'].append(ev)

print(f'共掃到 {len(found)} 件\n')

new_plans = []
for key, info in sorted(found.items()):
    n = norm(info['plan'])
    already = any(n == ek or (len(n) > 5 and (n in ek or ek in n)) for ek in existing_norm)
    if not already:
        new_plans.append((key, info))

print(f'=== 新增（不在現有 overseas 15 件）：{len(new_plans)} 件 ===\n')
for key, info in new_plans:
    print(f'[{"|".join(sorted(info["tag"]))}] {info["school"]}：{info["plan"][:55]}')
    print(f'     國家：{sorted(info["countries"])}')
    for line in info.get('evidence', []):
        print(f'     > {line[:120]}')
    print()
