import os, re, shutil
from collections import defaultdict

txt_dir   = r"C:\Users\Kuo\desktop\usr\114txt"
plan_file = r"C:\Users\Kuo\desktop\usr\web\qa_data\計劃總覽.txt"
qa_file   = r"C:\Users\Kuo\desktop\usr\web\qa_data\qa_custom.txt"

SDG_NAMES = {
    1:"消除貧窮", 2:"零飢餓", 3:"良好健康與福祉",
    4:"優質教育", 5:"性別平等", 6:"乾淨用水及衛生",
    7:"可負擔及乾淨能源", 8:"合宜工作與經濟成長", 9:"產業創新與基礎設施",
    10:"減少不平等", 11:"永續城市及社區", 12:"負責任的消費及生產",
    13:"氣候行動", 14:"水下生物", 15:"陸地生物",
    16:"和平正義與強大機構", 17:"全球夥伴關係",
}

def parse_sdgs_from_txt(filepath):
    content = None
    for enc in ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "cp950"]:
        try:
            with open(filepath, encoding=enc, errors="strict") as f:
                content = f.read()
            break
        except:
            pass
    if not content:
        with open(filepath, encoding="cp950", errors="replace") as f:
            content = f.read()
    m = re.search(r"SDGs?\s*關聯議題\s*[：:]?\s*([^\n]+)", content)
    if not m:
        return set()
    text = m.group(1)
    nums = set()
    for match in re.finditer(r"(?:☑|■)\s*(\d{1,2})(?=\D)", text):
        n = int(match.group(1))
        if 1 <= n <= 17:
            nums.add(n)
    return nums

# Build TXT SDG lookup
txt_files = [f for f in os.listdir(txt_dir) if f.endswith(".txt")]
txt_sdg_lookup = {}
for fname in txt_files:
    path = os.path.join(txt_dir, fname)
    sdgs = parse_sdgs_from_txt(path)
    m = re.match(r"^(.+?)_(.+?)\(", fname)
    if m:
        school = m.group(1).replace("國立", "").strip()
        plan   = m.group(2)[:12]
        txt_sdg_lookup[(school, plan)] = sdgs

no_sdg = [f for f in txt_files if not parse_sdgs_from_txt(os.path.join(txt_dir, f))]
print(f"TXT files with SDG: {len(txt_files)-len(no_sdg)}/{len(txt_files)}")

def find_txt_sdgs(ov_school, ov_plan):
    school_s = ov_school.replace("國立", "").strip()
    plan_p   = ov_plan[:12]
    key = (school_s, plan_p)
    if key in txt_sdg_lookup:
        return txt_sdg_lookup[key]
    for (ts, tp), sdgs in txt_sdg_lookup.items():
        if (school_s in ts or ts in school_s) and plan_p[:8] == tp[:8]:
            return sdgs
    return None

shutil.copy2(plan_file, plan_file + ".bak_final")

with open(plan_file, encoding="utf-8") as f:
    lines = f.read().split("\n")

boundary = len(lines)
for i, line in enumerate(lines):
    if "===" in line and ".txt" in line:
        boundary = i
        break

index_lines = lines[:boundary]
rest_lines  = lines[boundary:]

def parse_sections(idx_lines):
    secs = {}
    cur  = None
    for i, line in enumerate(idx_lines):
        m2 = re.match(r"SDG(\d+)", line)
        if m2 and "（共" in line:
            cur = int(m2.group(1))
            secs[cur] = {"header_idx": i, "entries": []}
        elif cur:
            em = re.match(r"\s+\d+\.\s+(.+?)\s+[—－–]\s+(.+)", line)
            if em:
                secs[cur]["entries"].append({
                    "new_idx": i,
                    "school":  em.group(1).strip(),
                    "plan":    em.group(2).strip(),
                })
    return secs

sections = parse_sections(index_lines)

lines_to_remove = set()
for sdg_num, sec in sections.items():
    for entry in sec["entries"]:
        txt_sdgs = find_txt_sdgs(entry["school"], entry["plan"])
        if txt_sdgs is not None and len(txt_sdgs) > 0:
            if sdg_num not in txt_sdgs:
                lines_to_remove.add(entry["new_idx"])

# Find missing entries
all_ov_plans = defaultdict(set)
for sdg_num, sec in sections.items():
    for entry in sec["entries"]:
        all_ov_plans[(entry["school"], entry["plan"][:12])].add(sdg_num)

plans_to_add = defaultdict(list)
for (school, plan_p), ov_sdgs in all_ov_plans.items():
    txt_sdgs = find_txt_sdgs(school, plan_p)
    if txt_sdgs:
        for sdg_num in txt_sdgs - ov_sdgs:
            full_plan = plan_p
            for sec in sections.values():
                for e in sec["entries"]:
                    if e["school"] == school and e["plan"][:12] == plan_p:
                        full_plan = e["plan"]
                        break
            plans_to_add[sdg_num].append((school, full_plan))

print(f"Lines to remove: {len(lines_to_remove)}, entries to add: {sum(len(v) for v in plans_to_add.values())}")
for sdg, plist in sorted(plans_to_add.items()):
    for s, p in plist:
        print(f"  Add SDG{sdg}: {s} — {p[:30]}")

new_index = [line for i, line in enumerate(index_lines) if i not in lines_to_remove]
new_secs  = parse_sections(new_index)

for sdg_num in sorted(plans_to_add.keys()):
    for (school, plan) in plans_to_add[sdg_num]:
        if sdg_num not in new_secs:
            continue
        already = any(e["school"] == school and e["plan"][:10] == plan[:10]
                      for e in new_secs[sdg_num]["entries"])
        if not already:
            entries  = new_secs[sdg_num]["entries"]
            last_idx = entries[-1]["new_idx"] if entries else new_secs[sdg_num]["header_idx"]
            new_index.insert(last_idx + 1, f"  0. {school} — {plan}")
            new_secs = parse_sections(new_index)

header_map = {sec["header_idx"]: sdg for sdg, sec in new_secs.items()}
entry_map  = {}
for sdg, sec in new_secs.items():
    for j, e in enumerate(sec["entries"], 1):
        entry_map[e["new_idx"]] = j

final = []
for i, line in enumerate(new_index):
    if i in header_map:
        sdg   = header_map[i]
        count = len(new_secs[sdg]["entries"])
        final.append(re.sub(r"（共\s*\d+\s*個計畫）", f"（共 {count} 個計畫）", line))
    elif i in entry_map:
        final.append(re.sub(r"^(\s+)\d+\.", f"\\g<1>{entry_map[i]}.", line))
    else:
        final.append(line)

with open(plan_file, "w", encoding="utf-8") as f:
    f.write("\n".join(final + rest_lines))

total = sum(len(sec["entries"]) for sec in new_secs.values())
print(f"\n計劃總覽 updated. Total: {total}")
for sdg in range(1, 18):
    if sdg in new_secs:
        print(f"  SDG{sdg}: {len(new_secs[sdg]['entries'])}")

# Regenerate qa_custom.txt SDG section
sections2 = {sdg: [(e["school"], e["plan"]) for e in sec["entries"]]
             for sdg, sec in new_secs.items()}

with open(qa_file, encoding="utf-8") as f:
    qa_content = f.read()
marker    = "\n# ── SDG 對應計畫索引（自動產生）"
cut_pos   = qa_content.find(marker)
base_content = qa_content[:cut_pos] if cut_pos != -1 else qa_content

blocks = ["\n# ── SDG 對應計畫索引（自動產生）────────────────────\n"]
overview_lines = ["**114 年度 USR 計畫 — 各 SDG 計畫數量概覽**\n"]
for num in range(1, 18):
    name      = SDG_NAMES.get(num, "")
    plans     = sections2.get(num, [])
    uni_count = len({p[0] for p in plans})
    overview_lines.append(f"- SDG{num:02d}「{name}」：{len(plans)} 個計畫 / {uni_count} 間學校")
overview_lines.append(f"\n合計（含重複）：{total} 筆（一所學校可對應多個 SDG）")
blocks.append("Q: 所有SDG計畫概覽|各SDG有幾個計畫|SDG統計|每個SDG的計畫數|全部SDG\nA: "
              + "\n".join(overview_lines) + "\n")

for num in range(1, 18):
    name  = SDG_NAMES.get(num, "")
    plans = sections2.get(num, [])
    if not plans:
        continue
    uni_set = {p[0] for p in plans}
    q_line  = (f"Q: SDG{num}有哪些計畫|SDG{num}對應的大學|SDG{num}的學校|"
               f"SDG{num}計畫清單|SDG{num}有幾個計畫|SDG{num}幾個|SDG{num}共幾|"
               f"SDG{num}{name}|SDG{num} {name}")
    a_lines = [f"**SDG{num}「{name}」** 共有 {len(plans)} 個計畫，涉及 {len(uni_set)} 間學校：\n"]
    for i, (uni, plan) in enumerate(plans, 1):
        a_lines.append(f"{i}. **{uni}**　{plan}")
    blocks.append(q_line + "\nA: " + "\n".join(a_lines) + "\n")

with open(qa_file, "w", encoding="utf-8") as f:
    f.write(base_content + "\n".join(blocks))

print("\nqa_custom.txt SDG section regenerated.")
