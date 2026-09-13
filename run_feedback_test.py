# -*- coding: utf-8 -*-
"""
run_feedback_test.py
批次跑回饋文件裡的測試問題，輸出成 Word 文件。

使用：
    python run_feedback_test.py
輸出：feedback_results.docx（與本腳本同目錄）
"""

import json, os, re, subprocess, sys, time, uuid
from pathlib import Path

import requests
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

sys.stdout.reconfigure(encoding="utf-8")

# ── 設定 ────────────────────────────────────────────────
SERVER_URL  = "http://127.0.0.1:8080"
ASK_URL     = f"{SERVER_URL}/ask"
LOGIN_URL   = f"{SERVER_URL}/login"
YEAR        = "114"
OUT_DOCX    = Path(__file__).parent / "feedback_results.docx"
TIMEOUT_SEC = 120   # 每題最長等待秒數

# ── 97 題完整清單（只跑 ● 類） ────────────────────────────
# follow_of: 接續題的前一題編號（None=獨立題）
QUESTIONS = [
    (1,  "●", "以淺山茶產業為載體之韌性農村社會實踐人才培力計畫之成效評估機制", None),
    (2,  "●", "尋找雲林農村生命力 – 永續農村營造計畫的計畫類型與相關計畫成果", None),
    (3,  "●", "食農教育相關計畫有哪些？", None),
    (5,  "●", "長庚科技大學-多元新視界，攜手嘉移人計畫的執行場域有哪些?", None),
    (8,  "●", "找出高屏澎東區的計畫，其內容跟生物多樣性相關有哪些計畫?", None),
    (9,  "●", "請提供澎湖科技大學USR計畫的執行場域", None),
    (10, "●", "114年一共有多少件USR計畫?", None),
    (11, "●", "桃竹苗宜花區的計畫，有哪些與旅遊的議題相關？", None),
    (13, "●", "請幫我查詢，有哪些計畫的實踐場域，涉及桃園的中壢？", None),
    (14, "●", "有哪些計畫涉及食農教育", None),
    (17, "●", "有哪些計畫的實踐場域是國外的學校，請列出計畫的學校、計畫名稱及國外場域名稱", None),
    (18, "●", "接續上一題，請只要列出實踐場域是國外的學校，呈現查詢結果清單只要計畫學校及其名稱之外，最重要是列出國外場域名稱，請不要列出計畫摘要內容", 17),
    (19, "●", "接續上題，我需要查詢有哪些計畫的實踐區域：是東南亞等國家、歐美、東北亞國家等其他海外地區，請列出這些計畫學校、計畫名稱、東南亞等國家、歐美、東北亞國家等其他海外地區的場域名稱", 18),
    (21, "●", "南臺科技大學有多少USR計畫", None),
    (22, "●", "有多少計畫已完成SROI成效評估?只要提供數量就好。", None),
    (23, "●", "雲嘉南區有哪些計畫涉及原住民議題？", None),
    (26, "●", "有哪些環境生態保育的計畫？", None),
    (27, "●", "續上，請更加詳細提供關於「弘光科技大學：共築中部沿海里仁淨零之美-深耕低碳+韌性+永續家園」計畫相關資料。", 26),
    (28, "●", "續上，學校如何吸引在地居民共同參與空污監控與改善？", 27),
    (29, "●", "請推薦我校務做得好的學校（教學與研究品質）", None),
    (30, "●", "有關數位教學的計畫有哪些", None),
    (31, "●", "雲科的計畫有哪些", None),
    (33, "●", "目前257份計畫報告中，哪些涉及「原鄉教育」的議題，請逐一列出「學校名稱」、「計畫名稱」及哪些地方有直接相關的「摘要說明」。", None),
    (34, "●", "目前257份計畫報告中，哪些計畫同時涉及「原鄉教育」與「偏鄉教育」的議題，而且實踐場域地處高雄、屏東或台東，請逐一列出「學校名稱」與「計畫名稱」。", None),
    (38, "●", "請問涉及偏鄉教育計畫有哪些?", None),
    (39, "●", "請問涉及食農教育計畫有哪些?", None),
    (40, "●", "請幫我統計有關親子教育課程有哪些?", None),
    (41, "●", "有哪些計畫有在做食農教育?", None),
    (42, "●", "接續上題，請幫我彙整總共有幾間學校，然後每間學校分別有幾個計畫在做食農教育", 41),
    (43, "●", "請幫我列出高雄醫學大學的USR計畫中，於114年辦理過的國際交流活動。", None),
    (44, "●", "請協助查詢桃竹宜花區USR計畫案以原住民為對象的計畫有哪些計畫案？", None),
    (45, "●", "請針對第四期計畫案以社區長照為議題的計畫案有幾個計畫案？", None),
    (46, "●", "清華大學114年全校的USR計畫案有哪些場域？", None),
    (49, "●", "全國第四期USR計畫案涉及地方創生的計畫有哪幾個計畫案？", None),
    (50, "●", "桃宜竹花區第四期通過USR計畫案，有完成SROI的計畫案有幾個？", None),
    (56, "●", "114年桃竹宜花區計畫案全年經費預算使用狀況，未達90%的有幾個計畫案", None),
    (57, "●", "114年全國USR計畫案中有開設學分學程，有哪幾個計畫案？", None),
    (58, "●", "114年全國USR計畫案，有哪幾個計畫案執行沒有場域？", None),
    (60, "●", "114年全國USR計畫案，有哪幾所科技大學的計畫案的場域在新竹縣和新竹市？", None),
    (61, "●", "114年全國USR計畫案，有哪幾所大學計畫案的主題在新住民和移工？", None),
    (62, "●", "114年全國USR計畫案，主題為文創議題，團隊成員專業跨領域超過三個以上，場域在桃園市、新竹縣及新竹縣有哪幾個計畫？", None),
    (63, "●", "全國USR計畫案屬於永續發展類國際合作型計畫場域在越南的科技大學有哪幾個計畫案？", None),
    (67, "●", "114年永續發展國際合作型計畫案，在海外國際的場域超過兩個國家以上的計畫案有哪幾個計畫？", None),
    (68, "●", "114年USR計畫中，有前往日本做參觀學習及交流的計畫案有哪幾個計畫案？", None),
    (69, "●", "114年全國USR計畫案，場域在桃園縣復興鄉的計畫案有哪幾個計畫？", None),
    (72, "●", "請列出清華大學114年USR計畫案中有做計畫變更的計畫有哪幾個？變更什麼？", None),
    (74, "●", "北北基有哪些計畫與社工合作", None),
    (75, "●", "社工跟USR團隊合作有什麼要注意的嗎?", None),
    (76, "●", "接續上題，如果我想找政治大學的教授合作，有聯絡方式嗎", 75),
    (77, "●", "我想知道哪些計畫團隊可以協助推動文化保存", None),
    (79, "●", "有哪些計畫跟河川保育有關", None),
    (80, "●", "有哪些計畫辦理的活動是有搭配課程的", None),
    (82, "●", "有哪些計畫與遺傳性表皮分解性水皰症有關", None),
    (83, "●", "有哪些計畫是以海洋教育為主?", None),
    (84, "●", "接續前題，請只提供計畫內容是實施海洋教育課程，請不再將環境教育相關的計畫列入", 83),
    (85, "●", "接續前題，請列出有關環境教育的計畫，但要扣除海洋教育的議題，請呈現計畫學校名稱及計畫名稱即可", 84),
    (88, "●", "哪一個學校的固定實踐場域最多", None),
    (89, "●", "請問國立臺灣大學「社區尺度之環境舒適度健檢與在地氣候行動：科學社會溝通作為方法」這支計畫的教育部114年度核定補助經費各項金額是多少?", None),
    (90, "●", "接續上題，請問人事費、業務費、設備及投資、學校配合款經費分別多少錢?", 89),
    (91, "●", "請問有關偏鄉教育計畫有哪些計畫?", None),
    (92, "●", "在地老化與社區長照服務計畫有哪些？", None),
    (93, "●", "接續上題，我要跟長照據點有關的", 92),
    (94, "●", "接續上題，能再更具焦像是提到長照abc據點或是照顧網之類的計畫嗎", 93),
    (95, "●", "查詢動物相關計畫", None),
    (96, "●", "提問深耕型計畫總數量", None),
    (97, "●", "計畫場域在台北的計畫", None),
]


# ── SSE 解析 ─────────────────────────────────────────────
def query_server(session: requests.Session, question: str, conv_id: str | None = None) -> str:
    """向 /ask 發送問題，解析 SSE 回傳完整答案文字。"""
    payload = {"question": question, "year": YEAR}
    if conv_id:
        payload["conv_id"] = conv_id
        payload["use_context"] = True

    try:
        resp = session.post(ASK_URL, json=payload, stream=True, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
    except Exception as e:
        return f"[ERROR] 連線失敗：{e}"

    parts = []
    mode = ""
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("data: "):
                try:
                    d = json.loads(raw[6:])
                except Exception:
                    continue
                t = d.get("type", "")
                if t == "chunk":
                    parts.append(d.get("text", ""))
                elif t == "done":
                    mode = d.get("mode", "")
                    break
    except Exception as e:
        parts.append(f"\n[STREAM ERROR] {e}")

    answer = "".join(parts).strip()
    if mode:
        answer = f"[mode={mode}]\n{answer}"
    return answer or "[無回應]"


# ── Word 輸出 ─────────────────────────────────────────────
def add_heading(doc: Document, text: str, level: int = 1):
    p = doc.add_heading(text, level=level)
    p.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)


def add_label(doc: Document, label: str, value: str):
    p = doc.add_paragraph()
    run_l = p.add_run(label)
    run_l.bold = True
    run_l.font.size = Pt(10)
    run_v = p.add_run(value)
    run_v.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(2)


def add_answer(doc: Document, answer: str):
    lines = answer.split("\n")
    for line in lines:
        p = doc.add_paragraph(line)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        for run in p.runs:
            run.font.size = Pt(10)
    doc.add_paragraph()


# ── 伺服器啟動 ────────────────────────────────────────────
def start_server() -> subprocess.Popen | None:
    """嘗試啟動 Flask 伺服器，若已在跑則跳過。"""
    try:
        requests.get(SERVER_URL, timeout=3)
        print("[INFO] 伺服器已在運行")
        return None
    except Exception:
        pass

    print("[INFO] 啟動 Flask 伺服器...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(Path(__file__).parent),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等待伺服器就緒（最多 90 秒）
    for _ in range(90):
        time.sleep(1)
        try:
            requests.get(SERVER_URL, timeout=2)
            print("[INFO] 伺服器已就緒")
            return proc
        except Exception:
            pass

    print("[ERROR] 伺服器啟動逾時")
    proc.terminate()
    return None


# ── 主程序 ────────────────────────────────────────────────
def main():
    proc = start_server()

    sess = requests.Session()

    # 嘗試登入（若伺服器設有密碼）
    try:
        r = sess.get(f"{SERVER_URL}/", timeout=5)
        if "login" in r.url.lower() or r.status_code == 401:
            username = os.getenv("SITE_USERNAME", "")
            password = os.getenv("SITE_PASSWORD", "")
            if username and password:
                sess.post(LOGIN_URL, json={"username": username, "password": password}, timeout=10)
    except Exception:
        pass

    # 建立 Word 文件
    doc = Document()
    doc.core_properties.title = "USR 系統測試結果"
    style = doc.styles["Normal"]
    style.font.name = "微軟正黑體"
    style.font.size = Pt(10)

    title = doc.add_heading("USR 計畫書查詢系統 ── 第一階段測試結果", 0)
    title.runs[0].font.size = Pt(16)
    doc.add_paragraph(f"測試日期：{time.strftime('%Y-%m-%d')}　　共 {len(QUESTIONS)} 題（● 類）")
    doc.add_paragraph()

    # 用於接續題的 conv_id 管理
    conv_map: dict[int, str] = {}   # num → conv_id

    total = len(QUESTIONS)
    for idx, (num, scope, question, follow_of) in enumerate(QUESTIONS, 1):
        print(f"[{idx:02d}/{total}] #{num} {question[:50]}...")

        # 決定 conv_id
        if follow_of and follow_of in conv_map:
            cid = conv_map[follow_of]
        else:
            cid = str(uuid.uuid4())
        conv_map[num] = cid

        answer = query_server(sess, question, conv_id=cid if follow_of else None)

        # 寫入 Word
        doc.add_heading(f"#{num} {question[:80]}", level=2)
        if follow_of:
            p = doc.add_paragraph(f"（接續 #{follow_of}）")
            p.runs[0].font.color.rgb = RGBColor(0x88, 0x88, 0x88)
            p.runs[0].font.italic = True
        add_answer(doc, answer)

        # 儲存進度（每 5 題存一次）
        if idx % 5 == 0:
            doc.save(str(OUT_DOCX))
            print(f"  → 已儲存進度（{idx}/{total}）")

    doc.save(str(OUT_DOCX))
    print(f"\n✓ 完成！結果已儲存至 {OUT_DOCX}")

    if proc:
        proc.terminate()
        print("[INFO] 伺服器已關閉")


if __name__ == "__main__":
    main()
