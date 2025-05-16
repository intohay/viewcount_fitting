import csv
from pathlib import Path
import sys
import re

import requests
from bs4 import BeautifulSoup

# コマンドライン引数からURLを取得
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <URL>")
    sys.exit(1)
URL = sys.argv[1]

html = requests.get(URL, timeout=30).text
soup = BeautifulSoup(html, "html.parser")

# h1.video-title からファイル名を決定
h1 = soup.find("h1", class_="video-title")
if not h1:
    raise RuntimeError("h1.video-title が見つかりませんでした")
title = h1.get_text(strip=True)
# ファイル名として使えない文字を除去
safe_title = re.sub(r'[\\/:*?"<>|]', '', title)
OUT = Path(f"{safe_title}_views.csv")

# ① テーブル行を取得
rows = soup.select("#video-details-table tbody tr")
if not rows:
    raise RuntimeError("履歴テーブル (#video-details-table) が見つかりませんでした")

records = []
for tr in rows:
    cells = [td.get_text(strip=True) for td in tr.find_all("td")]
    if len(cells) < 2:   # Date と Views の 2 列さえあれば良い
        continue
    dt   = cells[0]                              # 例: '2025-05-17 08:00'
    views = int(cells[1].replace(",", ""))       # カンマ除去して整数化
    records.append((dt, views))

# ② 日付昇順に並べ替え
records.sort(key=lambda x: x[0])

# 連続する同じ views の値が2回以上続く場合、最後の1回だけを残す
filtered_records = []
for i, (dt, views) in enumerate(records):
    # 次の行が同じ views ならスキップ（最後の1つだけ残す）
    if i + 1 < len(records) and views == records[i + 1][1]:
        continue
    filtered_records.append((dt, views))

# ③ CSV 出力
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["datetime", "views"])
    w.writerows(filtered_records)

print(f"wrote {len(filtered_records):,} rows to {OUT.resolve()}")
