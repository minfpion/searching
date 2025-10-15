import sqlite3
import os

DB = os.path.join(os.path.dirname(__file__), "test.sqlite")
print("DB 경로:", DB)
if not os.path.exists(DB):
    print(">>> test.sqlite 파일이 존재하지 않습니다!")
    raise SystemExit

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
print("테이블 목록:", tables)

# 테이블별 row 수 확인
for t in tables:
    name = t[0]
    cur.execute(f"SELECT COUNT(*) FROM {name}")
    cnt = cur.fetchone()[0]
    print(f" - {name}: {cnt} rows")

# sample 5개 출력 (video_embeddings가 있으면)
if ('video_embeddings',) in tables:
    cur.execute("SELECT id, video_name, frame_index FROM video_embeddings LIMIT 5;")
    print("샘플 rows:", cur.fetchall())

conn.close()
