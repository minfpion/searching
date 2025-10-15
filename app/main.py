# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routers import search, video
from app.utils.clip_utils import ensure_table_exists
import os

app = FastAPI(title="VIDEO SEARCHING API")

# 서버 시작 시 필요한 디렉토리와 테이블 생성
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "test.sqlite")
DB_PATH = os.path.abspath(DB_PATH)
ensure_table_exists(DB_PATH, "video_embeddings")

# 정적 파일 제공을 위한 디렉토리 마운트
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 라우터 등록
app.include_router(video.router)  # 비디오 처리 라우터
app.include_router(search.router)  # 검색 라우터

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/system-check")
async def system_check():
    """시스템 상태를 확인합니다."""
    try:
        # DB 존재 여부 확인
        if not os.path.exists("test.sqlite"):
            return {"status": "not_ready", "message": "데이터베이스가 아직 생성되지 않았습니다. 먼저 비디오를 처리해주세요."}
        
        # 프레임 폴더 존재 여부 확인
        if not os.path.exists("test_frames"):
            os.makedirs("test_frames")
        
        import sqlite3
        conn = sqlite3.connect("test.sqlite")
        cur = conn.cursor()
        
        # 테이블 존재 여부 및 데이터 확인
        try:
            cur.execute("SELECT COUNT(*) FROM video_embeddings")
            count = cur.fetchone()[0]
            if count == 0:
                return {"status": "not_ready", "message": "데이터베이스에 처리된 비디오 프레임이 없습니다. 먼저 비디오를 처리해주세요."}
        except sqlite3.OperationalError:
            return {"status": "not_ready", "message": "데이터베이스 테이블이 초기화되지 않았습니다. 먼저 비디오를 처리해주세요."}
        finally:
            conn.close()

        return {"status": "ready", "message": "시스템이 정상적으로 동작 중입니다."}
    except Exception as e:
        return {"status": "error", "message": f"시스템 상태 확인 중 오류가 발생했습니다: {str(e)}"}