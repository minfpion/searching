# app/routers/video.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.services.video_to_db import extract_frames_for_processing, process_video_to_db
import os
from typing import Dict

router = APIRouter(prefix="/video", tags=["Video Processing"])

# 폴더와 DB 설정
FRAME_FOLDER = "test_frames"
VIDEOS_FOLDER = "videos"  # 비디오 파일 저장 폴더
DB_NAME = "test.sqlite"
TABLE_NAME = "video_embeddings"
TARGET_FPS = 1

# 비디오 저장 폴더가 없으면 생성
if not os.path.exists(VIDEOS_FOLDER):
    os.makedirs(VIDEOS_FOLDER)

@router.post("/process")
async def process_video(file: UploadFile = File(...)) -> Dict:
    """
    업로드된 비디오를 처리하여 프레임을 추출하고 임베딩을 생성합니다.
    """
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="비디오 파일만 업로드 가능합니다.")

    try:
        try:
            # 원본 파일 이름 그대로 사용
            video_filename = file.filename
            video_name = os.path.splitext(video_filename)[0]
            video_path = os.path.join(VIDEOS_FOLDER, video_filename)
            content = await file.read()

            # 비디오 파일을 videos 폴더에 저장
            with open(video_path, "wb") as buffer:
                buffer.write(content)

            # 프레임 추출
            extracted_count = extract_frames_for_processing(video_path, FRAME_FOLDER, TARGET_FPS)
            
            if extracted_count == 0:
                # 실패 시 비디오 파일 삭제
                if os.path.exists(video_path):
                    os.remove(video_path)
                raise HTTPException(status_code=400, detail="비디오에서 프레임을 추출할 수 없습니다.")

            # 추출된 프레임을 처리하고 DB에 저장
            process_video_to_db(video_path, FRAME_FOLDER, DB_NAME, TABLE_NAME, extracted_count)

            # FAISS 인덱스 갱신
            from app.utils.faiss_utils import load_vectors_from_db, build_faiss_index
            from app.routers.search import vectors, ids, vnames, findices, index
            
            # 전역 변수 업데이트
            vectors, ids, vnames, findices = load_vectors_from_db(DB_NAME, TABLE_NAME)
            if len(vectors) > 0:
                index = build_faiss_index(vectors)

            return {
                "status": "success",
                "message": f"비디오 처리가 완료되었습니다. {extracted_count}개의 프레임이 추출되었습니다.",
                "frames_extracted": extracted_count,
                "video_name": video_name  # 저장된 비디오 이름 반환
            }

        finally:
            pass  # 비디오 파일을 유지

        return {
            "status": "success",
            "message": f"비디오 처리가 완료되었습니다. {extracted_count}개의 프레임이 추출되었습니다.",
            "frames_extracted": extracted_count,
            "video_name": video_filename  # 저장된 비디오 파일 이름 반환
        }

    except Exception as e:
        # 처리 실패 시에만 비디오 파일 삭제
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=f"비디오 처리 중 오류가 발생했습니다: {str(e)}")

@router.get("/frames/{video_name}/{frame_index}")
async def get_frame(video_name: str, frame_index: int):
    """
    특정 비디오의 프레임 번호에 해당하는 이미지를 반환합니다.
    """
    try:
        # 모든 프레임이 test 폴더에 저장되므로 경로 수정
        frame_path = os.path.join(FRAME_FOLDER, "test", f"frame_{frame_index:05d}.jpg")
        if not os.path.exists(frame_path):
            raise HTTPException(status_code=404, detail=f"프레임을 찾을 수 없습니다: {frame_index}")
        return FileResponse(frame_path, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"프레임 로드 중 오류가 발생했습니다: {str(e)}")

@router.get("/stream/{video_name}")
async def stream_video(video_name: str):
    """
    비디오 파일을 스트리밍합니다.
    """
    video_path = os.path.join(VIDEOS_FOLDER, video_name)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="비디오 파일을 찾을 수 없습니다.")
    
    return FileResponse(video_path, media_type="video/mp4")

@router.get("/status")
async def get_processing_status() -> Dict:
    """
    현재 처리된 비디오와 프레임의 상태를 확인합니다.
    """
    try:
        # 각 비디오 폴더별 프레임 개수를 계산
        frame_counts = {}
        if os.path.exists(FRAME_FOLDER):
            for video_folder in os.listdir(FRAME_FOLDER):
                video_path = os.path.join(FRAME_FOLDER, video_folder)
                if os.path.isdir(video_path):
                    frame_counts[video_folder] = len([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
        total_frames = sum(frame_counts.values())
        
        import sqlite3
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        db_count = cur.fetchone()[0]
        
        cur.execute(f"SELECT DISTINCT video_name FROM {TABLE_NAME}")
        videos = [row[0] for row in cur.fetchall()]
        
        conn.close()

        return {
            "status": "success",
            "frames_by_video": frame_counts,
            "total_frames": total_frames,
            "embeddings_in_db": db_count,
            "processed_videos": videos
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상태 확인 중 오류가 발생했습니다: {str(e)}")