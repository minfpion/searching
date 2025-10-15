# app/routers/search.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import os

# 우리가 만든 유틸리티 함수들을 불러옵니다.
from app.utils.clip_utils import image_to_clip_embedding
from app.utils.faiss_utils import load_vectors_from_db, build_faiss_index, search_similar

# FastAPI 라우터 객체 생성
router = APIRouter(prefix="/search", tags=["Search"])

# 전역 변수 선언
vectors = None
ids = None
vnames = None
findices = None
index = None

# --- 서버가 시작될 때 단 한 번만 실행되는 부분 ---
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "test.sqlite")
DB_PATH = os.path.abspath(DB_PATH)
TABLE_NAME = "video_embeddings"

print("--- 서버 시작: DB에서 벡터를 로드하고 FAISS 인덱스를 구축합니다. ---")

try:
    # DB에서 모든 벡터 데이터를 메모리로 불러옵니다.
    vectors, ids, vnames, findices = load_vectors_from_db(DB_PATH, TABLE_NAME)
    # 불러온 벡터로 FAISS 인덱스를 생성합니다.
    index = build_faiss_index(vectors)
    print("--- FAISS 인덱스 준비 완료. API 서버가 요청을 받을 준비가 되었습니다. ---")
except Exception as e:
    print(f"!!! 경고: FAISS 인덱스 초기화 실패: {str(e)}")

# ----------------------------------------------------

@router.post("/")
async def search_similar_frames(file: UploadFile = File(...)):
    """
    사용자가 업로드한 이미지와 가장 유사한 동영상 프레임을 검색합니다.
    """
    # 전역 변수 사용 선언
    global vectors, ids, vnames, findices, index

    # DB 재로드 시도
    try:
        vectors, ids, vnames, findices = load_vectors_from_db(DB_PATH, TABLE_NAME)
        index = build_faiss_index(vectors)
    except Exception as e:
        print(f"DB 재로드 실패: {str(e)}")
        
    if index is None:
        raise HTTPException(status_code=503, 
                          detail="검색 시스템이 아직 준비되지 않았습니다. 먼저 비디오를 처리해주세요.")

    # 파일 형식 검증
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    try:
        # 1. 사용자가 업로드한 이미지 파일을 읽습니다.
        image_bytes = await file.read()
        image_io = io.BytesIO(image_bytes)
        image = Image.open(image_io)
        
        # 이미지 포맷 확인 및 변환
        if not hasattr(image, 'mode') or image.mode != 'RGB':
            image = image.convert('RGB')

        # 2. 이미지를 CLIP 벡터로 변환합니다.
        query_vec = image_to_clip_embedding(image)
        if query_vec is None:
            raise HTTPException(status_code=500, detail="이미지 임베딩 생성에 실패했습니다.")

        # 3. FAISS에서 유사한 벡터를 검색합니다.
        distances, indices = search_similar(index, query_vec, top_k=5)

        # 4. 검색 결과를 사용자가 보기 좋게 정리합니다.
        results = []
        for i, idx in enumerate(indices):
            # DB에 저장된 video_name은 이미 전체 파일명을 포함
            video_filename = vnames[idx]
            results.append({
                "rank": i + 1,
                "video_name": video_filename,
                "frame_index": int(findices[idx]),
                "similarity_score": float(1 - distances[i])
            })
        
        return {"results": results}

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, 
                          detail="이미지 파일을 인식할 수 없습니다. 다른 이미지를 시도해주세요.")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        raise HTTPException(status_code=500, 
                          detail="이미지 처리 중 오류가 발생했습니다.")