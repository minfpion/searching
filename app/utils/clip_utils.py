import torch
import numpy as np
import sqlite3
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# -----------------------------------------------------------
# 1. CLIP 모델 로드 및 전처리기 정의
# -----------------------------------------------------------
# CLIP 모델을 불러옵니다. 'openai/clip-vit-base-patch32'는 가장 일반적인 모델
# CPU 메모리를 절약하기 위해 여기서는 모델을 전역 변수
# GPU가 있다면 device = "cuda"로 설정가능
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"

# CLIP 모델 객체
CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(device)
# CLIP 모델이 요구하는 형태로 이미지를 변환해주는 전처리기(Processor)
CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name, use_fast=True)


def image_to_clip_embedding(image_or_path) -> np.ndarray:
    """
    이미지 파일 경로나 PIL Image 객체를 받아 CLIP 모델을 이용해 512차원 벡터 임베딩으로 변환합니다.

     image_or_path: 입력 이미지 파일 경로 또는 PIL Image 객체
     return: 512차원 CLIP 임베딩 벡터 (NumPy 배열 형태)
    """
    try:
        # 1. 이미지 로드 또는 사용
        if isinstance(image_or_path, str):
            # 파일 경로가 주어진 경우
            try:
                image = Image.open(image_or_path)
            except FileNotFoundError:
                print(f"오류: 이미지 파일 '{image_or_path}'을(를) 찾을 수 없습니다.")
                return None
            except Exception as e:
                print(f"오류: 이미지 로드 중 문제가 발생했습니다: {e}")
                return None
        else:
            # PIL Image 객체가 직접 주어진 경우
            image = image_or_path

        # 이미지 포맷 확인 및 변환
        if not hasattr(image, 'mode') or image.mode != 'RGB':
            image = image.convert('RGB')

        # 2. 이미지 전처리
        # CLIP 모델의 입력 형식(크기, 정규화 등)에 맞게 이미지를 변환
        # return_tensors='pt'로 설정하여 PyTorch 텐서 형태로 반환
        inputs = CLIP_PROCESSOR(images=image, return_tensors="pt").to(device)

        # 3. 임베딩 생성 (벡터 변환)
        with torch.no_grad():
            # CLIP 모델의 get_image_features를 사용하여 512차원 임베딩을 추출
            image_features = CLIP_MODEL.get_image_features(pixel_values=inputs.pixel_values)

        # 4. 결과 정리
        # PyTorch 텐서를 CPU로 이동시키고, NumPy 배열로 변환한 후 정규화
        embedding = image_features.cpu().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    except Exception as e:
        print(f"오류: 이미지 처리 중 문제가 발생했습니다: {e}")
        return None

    # 3. 임베딩 생성 (벡터 변환)
    with torch.no_grad():
        # CLIP 모델의 get_image_features를 사용하여 512차원 임베딩을 추출
        image_features = CLIP_MODEL.get_image_features(pixel_values=inputs.pixel_values)

    # 4. 결과 정리
    # PyTorch 텐서를 CPU로 이동시키고, NumPy 배열로 변환한 후 정규화
    # 임베딩은 보통 L2-정규화(길이가 1)되어 사용
    embedding = image_features.cpu().numpy().flatten()
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


def ensure_table_exists(db_name: str, table_name: str):
    """
    데이터베이스에 테이블이 없으면 생성합니다.
    
    Args:
        db_name: 데이터베이스 파일 이름
        table_name: 생성할 테이블 이름
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # 테이블이 없으면 생성
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT NOT NULL,
            frame_index INTEGER NOT NULL,
            vector_blob BLOB NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

def save_embedding_to_db(db_name: str, table_name: str, video_name: str, frame_index: int, embedding: np.ndarray):
    """
    CLIP 임베딩 벡터를 SQLite 데이터베이스에 BLOB 형태로 저장.

     db_name: 데이터베이스 파일 이름 (예: 'vector_db.sqlite')
     table_name: 테이블 이름 (예: 'video_embeddings')
     video_name: 원본 동영상 파일 이름
     frame_index: 추출된 프레임의 순서 번호
     embedding: 저장할 CLIP 임베딩 벡터 (NumPy 배열)
    """
    # 1. 데이터베이스 연결
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 2. 테이블 생성 (테이블이 없다면)
    # BLOB 형태의 데이터를 저장하기 위해 'VECTOR_BLOB' 필드를 정의
    # REAL 타입은 검색에 용이하지만, BLOB은 대용량 바이너리 데이터 저장에 효율적
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT NOT NULL,
            frame_index INTEGER NOT NULL,
            vector_blob BLOB NOT NULL
        )
    """)
    conn.commit()

    # 3. 임베딩 벡터를 BLOB 형태로 변환
    # NumPy 배열을 SQLite에 저장 가능한 바이너리(BLOB) 형태로 변환
    embedding_blob = embedding.tobytes()

    # 4. 데이터 삽입
    cursor.execute(f"""
        INSERT INTO {table_name} (video_name, frame_index, vector_blob)
        VALUES (?, ?, ?)
    """, (video_name, frame_index, embedding_blob))

    # 5. 변경사항 저장 및 연결 종료
    conn.commit()
    conn.close()


def load_embedding_from_db(db_name: str, table_name: str, row_id: int) -> np.ndarray:
    """
    SQLite 데이터베이스에서 BLOB 형태의 벡터를 불러와 NumPy 배열로 복원
    (선택 사항: 데이터 확인 및 검색에 사용)

     db_name: 데이터베이스 파일 이름
     table_name: 테이블 이름
     row_id: 불러올 데이터의 ID
    :return: 복원된 임베딩 벡터 (NumPy 배열)
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 데이터베이스에서 BLOB 데이터를 가져옴
    cursor.execute(f"SELECT vector_blob FROM {table_name} WHERE id=?", (row_id,))
    result = cursor.fetchone()

    conn.close()

    if result and result[0]:
        # BLOB 데이터를 NumPy 배열로 다시 변환 (float32 타입으로 복원)
        embedding = np.frombuffer(result[0], dtype=np.float32)
        return embedding
    return None

