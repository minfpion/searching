# app/utils/faiss_utils.py
import faiss
import numpy as np
import sqlite3

def load_vectors_from_db(db_path: str, table_name: str):
    """SQLite DB에서 모든 벡터와 관련 메타데이터를 불러옵니다."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"SELECT id, video_name, frame_index, vector_blob FROM {table_name}")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return np.array([]), [], [], []

    vectors = [np.frombuffer(row[3], dtype=np.float32) for row in rows]
    ids = [row[0] for row in rows]
    video_names = [row[1] for row in rows]
    frame_indices = [row[2] for row in rows]

    return np.array(vectors).astype('float32'), ids, video_names, frame_indices


def build_faiss_index(vectors: np.ndarray):
    """NumPy 벡터 배열로부터 FAISS 인덱스를 구축합니다."""
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index


def search_similar(index, query_vector: np.ndarray, top_k: int = 5):
    """FAISS 인덱스에서 유사한 벡터를 검색합니다."""
    query_vector = np.array([query_vector]).astype('float32')
    distances, indices = index.search(query_vector, top_k)
    return distances[0], indices[0]