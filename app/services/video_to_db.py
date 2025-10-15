#pip install --upgrade opencv-python numpy torch torchvision transformers Pillow sqlite3
#라이브러리 설치 명령어
# c:/Users/강도형/venv/Scripts/Activate.ps1 
# 가상환경 켜기



import cv2
import os
import shutil # 임시 폴더 삭제를 위해 추가
# import time 제거
# CLIP 유틸리티 함수 가져오기
from app.utils.clip_utils import image_to_clip_embedding, save_embedding_to_db

# -----------------------------------------------------------
# 설정 변수
# -----------------------------------------------------------
# 실제 파일 경로와 일치하도록 수정하세요.
VIDEO_FILE = 'test.mp4' 
FRAME_FOLDER = 'test_frames'       
DB_NAME = os.path.join(os.path.dirname(__file__), "..", "..", "test.sqlite")
DB_NAME = os.path.abspath(DB_NAME)
TABLE_NAME = 'video_embeddings'    
TARGET_FPS = 1                     


def extract_frames_for_processing(video_path: str, output_folder: str, target_fps: int) -> int:
    """
    동영상에서 1fps 간격으로 프레임을 추출하여 폴더에 저장합니다.
    각 비디오는 별도의 하위 폴더에 저장됩니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # 오류 발생 시 사용자에게 간결하게 알리고 종료
        print(f"오류: 동영상 파일 '{video_path}'을(를) 열 수 없습니다. 경로와 파일명을 확인하세요.")
        return 0

    # 항상 test 폴더에 저장
    video_folder = os.path.join(output_folder, "test")
    
    # 기존 폴더가 있으면 삭제하고 새로 생성
    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    os.makedirs(video_folder)

    original_fps = cap.get(cv2.CAP_PROP_FPS)#오리지널 영상 초당 프레임 확인
    frame_save_interval = round(original_fps / target_fps) # 저장 간격 결정
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) #파일 생성
        
    current_frame_index = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame_index % frame_save_interval == 0:#저장 지점인지 결정
            filename = os.path.join(video_folder, f"frame_{saved_frame_count:05d}.jpg")#이미지 파일 이름 결정 총 5자리 ex) saved_frame_count=0이면 frame_00000.jpg
            cv2.imwrite(filename, frame)
            saved_frame_count += 1

        current_frame_index += 1
        
    cap.release()
    return saved_frame_count


def process_video_to_db(video_file: str, frame_folder: str, db_name: str, table_name: str, total_frames: int):
    """
    추출된 프레임을 CLIP 임베딩으로 변환하고 DB에 저장합니다.
    """
    
    # 비디오 이름을 가져와서 해당 비디오의 프레임 폴더 경로 생성
    video_name_only = os.path.splitext(os.path.basename(video_file))[0]
    video_frame_folder = os.path.join(frame_folder, "test")  # test 폴더 안에 프레임이 저장됨
    
    frame_files = sorted([f for f in os.listdir(video_frame_folder) if f.endswith(('.jpg', '.png'))])
    if not frame_files:
        print(f"경고: {video_frame_folder}에서 프레임을 찾을 수 없습니다.")
        return

    for i, frame_name in enumerate(frame_files):
        full_path = os.path.join(video_frame_folder, frame_name)  # test 폴더 안의 이미지 경로
        
        try:
            # 파일명에서 인덱스 추출 (frame_00000.jpg 형식)
            frame_index = int(frame_name.split('_')[1].split('.')[0])
        except:
            frame_index = i

        # 임베딩 생성 및 DB 저장
        embedding = image_to_clip_embedding(full_path)
        
        if embedding is not None:
            save_embedding_to_db(db_name, table_name, video_name_only, frame_index, embedding)


# -----------------------------------------------------------
# 메인 실행
# -----------------------------------------------------------
if __name__ == '__main__':
    print("--- 동영상 벡터 임베딩 처리 시작 ---")
    
    # 1. 프레임 추출
    extracted_count = extract_frames_for_processing(VIDEO_FILE, FRAME_FOLDER, TARGET_FPS)
    
    if extracted_count > 0:
        print(f"총 {extracted_count}개의 프레임 추출 완료. 이제 벡터 변환을 시작합니다.")
        
        # 2. 임베딩 변환 및 DB 저장
        process_video_to_db(VIDEO_FILE, FRAME_FOLDER, DB_NAME, TABLE_NAME, extracted_count)
        
        print(f" 모든 작업 완료. 결과는 '{DB_NAME}'에 저장되었습니다.")
        
        # 3. 임시 폴더 삭제 (선택 사항 - 주석 해제 시)
        # try:
        #     shutil.rmtree(FRAME_FOLDER)
        # except OSError as e:
        #     pass
    
    print("--- 스크립트 종료 ---")