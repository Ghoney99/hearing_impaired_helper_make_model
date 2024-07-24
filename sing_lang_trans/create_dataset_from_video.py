import cv2
import sys
import os
import mediapipe as mp
import numpy as np
import modules.holistic_module as hm
from modules.utils import createDirectory, Vector_Normalization
import time

# 출력 비디오를 저장할 디렉토리 생성
createDirectory('dataset/output_video')

# 설정
seq_length = 10  # 시퀀스 길이
actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']  # 인식할 동작 목록

# 데이터셋 초기화
dataset = {i: [] for i in range(len(actions))}

# MediaPipe 홀리스틱 모델 초기화
detector = hm.HolisticDetector(min_detection_confidence=0.3)

# 비디오 파일 목록 생성
videoFolderPath = "dataset/output_video"
testTargetList = []

def get_video_files(folder_path):
    """
    지정된 폴더에서 모든 비디오 파일의 경로를 재귀적으로 찾아 리스트로 반환
    """
    for videoPath in os.listdir(folder_path):
        actionVideoPath = os.path.join(folder_path, videoPath)
        for actionVideo in os.listdir(actionVideoPath):
            fullVideoPath = os.path.join(actionVideoPath, actionVideo)
            testTargetList.append(fullVideoPath)

get_video_files(videoFolderPath)

# 비디오 파일 목록을 정렬 (파일명의 두 번째 부분을 기준으로 역순 정렬)
testTargetList = sorted(testTargetList, key=lambda x: x.split("/")[-2], reverse=True)
print("Video List:", testTargetList)

def process_video(video_path):
    """
    주어진 비디오 파일을 처리하여 핸드 랜드마크 데이터를 추출
    """
    data = []
    idx = actions.index(video_path.split("/")[-2])
    print("Now Streaming:", video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Camera open failed!")
        sys.exit()

    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = round(1000/fps) if fps != 0 else round(1000/30)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = detector.findHolistic(img, draw=True)
        _, right_hand_lmList = detector.findRighthandLandmark(img)

        if right_hand_lmList is not None:
            joint = np.zeros((42, 2))
            for j, lm in enumerate(right_hand_lmList.landmark):
                joint[j] = [lm.x, lm.y]

            # 벡터 정규화 및 각도 계산
            vector, angle_label = Vector_Normalization(joint)
            angle_label = np.append(angle_label, idx)
            d = np.concatenate([vector.flatten(), angle_label.flatten()])
            data.append(d)

        cv2.waitKey(delay)
        if cv2.waitKey(delay) == 27:  # ESC 키를 누르면 종료
            break

    print("Finish Video Streaming")
    return np.array(data), idx

# 각 비디오 처리
for target in testTargetList:
    data, idx = process_video(target)
    
    # 시퀀스 데이터 생성
    for seq in range(len(data) - seq_length):
        dataset[idx].append(data[seq:seq + seq_length])

# 데이터 저장 (현재 주석 처리됨)
'''
def save_dataset():
    """
    추출된 데이터셋을 파일로 저장
    """
    created_time = int(time.time())
    for i in range(len(actions)):
        save_data = np.array(dataset[i])
        np.save(os.path.join('dataset', f'seq_{actions[i]}_{created_time}'), save_data)
    print("Finish Save Dataset")

save_dataset()
'''