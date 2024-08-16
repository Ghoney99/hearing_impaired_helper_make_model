# 수어 인식 및 번역 모델

## 📋 프로젝트 개요

본 프로젝트는 청각장애 학생들의 교육 환경 개선을 위한 수어 인식 및 번역 모델 개발을 목표로 합니다.

## 🎯 개발 배경

- 국립국어원 2020년 보고서: 청각장애 학생 50.4%가 '수어 없는 수업 내용 이해 어려움' 호소
- 2023년 특수교육통계: 농학생 80.3%가 일반학교 재학 중
- 수어 교육 부재와 수업 참여의 어려움 해소 필요성 대두

## 📊 데이터셋

- **출처**: AI Hub - 수어영상
- **구성**:
  - 총 536,000 수어영상 클립 (.mp4)
  - 수어문장 2000개, 수어단어 3000개
  - 지숫자/지문자 1000개
- **사용**: 지문자 영상에 대해서만 학습 진행
  (저작권 문제로 실제 datasets에는 다른 파일 포함)

## 🛠 기술 스택

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0F9D58?style=for-the-badge&logo=Google&logoColor=white)

## 💻 주요 기능 및 구현

### 1. 데이터 전처리 (`create_dataset_from_video.py`)

- **데이터 수집 및 초기 처리**
  - OpenCV 활용 비디오 데이터 읽기 (cv2.VideoCapture)
  - 30fps 비디오에서 초당 30개 이미지 추출

- **키포인트 추출**
  - MediaPipe HolisticDetector로 프레임별 21개 키포인트 추출

- **키포인트 데이터 가공**
  - 키포인트를 벡터와 각도 정보로 변환
  - 관절 움직임 표현을 위한 벡터 각도 계산

- **데이터 정규화 및 형식화**
  - 벡터 값 정규화 및 각도 정보 계산 (arccosine 함수 사용)
  - 56차원 데이터 생성: (20,2) 벡터 값 + 15개 각도 값 + 라벨

- **시퀀스 데이터 생성**
  - 비디오 데이터를 시퀀스 형태로 변환 (시퀀스 길이: 10)
  - 최종 데이터 형태: (488, 10, 56) (데이터 개수, 시퀀스 길이, 특성 값)

### 2. 모델 학습 (`train_hand_gesture.ipynb`)

- npy 파일 로드 후 모델 생성
- keras model 및 tflite model 생성 (tflite model 활용)

**모델 구성**:
- L2 Norm regularization (과적합 방지)
- ReLU 활성화 함수
- Dropout(0.3) (과적합 방지)
- Categorical_CrossEntropy 손실 함수
- Adam 옵티마이저
- ReduceLROnPlateau (학습률 자동 조정)
- 21 Epoch (Early stopping 적용)

### 3. 실시간 테스트 (`webcam_test_model_tflite.py`)

- 웹캠을 활용한 실시간 수어 인식 테스트

## 🚀 시작하기

```bash
# 저장소 클론
git clone https://github.com/your-username/sign-language-recognition.git

# 필요 라이브러리 설치
pip install -r requirements.txt

# 데이터 전처리
python create_dataset_from_video.py

# 모델 학습
jupyter notebook train_hand_gesture.ipynb

# 실시간 테스트
python webcam_test_model_tflite.py
```

