프로젝트명
수어 인식 및 번역 모델

설명

개발 배경
청각장애 학생들은 일반 교육 환경에서 여러 어려움에 직면합니다. 국립국어원의 2020년 보고서에 따르면, 전체 청각장애 학생 중 50.4%가 '수어 없는 수업의 내용을 이해하지 못한다'고 응답했습니다. 또한, 2023년 특수교육통계에 따르면 농학생 중 80.3%가 일반학교에 재학 중입니다. 이는 수어 교육의 부재와 함께 수업 참여의 어려움을 야기합니다.

기술 스택
TensorFlow
Keras
MediaPipe
OpenCV

사용 방법
1. create_dataset_from_video.py
- 수어 데이터를 활용하여 양손 관절 및 각도를 시퀀스 데이터로 변환하여 npy 파일로 저장합니다.

2. train_hand_gesture.ipynb
- npy file load하여 모델을 생성합니다.
- keras model, tflite model 두 종류의 모델을 생성합니다. (tflite model만 활용)

3. webcam_test_model : 웹캠을 활용하여 테스트합니다.
- tflite.py : tflite model을 테스트합니다.
