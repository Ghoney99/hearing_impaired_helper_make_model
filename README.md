
1. create_dataset_from_video.py
- 수어 데이터를 활용하여 양손 관절 및 각도를 시퀀스 데이터로 변환하여 npy 파일로 저장합니다.

2. train_hand_gesture.ipynb
- npy file load하여 모델을 생성합니다.
- keras model, tflite model 두 종류의 모델을 생성합니다. (tflite model만 활용)

3. webcam_test_model : 웹캠을 활용하여 테스트합니다.
- tflite.py : tflite model을 테스트합니다.
