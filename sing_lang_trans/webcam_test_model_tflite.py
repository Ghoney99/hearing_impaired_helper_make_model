import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from modules.utils import Vector_Normalization
from PIL import ImageFont, ImageDraw, Image

# 설정
actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
seq_length = 10
fontpath = "fonts/HMKMMAG.TTF"
font = ImageFont.truetype(fontpath, 40)

def initialize_detector_and_model():
    """MediaPipe 홀리스틱 모델과 TFLite 모델 초기화"""
    detector = hm.HolisticDetector(min_detection_confidence=0.3)
    interpreter = tf.lite.Interpreter(model_path="models/multi_hand_gesture_classifier.tflite")
    interpreter.allocate_tensors()
    return detector, interpreter

def process_hand_landmarks(right_hand_lmList):
    """손 랜드마크 처리 및 벡터 정규화"""
    joint = np.zeros((42, 2))
    for j, lm in enumerate(right_hand_lmList.landmark):
        joint[j] = [lm.x, lm.y]
    vector, angle_label = Vector_Normalization(joint)
    return np.concatenate([vector.flatten(), angle_label.flatten()])

def predict_action(interpreter, input_data):
    """TFLite 모델을 사용하여 동작 예측"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    return y_pred[0]

def draw_text_on_image(img, text):
    """이미지에 한글 텍스트 그리기"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 30), text, font=font, fill=(255, 255, 255))
    return np.array(img_pil)

def main():
    detector, interpreter = initialize_detector_and_model()
    cap = cv2.VideoCapture(0)
    
    seq = []
    action_seq = []
    last_action = None

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = detector.findHolistic(img, draw=True)
        _, right_hand_lmList = detector.findRighthandLandmark(img)

        if right_hand_lmList is not None:
            d = process_hand_landmarks(right_hand_lmList)
            seq.append(d)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = predict_action(interpreter, input_data)
            
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.9:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
                if last_action != this_action:
                    last_action = this_action

            img = draw_text_on_image(img, f'{action.upper()}')

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()