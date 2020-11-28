from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

## 모델 로드하기
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
model = load_model('models/mask_detector.model')

"""
## 이미지 로드하기
img = cv2.imread('imgs/03.jpg')
h, w = img.shape[:2]
plt.figure(figsize=(16, 10))
"""

## 실시간으로 비디오 캡쳐하기
cap = cv2.VideoCapture(0)
# 화면 크기 조절
cap.set(3, 800)
cap.set(4, 450)

faces = []

while True:
    ret, frame = cap.read()
    # 좌우반전
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    if ret:
        ## 얼굴을 네모로 짤라주는 과정.
        # cv2.dnn,blobFromImage() : dnn 모듈이 사용하는 형태로 이미지를 변형 axis 순서만 바뀐다.
        # cv2.imshow('video', frame)
        blob = cv2.dnn.blobFromImage(frame, scalefactor = 1., size = (300, 300), mean = (104., 177., 123.))
        facenet.setInput(blob)
        dets = facenet.forward()

        ## Detect Faces
        for i in range(dets.shape[2]):
            confidence = dets[0, 0, i, 2]
            if confidence < 0.5:
                continue

            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)

            face = frame[y1 : y2, x1 : x2]
            faces.append(face)

        # 전처리 하는 과정
        for i, face in enumerate(faces):
            face_input = cv2.resize(face, dsize = (224, 224))
            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
            face_input = preprocess_input(face_input)
            face_input = np.expand_dims(face_input, axis = 0)

            mask, nomask = model.predict(face_input).squeeze()

        str = '%.2f %%' % (mask * 100)
        cv2.putText(frame, str, (20, 40), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 255, 0))
        cv2.imshow('', cv2.resize(frame, (800, 600)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destoryAllWindows()

"""
# 뒤에 -1을 붙여주면 OpenCV는 BGR로 읽히지만 RGB로 바꿔준다.
plt.imshow(img[:, :, ::-1])

# 얼굴 네모로 자르기
for i, face in enumerate(faces):
    plt.subplot(1, len(faces), i + 1)
    plt.imshow(face[:, :, ::-1])
    #plt.show()

# 마스크 착용 여부 예측 모델
plt.figure(figsize = (16, 5))

#전처리 하는 과정
for i, face in enumerate(faces):
    face_input = cv2.resize(face, dsize = (224, 224))
    face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
    face_input = preprocess_input(face_input)
    face_input = np.expand_dims(face_input, axis = 0)

    mask, nomask = model.predict(face_input).squeeze()

    plt.subplot(1, len(faces), i + 1)
    plt.imshow(face[:, :, ::-1])
    plt.title('%.2f%%' % (mask * 100)) 
    
    plt.show()
"""