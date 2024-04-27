import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import time

#loading the model
model = tf.keras.models.load_model("model1.h5")

video = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

#categorizing the labels as alphabets
data = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',26: ' '}

while True:
    x = []
    y = []

    cap, frame = video.read()
    H, W, _ = frame.shape

    frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)


    time.sleep(0.1)

    result = hands.process(frame1)
    data_xy1 = [];
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            data_xy = [];
            for j in range(len(hand_landmarks.landmark)):
                # print(j)qqqqq
                data_xy.append(hand_landmarks.landmark[j].x)
                data_xy.append(hand_landmarks.landmark[j].y)
                x.append(hand_landmarks.landmark[j].x)
                y.append(hand_landmarks.landmark[j].y)
        data_xy = np.reshape(data_xy, (-1, 42))

        x1 = int(min(x)*W)-10
        y1 = int(min(y)*H)-10

        x2 = int(max(x)*W)-10
        y2 = int(max(y)*H)-10

        prediction = model.predict(np.asarray(data_xy))
        # res = res + data[np.argmax(prediction[0])]
        print(data[np.argmax(prediction[0])])
        res = data[np.argmax(prediction[0])]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
        cv2.putText(frame,res,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,0),4)
    cv2.imshow("video",frame)
    cv2.waitKey(1)
print("outside the while loop")
# cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()
