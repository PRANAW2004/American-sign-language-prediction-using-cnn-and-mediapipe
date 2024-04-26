import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp

#loading the model
model = tf.keras.models.load_model("model1.h5")

video = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

#categorizing the labels as alphabets
data = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',26: ' '}


while True:
    cap,frame = video.read()
    cv2.imshow("video",frame)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            data_xy = [];
            for j in range(len(hand_landmarks.landmark)):
                data_xy.append(hand_landmarks.landmark[j].x)
                data_xy.append(hand_landmarks.landmark[j].y)

            data_xy = np.reshape(data_xy,(-1,42))
            #predicting the output
            prediction = model.predict(np.asarray(data_xy))
            print(data[np.argmax(prediction[0])])
    if(cv2.waitKey(25) == ord('q')):
        break
video.release()
cv2.destroyAllWindows()
