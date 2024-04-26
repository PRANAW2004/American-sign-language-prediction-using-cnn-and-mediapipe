import os
import pickle
import numpy as np
import cv2
import mediapipe as mp

#data directory that has the alphabetical signatures comment this line after the data is stored in the pickle file
dir = ('./data')
#uncomment the below line after commenting the above line
#dir = './space_data'

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
#
hands = mp_hands.Hands()

data = []
label = []

#comment the above line of code if dir = './data' line is commented and uncomemnt the below line of code

# data1 = pickle.load(open("data.pickle",'rb'))
# data = np.asarray(data1['data'])
# label = np.asarray(data1['label'])

print(data,label)

for i in os.listdir(dir):
    for j in os.listdir(os.path.join(dir,i)):
        print(j)
        # print(i)
        a = os.path.join(dir,i,j);
        print(a)
        image = cv2.imread(a)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        result = hands.process(image)

        for k in result.multi_hand_landmarks:
            # print(k)
            data_xy = []
            for m in range(len(k.landmark)):
                data_xy.append(k.landmark[m].x)
                data_xy.append(k.landmark[m].y)
        data.append(data_xy)
        label.append(int(i))
print(len(data),len(label))


data2 = open("data1.pickle",'wb')
pickle.dump({'data':data,'label': label},data2)
data2.close()
