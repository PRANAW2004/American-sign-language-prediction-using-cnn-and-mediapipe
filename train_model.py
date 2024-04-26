import tensorflow as tf
from tensorflow.keras import models,layers
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

#load the data from the pickle file
data1 = pickle.load(open("data1.pickle",'rb'))
data = np.asarray(data1['data'])
labels = np.asarray(data1['label'])

#split the data into training and testing data
x_train,x_test,y_train,y_test = train_test_split(data,labels,shuffle=True,test_size=0.25)

#creating the convolutional neural network architecture
model = models.Sequential()

model.add(layers.Reshape((42,1), input_shape=(42,)))
model.add(layers.Conv1D(42, 3, activation='relu',padding='same'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(84, 3, activation='relu',padding='same'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(168, 3, activation='relu',padding='same'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(336, 3, activation='relu',padding='same'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(672, 3, activation='relu',padding='same'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(27))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#giving the data as the input to the neural network
print("model training has started")
model.fit(x_train, y_train, epochs=50,validation_data=(x_test, y_test))
print("model training has finished")

#saving the model
model.save("model1.h5")