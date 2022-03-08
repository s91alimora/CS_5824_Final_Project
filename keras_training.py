import pandas as pd
from keras.models import Sequential
from keras.layers import *
import keras.metrics
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
import keras
import glob
import cv2
import time
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.vis_utils import plot_model
import keras_evaluation

start = time.time()

#data preprocessing
training_dataset = pd.DataFrame()
training_files = glob.glob("Training_Data/*.csv")
labels = []
for file in training_files:
    data = pd.read_csv(file)
    filename = file.split(".")[0].split("\\")[1]
    data['label'] = filename        #len(labels)/len(training_files)
    labels.append(filename)
    training_dataset = pd.concat([training_dataset,data],)

training_dataset = training_dataset.sample(frac=1)                        #shuffle the dataset. frac=1 means random sampling will return 100% of the data
print(training_dataset.head())
X = training_dataset.drop('label',axis=1).values
Y = pd.get_dummies(training_dataset[['label']]).values                      #convert labels to categorical
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, train_size=0.8)

#training
model = Sequential()
model.add(Dense(30,input_dim=63))                 #hidden layer
model.add(Dense(len(labels),activation='softmax'))                   #output layer
model.compile(loss='categorical_crossentropy',metrics=['CategoricalAccuracy','Precision','Recall','FalsePositives','FalseNegatives'])

logger = keras.callbacks.TensorBoard(
    log_dir = "logs",
    write_graph = True,
    histogram_freq = 5
)

model.fit(
    X_train,
    Y_train,
    epochs=40,
    verbose=2,
    callbacks=[logger],
    validation_data=(X_test,Y_test)
)
model.save("trained_model.h5")
pd.DataFrame(labels).to_csv("labels.csv",index=False,header=False)
training_dataset.to_csv("compiled_dataset.csv",index=False,header=True)

# for layer in model.layers:
#     weights = layer.get_weights()
#     print(weights)
print(model.summary())

# model = keras.models.load_model('trained_model.h5')
keras_evaluation.show_confusion_matrix(model,X_test,Y_test)

print(f"training time={time.time()-start}")