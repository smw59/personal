import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
print(len(X_train))
print(X_train[0].shape)
print(X_train.shape)

plt.matshow(X_train[0])
print(y_train[:5])

X_train = X_train / 255
X_test = X_test / 255

# scaling improves accuracy

X_train_flat = X_train.reshape(len(X_train), 28*28)
X_test_flate = X_test.reshape(len(X_train), 28*28)

model = keras.Sequential([
    
    keras.layers.Dense(100, input_shape=(784,),activation = 'relu'),
  # 1 hidden layer
  # adds computation time
    keras.layers.Dense(10,activation = 'sigmoid')
  # 10 = output neurons, 784 = input neurons
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              # use this loss for one hot encoder
              # can also use mean squared error
              metrics = ['accuracy'])
model.fit(X_train_flat, y_train, epochs=5)

model.evaulate(X_test_flat, y_test)
plt.matshow()
y_predict = model.predict(X_test_flat)
np.argmax(y_predict[1])
y_predict_labels = [np.argmax(i) for i in y_predict]
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predict_labels)

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
