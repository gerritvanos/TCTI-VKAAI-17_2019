import pickle , gzip , os
import numpy as np
from urllib import request
from pylab import imshow , show , cm
import keras 
import tensorflow

url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
if not os.path.isfile("mnist.pkl.gz"):
    request.urlretrieve(url, "mnist.pkl.gz")

f = gzip.open('mnist.pkl.gz', 'rb')
train_set , valid_set , test_set = pickle.load(f, encoding='latin1')
f.close()

def get_image(number):
    (X, y) = [img[number] for img in train_set]
    return (np.array(X), y)

def view_image(number):
    (X, y) = get_image(number)
    print("Label: %s" % y)

def get_outputs(input_set):
    outputs =[]
    for label in range(len(input_set[1])):
        number = input_set[1][label]
        if number == 0:
            outputs.append([1,0,0,0,0,0,0,0,0,0])
        elif number == 1:
            outputs.append([0,1,0,0,0,0,0,0,0,0])
        elif number == 2:
            outputs.append([0,0,1,0,0,0,0,0,0,0])
        elif number == 3:
            outputs.append([0,0,0,1,0,0,0,0,0,0])
        elif number == 4:
            outputs.append([0,0,0,0,1,0,0,0,0,0])
        elif number == 5:
            outputs.append([0,0,0,0,0,1,0,0,0,0])
        elif number == 6:
            outputs.append([0,0,0,0,0,0,1,0,0,0])
        elif number == 7:
            outputs.append([0,0,0,0,0,0,0,1,0,0])
        elif number == 8:
            outputs.append([0,0,0,0,0,0,0,0,1,0])
        elif number == 9:
            outputs.append([0,0,0,0,0,0,0,0,0,1])
    return outputs

train_outputs = get_outputs(train_set)
test_outputs = get_outputs(test_set)
validation_outputs = get_outputs(valid_set)

model = keras.Sequential()
model.add(keras.layers.Dense(units=784, activation='sigmoid'))
model.add(keras.layers.Dense(units= 16, activation='sigmoid'))
model.add(keras.layers.Dense(units= 16, activation='sigmoid'))
model.add(keras.layers.Dense(units= 10, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(train_set[0],np.array(train_outputs), validation_data=(valid_set[0],np.array(validation_outputs)) ,epochs=100, batch_size=32, verbose=1)

score = model.evaluate(test_set[0], np.array(test_outputs))

print("accuracy:",round(score[1]*100,2), "%")

#my neural network constists of the following layers
# 784 -- 16 -- 16 -- 10
# the 784 is based on 28*28 pixels inputs
# the 10 output neurons are based on the fact that we need to recognize 10 different characters
# the 2 hidden layers with both 16 neurons is roughly based on the following video https://www.youtube.com/watch?v=aircAruvnKk& 
# this video explains the theory behind the OCR
# this is based on the diffrent lines that are used by all the numbers. for example a 3 and a 5 use most of the same lines but are different.
# based on this theory two layers of 16 should be enough to distingush all numbers.
