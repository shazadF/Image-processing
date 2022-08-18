from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from matplotlib.cm import binary
from random import randint, seed
import numpy as np
from keras.callbacks import Callback
import h5py
import os
from os import path
import re

batch_size = 128
nb_classes = 10
nb_epoch =2


# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#finding random patch from X_train
data_set=[]
for i in range(0, 32):

    	x=randint(0,49999)
    	dirran=randint(0, 23)
    	img_tem=X_train[x]
    	img=img_tem[:,dirran:dirran+3,dirran:dirran+3]
    	# print(img.shape)
    	data_set.append(img)

data_set=np.asarray(data_set)
new_rshape = data_set.reshape(32,1,3,3)
new_rshape = np.asarray(new_rshape)
print(len(new_rshape))

weight = new_rshape
bias = np.zeros(32)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols),
			weights=[weight,bias]))
model.summary()
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


# class prediction_history(Callback):
#     def __init__(self):
#         self.predhis = []
#     def on_epoch_end(self, epoch, logs={}):
#         self.predhis.append(model.predict(predictor_train))

#Calling the subclass
# predictions=prediction_history()

# history = LossHistory()

for i in range (10):
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                verbose=1, validation_data=(X_test, Y_test))
    # else:
    #     print("Error")



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# plotting/visualizing layers weight
def plot_layer(layer, x, y):
    layer_config = layer.get_config()
    print("Layer Name: ", layer_config['name'])
    layer_weight, layer_bias = layer.get_weights()
    figure = plt.figure()
    outpath = "/home/forhad/plot/"
    # layer_weight = np.zeros(layer_weight)
    layer_weight = layer_weight.reshape((32,9))
    # figure.savefig(path.join(outpath,"figure_0.png"))
    for i in range(len(layer_weight)):
    #     ax = figure.add_subplot(y, x, i+1)
    #     ax.matshow(layer_weight[i][0], cmap=binary)
    #     plt.xticks(np.array([]))
    #     plt.yticks(np.array([]))
    # figure.set_tight_layout(True)
    # plt.show()
        x = np.arange(0,9)
        y = layer_weight[i]
        plt.plot(x,y)
        plt.ylabel("Image points")
        plt.draw()
        figure.savefig(outpath,"figure_{0}.png".format(i))
        # fileNameTemplate = r'/home/ulab/plot{0:02d}.png'
        # rootdir='/home/ulab/plot for each search id'
        # for subdir,dirs,files in os.walk():
        #     for count, file in enumerate(files):
        #         # Generate a plot in `pl`
        #         plt.savefig(fileNameTemplate.format(count), format='png')
        #         plt.clf()

        # plt.axis(nb_epoch, 27)
        # plt.show()
    # figure.savefig('test.png')



#Executing the model.fit of the neural network
# model.fit(X=predictor_train, y=target_train, nb_epoch=2, batch_size=batch,validation_split=0.1,callbacks=[predictions])

#Printing the prediction history
# print predictions.predhis
#
# class LossHistory(Callback):
#     def on_epoch_begin(self, epoch, logs={}):
#         self.epoch_counter = []
#     def on_epoch_end(self, epoch, logs={}):
#         self.epoch_counter.append(logs)


# visualizing weights for first layer
plot_layer(model.layers[0], 8, 4)
