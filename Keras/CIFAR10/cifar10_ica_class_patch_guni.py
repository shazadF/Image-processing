'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from matplotlib.cm import binary
from keras.utils import np_utils
from random import randint, seed
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from numpy import linalg
from sklearn.decomposition import PCA, FastICA
from scipy.misc import toimage

batch_size = 27
nb_classes = 10
nb_epoch = 20
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
classes = np.unique(y_train)
# y_train = y_train[0:1000]
# for i in range(10):
#     labels = y_train[(i-1)*1000:i*1000]
#     print("--------", labels)
#     print(i,"--------")


new_patch = X_train[0:1000]
# print("new_patch : ", new_patch.shape[0])
# for i in range(len(new_patch[0:10])):
# 	plt.subplot(330 + 1 + i)
# 	plt.imshow(toimage(new_patch[i]))
# # show the plot
# plt.show()
#shazad // start
#finding random patch from X_train
data_set=[]
resize = []
a = []
b = 0
lvl_tem=new_patch
img=[]

newimg = []
for i in range(0,1000):
    #print(lvl_tem[i])
    b=0
    x= lvl_tem[i]
    # print(x[0])
    for j in range (0,10):
            lvl_tem1 = x[:, b:b+3, b:b+3]
            #x+=1
            b = b+3
            img.append(lvl_tem1)
            # print(len(img),"Image khoma")
new_img = np.asarray(img)
#new_img = np.asarray(img)
print(new_img.shape)


x = []
b = []
inside = []
k = []
new_array = []
for i in range (len(new_img)):
    b = new_img[i]
    b = np.ndarray.flatten(b, 'C')

    b = b.reshape((1,27))
    print (i,"--------", b.shape)
    for j in range (len(b)):
        inside = b[j]
        # print("&&", inside.shape)
        k.append(inside)
    # new_array.append(b)
new_array = np.asarray(k)
print(new_array.shape)

A = new_array  # Mixing matrix
# X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=27)
S_ = ica.fit_transform(A)  # Reconstruct signals
# print("S_ shape: ", S_.shape)
A_ = ica.mixing_  # Get estimated mixing matrix
# print(A_.shape)

x = []
for i in range (len(A_)):
    ver_vec = A_[i]
    x.append(ver_vec)

x = np.asarray(x)
new_xshape = x.reshape(27,3,3,3)
# print (new_xshape)
# print("------------", new_xshape.shape)

norm = []
for i in range (len(new_xshape)):
    ver_vec = new_xshape[i]
    ver_vec = ver_vec/np.linalg.norm(new_xshape[i])
    # print(len(ver_vec))
    norm.append(ver_vec)

norm = np.asarray(norm)
# print(norm)
norm = np.ndarray.flatten(norm, 'C')
new_xshape = norm.reshape(27,3,3,3)
# print (new_xshape)
print("------------", new_xshape.shape)


weight = new_xshape
bias = np.zeros(27)
# the CIFAR10 images are RGB
img_channels = 3


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(27, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols),
			weights=[weight,bias], init = 'glorot_uniform'))
model.add(Activation('relu'))
# model.add(Convolution2D(32, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Convolution2D(64, 3, 3, border_mode='same'))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        verbose = 2,
                        validation_data=(X_test, Y_test))
