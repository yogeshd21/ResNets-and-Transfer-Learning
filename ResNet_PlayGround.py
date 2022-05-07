import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#from keras.datasets import cifar100 #Replace use
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop

from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

def load_dataset():
  (trainX, trainY), (testX, testY) = cifar10.load_data()
  #(trainX, trainY), (testX, testY) = cifar100.load_data(label_mode="fine") #Replace use
  num_labels = len(np.unique(trainY))
  trainY = to_categorical(trainY)
  testY = to_categorical(testY)
  return trainX, trainY, testX, testY, num_labels

def prep_pixels(train, test):
  train_norm = train.astype('float32')
  test_norm = test.astype('float32')
  # normalize
  train_norm = train_norm / 255.0
  test_norm = test_norm / 255.0
  return train_norm, test_norm

def lr_schedule(epoch):
  lr = 1e-3
  if epoch > 180:
    lr *= 0.5e-3
  elif epoch > 160:
    lr *= 1e-3
  elif epoch > 120:
    lr *= 1e-2
  elif epoch > 80:
    lr *= 1e-1
  print('Learning rate: ', lr)
  return lr


def resnet_layer(inputs,
                 num_filters,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

  conv = Conv2D(num_filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-4))

  x = inputs
  if conv_first:
    x = conv(x)
    if batch_normalization:
      x = BatchNormalization()(x)
    if activation is not None:
      x = Activation(activation)(x)
  else:
    if batch_normalization:
      x = BatchNormalization()(x)
    if activation is not None:
      x = Activation(activation)(x)
    x = conv(x)
  return x


def resnet_v1(input_shape, n_val, num_classes, filter_size, no_resblk, optm):
  num_filters = filter_size

  inputs = Input(shape=input_shape)
  x = resnet_layer(inputs=inputs, num_filters=num_filters)

  for stack in range(no_resblk):
    for res_block in range(n_val):
      strides = 1
      if stack > 0 and res_block == 0:
        strides = 2
      y = resnet_layer(inputs=x,
                      num_filters=num_filters,
                      strides=strides)
      y = resnet_layer(inputs=y,
                      num_filters=num_filters,
                      activation=None)
      if stack > 0 and res_block == 0:
        x = resnet_layer(inputs=x,
                         num_filters=num_filters,
                         kernel_size=1,
                         strides=strides,
                         activation=None,
                         batch_normalization=False)
      x = keras.layers.add([x, y])
      x = Activation('relu')(x)
    num_filters *= 2

  if no_resblk == 3:
      x = AveragePooling2D(pool_size=8)(x)
  else:
      x = AveragePooling2D(pool_size=4)(x)
  y = Flatten()(x)
  outputs = Dense(num_classes,
                  activation='softmax',
                  kernel_initializer='he_normal')(y)

  model = Model(inputs=inputs, outputs=outputs)

  if optm == 'SGD':
    opt = SGD(learning_rate=lr_schedule(0), momentum=0.9)
  elif optm == 'ADAM':
    opt = Adam(learning_rate=lr_schedule(0))
  elif optm == 'RMSProp':
    opt = RMSprop(learning_rate=lr_schedule(0), momentum=0.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

  return model

def run_main():
  trainX, trainY, testX, testY, num_labels = load_dataset()
  trainX, testX = prep_pixels(trainX, testX)
  image_size = trainX.shape[1]
  df = pd.read_csv('./Data.csv')
  best = 0
  for i in range(len(df)):
      model = resnet_v1(input_shape=(image_size, image_size, 3), n_val=df['n value'][i], num_classes=num_labels, filter_size=df['Filter Size Start'][i], no_resblk = df['No. of Residual Blocks'][i], optm=df['Optimizer'][i])
      history = model.fit(trainX, trainY, epochs=100, batch_size=32, validation_data=(testX, testY), shuffle=True, verbose=0)

      _, acc = model.evaluate(testX, testY, verbose=0)
      df.loc[i, 'Accuracy'] = round(acc * 100.0, 3)
      print(i, '> %.3f' % (acc * 100.0))
      if acc > best:
        best = acc
        model.save('best_model_CIFAR10.h5')
        #model.save('./best_model_CIFAR100.h5') #Replace use for CIFAR-100 Dataset
  df.to_csv('Ans_CIFAR10.csv', index=False)
  #df.to_csv('Ans_CIFAR100.csv', index=False) #Replace use for CIFAR-100 Dataset

run_main()
