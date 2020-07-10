import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import keras
from keras.layers import *
from keras.models import Model
from keras.callbacks import LearningRateScheduler

import data_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Parameter for training.")

    parser.add_argument(
        "--data_dir", type=str, help="path to .npy file of input data", default="data"
    )
    parser.add_argument(
        "--id_dir", type=str, help="path to id_list", default="data/id_list"
    )
    parser.add_argument(
        "--output_dir", type=str, help="path to model output", default="data/output"
    )
    parser.add_argument(
        "--n_epochs", type=int, help="number of all epochs for training", default=80000
    )
    parser.add_argument(
        "--lr", type=float, help="initial learning rate", default=0.005
    )

    return parser.parse_args()


def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)


def model_generator(n_stack=5, input_size=128, epsilon=0.001):
    # _ shows output after activation layer

    inputs = Input(shape=(input_size, input_size, 3*n_stack)) # (height, width, channel*stack)
    f0_1 = Conv2D(15, kernel_size=5, padding='same')(inputs)
    f0_1 = BatchNormalization(epsilon=epsilon)(f0_1)
    f0_1_ = Activation('relu')(f0_1) # ReLU(f0)
    f0_2 = Conv2D(64, kernel_size=3, padding='same')(f0_1_) # Add this layer due to tensor shape
    f0_2 = BatchNormalization(epsilon=epsilon)(f0_2)
    f0_2_ = Activation('relu')(f0_2)

    d1 = Conv2D(64, kernel_size=3, strides=2, padding='same')(f0_2_)
    d1 = BatchNormalization(epsilon=epsilon)(d1)
    d1_ = Activation('relu')(d1)
    f1_1 = Conv2D(128, kernel_size=3, padding='same')(d1_)
    f1_1 = BatchNormalization(epsilon=epsilon)(f1_1)
    f1_1_ = Activation('relu')(f1_1)
    f1_2 = Conv2D(128, kernel_size=3, padding='same')(f1_1_)
    f1_2 = BatchNormalization(epsilon=epsilon)(f1_2)
    f1_2_ = Activation('relu')(f1_2)

    d2 = Conv2D(256, kernel_size=3, strides=2, padding='same')(f1_2_)
    d2 = BatchNormalization(epsilon=epsilon)(d2)
    d2_ = Activation('relu')(d2)
    f2_1 = Conv2D(256, kernel_size=3, padding='same')(d2_)
    f2_1 = BatchNormalization(epsilon=epsilon)(f2_1)
    f2_1_ = Activation('relu')(f2_1)
    f2_2 = Conv2D(256, kernel_size=3, padding='same')(f2_1_)
    f2_2 = BatchNormalization(epsilon=epsilon)(f2_2)
    f2_2_ = Activation('relu')(f2_2)
    f2_3 = Conv2D(256, kernel_size=3, padding='same')(f2_2_)
    f2_3 = BatchNormalization(epsilon=epsilon)(f2_3)
    f2_3_ = Activation('relu')(f2_3)

    d3 = Conv2D(512, kernel_size=3, strides=2, padding='same')(f2_3_)
    d3 = BatchNormalization(epsilon=epsilon)(d3)
    d3_ = Activation('relu')(d3)
    f3_1 = Conv2D(512, kernel_size=3, padding='same')(d3_)
    f3_1 = BatchNormalization(epsilon=epsilon)(f3_1)
    f3_1_ = Activation('relu')(f3_1)
    f3_2 = Conv2D(512, kernel_size=3, padding='same')(f3_1_)
    f3_2 = BatchNormalization(epsilon=epsilon)(f3_2)
    f3_2_ = Activation('relu')(f3_2)
    f3_3 = Conv2D(512, kernel_size=3, padding='same')(f3_2_)
    f3_3 = BatchNormalization(epsilon=epsilon)(f3_3)
    f3_3_ = Activation('relu')(f3_3)

    u1 = Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(f3_3_)
    u1 = BatchNormalization(epsilon=epsilon)(u1)
    s1 = Average()([u1, f2_3]) # Original paper might have intended Sum layer
    s1_ = Activation('relu')(s1)
    f4_1 = Conv2D(256, kernel_size=3, padding='same')(s1_)
    f4_1 = BatchNormalization(epsilon=epsilon)(f4_1)
    f4_1_ = Activation('relu')(f4_1)
    f4_2 = Conv2D(256, kernel_size=3, padding='same')(f4_1_)
    f4_2 = BatchNormalization(epsilon=epsilon)(f4_2)
    f4_2_ = Activation('relu')(f4_2)
    f4_3 = Conv2D(256, kernel_size=3, padding='same')(f4_2_)
    f4_3 = BatchNormalization(epsilon=epsilon)(f4_3)
    f4_3_ = Activation('relu')(f4_3)

    u2 = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(f4_3_)
    u2 = BatchNormalization(epsilon=epsilon)(u2)
    s2 = Average()([u2, f1_2])
    s2_ = Activation('relu')(s2)
    f5_1 = Conv2D(128, kernel_size=3, padding='same')(s2_)
    f5_1 = BatchNormalization(epsilon=epsilon)(f5_1)
    f5_1_ = Activation('relu')(f5_1)
    f5_2 = Conv2D(64, kernel_size=3, padding='same')(f5_1_)
    f5_2 = BatchNormalization(epsilon=epsilon)(f5_2)
    f5_2_ = Activation('relu')(f5_2)

    u3 = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(f5_2_)
    u3 = BatchNormalization(epsilon=epsilon)(u3)
    s3 = Average()([u3, f0_2]) # Change from author implemantation due to tensor shape
    s3_ = Activation('relu')(s3)
    f6_1 = Conv2D(15, kernel_size=3, padding='same')(s3_)
    f6_1 = BatchNormalization(epsilon=epsilon)(f6_1)
    f6_1_ = Activation('relu')(f6_1)
    f6_2 = Conv2D(3, kernel_size=3, padding='same')(f6_1_)
    f6_2 = BatchNormalization(epsilon=epsilon)(f6_2)
    f6_2_ = Activation('relu')(f6_2)

    inputs_mid = crop(3,n_stack//2*3,(n_stack//2+1)*3)(inputs)
    s4 = Average()([inputs_mid, f6_2])
    s4_ = Activation('sigmoid')(s4)

    model = Model(inputs=inputs, outputs=s4_)
    
    return model


def plot_history(history, output_dir):
    # Setting Parameters
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 1) Accuracy Plt
    plt.plot(epochs, acc, 'bo' ,label = 'training acc')
    plt.plot(epochs, val_acc, 'b' , label= 'validation acc')
    plt.title('Training and Validation acc')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'acc.png'))
    plt.figure()

    # 2) Loss Plt
    plt.plot(epochs, loss, 'bo' ,label = 'training loss')
    plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'loss.png'))


def step_decay(epoch, lr=0.005, first_iter=24000, half_iter=8000, min_lr=0.000001):
    if epoch <= first_iter:
        return lr
    else:
        lr_tmp = lr / (2 ** ((epoch-first_iter)//half_iter))
        return min(lr_tmp, min_lr)


def train(n_epochs, data_dir, id_dir, output_dir, lr=0.005):
    
    f = open(os.path.join(id_dir,'train_id.pickle'), 'rb')
    train_IDs = pickle.load(f)
    
    model = model_generator()
    model.summary()
    
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='MSE', metrics=['accuracy'])

    train_gen = data_generator.DataGenerator(train_IDs[:100], data_dir)
    val_gen = data_generator.DataGenerator(train_IDs[500:532], data_dir)

    model_save_path = os.path.join(output_dir,'model_weights.h5')
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=model_save_path, monitor="val_loss", verbose=1, save_best_only=True, period=1
    )

    lr_decay = LearningRateScheduler(step_decay)

    model.fit_generator(train_gen,
                    steps_per_epoch=train_gen.__len__(), 
                    validation_data=val_gen, 
                    validation_steps=val_gen.__len__(),
                    epochs=n_epochs,
                    shuffle=True,
                    callbacks=[model_checkpoint, lr_decay])
    
    history = model.history
    plot_history(history, output_dir)
    
    return


if __name__ == '__main__':

    args = parse_args()

    train(
        n_epochs=args.n_epochs,
        data_dir=args.data_dir,
        id_dir=args.id_dir,
        output_dir=args.output_dir,
        lr=args.lr,
    )