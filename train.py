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
import model_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Parameter for training.")

    parser.add_argument(
        "--data_dir", type=str, help="path to .npy file of input data", default="data"
    )
    parser.add_argument(
        "--id_dir", type=str, help="dir of id_list", default="data/id_list"
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
    parser.add_argument(
        "--train_val_split", type=float, help="ratio of val", default=0.1
    )

    return parser.parse_args()


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


# need to adjust parameters in this function due to keras implementation rule
def step_decay(epoch, lr=0.005, first_iter=24000, half_iter=8000, min_lr=0.000001):
    if epoch <= first_iter:
        return lr
    else:
        lr_tmp = lr / (2 ** ((epoch-first_iter)//half_iter))
        return min(lr_tmp, min_lr)


def train(n_epochs, data_dir, id_dir, output_dir, lr, train_val_split):
    
    f = open(os.path.join(id_dir,'train_id.pickle'), 'rb')
    train_IDs = pickle.load(f)
    
    model = model_generator.ModelGenerator().model()
    model.summary()
    
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='MSE', metrics=['accuracy'])

    val_idx = int(len(train_IDs) * train_val_split)
    val_gen = data_generator.DataGenerator(train_IDs[:val_idx], data_dir)
    train_gen = data_generator.DataGenerator(train_IDs[val_idx:], data_dir)

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
        train_val_split=args.train_val_split,
    )