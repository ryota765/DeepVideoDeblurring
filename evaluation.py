import argparse
import os

import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils import data_generator
from utils import model_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Parameter for evaluation.")

    parser.add_argument(
        "--data_dir", type=str, help="path to .npy file of input data", default="data"
    )
    parser.add_argument(
        "--test_id_path", type=str, help="path to test_id_list", default="id_list/test_id.pickle"
    )
    parser.add_argument(
        "--model_path", type=str, help="path to trained model weights", default="data/output/model_weights.h5"
    )
    parser.add_argument(
        "--output_dir", type=str, help="path to output dir", default="data/output"
    )
    parser.add_argument(
        "--batch_size", type=int, help="number of batch for prediction", default=16
    )
    parser.add_argument(
        "--n_bins", type=int, help="number of bins for histogram", default=16
    )

    return parser.parse_args()


def draw_histogram(history_list, output_dir, n_bins=16):
    history_list_t = np.array(history_list).T.tolist()

    max_input = max(history_list_t[0])
    min_input = min(history_list_t[0])

    bin_list = np.linspace(min_input, max_input, n_bins).tolist()
    hist_list = [[] for _ in range(n_bins)]

    for elem in history_list:
        input_psnr = elem[0]
        dif_psnr = elem[1]
        for i, threshold in enumerate(bin_list):
            if input_psnr <= threshold:
                hist_list[i].append(dif_psnr)
                break

    hist_mean_list = [np.mean(row) if len(row) > 0 else 0 for row in hist_list]

    plt.plot(bin_list, hist_mean_list)
    plt.savefig(os.path.join(output_dir, 'histogram.png'))


def evaluate(data_dir, test_id_path, model_path, output_dir='output', batch_size=16, n_bins=16, dim=(128,128), n_stack=5, n_channels=3):

    f = open(os.path.join(data_dir, test_id_path), 'rb')
    test_id_list = pickle.load(f)

    model = model_generator.ModelGenerator().model()
    model.load_weights(model_path)

    history_list = []

    for i in range(len(test_id_list)//batch_size):
        X, y = data_generator.load_batch_data(test_id_list[i*batch_size:(i+1)*batch_size], data_dir)
        pred = model.predict(X)
        
        X = (X*255).astype(np.uint8)
        y = (y*255).astype(np.uint8)
        pred = (pred*255).astype(np.uint8)

        for gt_img, input_img, pred_img, in zip(y, X[:,:,:,n_channels*(n_stack//2):n_channels*(n_stack//2+1)], pred):
            input_psnr = cv2.PSNR(gt_img, input_img) # only accepts uint8
            pred_psnr = cv2.PSNR(gt_img, pred_img)
            history_list.append([input_psnr, pred_psnr-input_psnr])

    draw_histogram(history_list, output_dir, n_bins)

    return

    
if __name__ == '__main__':

    args = parse_args()

    evaluate(
        data_dir=args.data_dir,
        test_id_path=args.test_id_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        n_bins=args.n_bins,
    )
    

        

    


