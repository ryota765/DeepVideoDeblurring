import os
import argparse

import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import data_generator
from utils import model_generator


def parse_args():
    parser = argparse.ArgumentParser(description="Parameter for prediction.")

    parser.add_argument(
        "--data_dir", type=str, help="path to .npy file of input data", default="data"
    )
    parser.add_argument(
        "--test_id_path",
        type=str,
        help="path to test_id_list",
        default="id_list/test_id.pickle",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to trained model weights",
        default="data/output/model_weights.h5",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="path to output dir",
        default="data/output/prediction",
    )
    parser.add_argument(
        "--batch_size", type=int, help="number of batch for prediction", default=1
    )

    return parser.parse_args()


def predict(
    data_dir, test_id_path, model_path, output_dir, batch_size, n_stack=5, n_channels=3
):

    f = open(os.path.join(data_dir, test_id_path), "rb")
    test_id_list = pickle.load(f)[10:20]

    """
    test_id_list = ['00160016',
                    '00291057',
                    '00214049',
                    '01053087',
                    '01072049',
                    '00235087',
                    '00182093',
                    '01108097',
                    '01127074',
                    '01306016',
                    '01290060',
                    '01161021',
                    '00160002',
                    '00291043',
                    '00307035',
                    '00126057',
                    '00252008',
                    '01053093',
                    '01034008',
                    '00182087']
                    """

    os.makedirs(output_dir, exist_ok=True)

    model = model_generator.ModelGenerator().model()
    model.load_weights(model_path)

    for i in range(len(test_id_list) // batch_size):

        X, y = data_generator.load_batch_data(
            test_id_list[i * batch_size : (i + 1) * batch_size],
            data_dir,
            batch_size=batch_size,
        )

        pred = model.predict(X)

        X = (X * 255).astype(np.uint8)
        y = (y * 255).astype(np.uint8)
        pred = (pred * 255).astype(np.uint8)

        plt.subplot(1, 3, 1)
        plt.imshow(y[0])
        plt.subplot(1, 3, 2)
        plt.imshow(
            X[0][:, :, n_channels * (n_stack // 2) : n_channels * (n_stack // 2 + 1)]
        )
        plt.subplot(1, 3, 3)
        plt.imshow(pred[0])

        plt.savefig(os.path.join(output_dir, "{}.png".format(test_id_list[i])))


if __name__ == "__main__":

    args = parse_args()

    predict(
        data_dir=args.data_dir,
        test_id_path=args.test_id_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
