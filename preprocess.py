import glob
import os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle


def load_image_from_dir(img_dir):
    input_paths = glob.glob(os.path.join(img_dir, "input/*.jpg"))
    gt_paths = glob.glob(os.path.join(img_dir, "GT/*.jpg"))
    print(img_dir, len(input_paths))

    input_list = []
    gt_list = []

    for input_path, gt_path in zip(input_paths, gt_paths):
        input_img = np.array(Image.open(input_path))
        input_list.append(input_img)
        gt_img = np.array(Image.open(gt_path))
        gt_list.append(gt_img)

    return np.array(input_list), np.array(gt_list)


# methods for augmentation
def flip(img):
    return np.array([img, np.fliplr(img)])


def rotate(img):
    img_list = [img]

    for i in range(1, 4):
        img_rot = np.rot90(img, i)
        img_list.append(img_rot)

    return np.array(img_list)


def scale(img, scale_list=[2, 3, 4]):
    img_list = [img]
    height, width, _ = img.shape

    for scale in scale_list:
        _height = height // scale
        _width = width // scale
        img_resized = cv2.resize(img, (_width, _height))
        img_list.append(img_resized)

    return np.array(img_list)


# process gt and input simultaneously to align random crop
def random_crop(input_img, gt_img, crop_seed, crop_size=(128, 128)):

    input_img_list = []
    gt_img_list = []

    height, width, _ = input_img.shape

    for i in range(len(crop_seed)):

        # adjust seed for each iteration to get sequential input data
        np.random.seed(seed=crop_seed[i])
        top = np.random.randint(0, height - crop_size[0])

        np.random.seed(seed=crop_seed[i])
        left = np.random.randint(0, width - crop_size[1])

        bottom = top + crop_size[0]
        right = left + crop_size[1]

        input_img_cropped = input_img[top:bottom, left:right, :]
        gt_img_cropped = gt_img[top:bottom, left:right, :]

        input_img_list.append(input_img_cropped)
        gt_img_list.append(gt_img_cropped)

    return np.array(input_img_list), np.array(gt_img_list)


# flip, rotate, scale, crop is done refering to original paper
def augmentation(input_img, gt_img, crop_seed):
    input_img_list = []
    gt_img_list = []

    flip_input_img_array = flip(input_img)
    flip_gt_img_array = flip(gt_img)

    for i in range(flip_input_img_array.shape[0]):
        rotate_input_img_array = rotate(flip_input_img_array[i])
        rotate_gt_img_array = rotate(flip_gt_img_array[i])
        for j in range(rotate_input_img_array.shape[0]):
            scale_input_img_array = scale(rotate_input_img_array[j])
            scale_gt_img_array = scale(rotate_gt_img_array[j])
            for k in range(scale_input_img_array.shape[0]):
                crop_input_img_array, crop_gt_img_array = random_crop(
                    scale_input_img_array[k], scale_gt_img_array[k], crop_seed
                )
                crop_input_img_array.tolist()
                input_img_list.extend(crop_input_img_array)
                crop_gt_img_array.tolist()
                gt_img_list.extend(crop_gt_img_array)

    return np.array(input_img_list), np.array(gt_img_list)


def main(data_path, output_dir, n_stack, crop_num):

    train_IDs = []
    test_IDs = []

    movie_paths = glob.glob(os.path.join(data_path, "*"))

    # test split by author (https://github.com/shuochsu/DeepVideoDeblurring/issues/2)
    test_paths = [
        "IMG_0030",
        "IMG_0049",
        "IMG_0021",
        "IMG_0032",
        "IMG_0033",
        "IMG_0031",
        "IMG_0003",
        "IMG_0039",
        "IMG_0037",
        "720p_240fps_2",
    ]

    for movie_id, path in enumerate(movie_paths):

        movie_name = path.split("/")[-1]

        if movie_name in test_paths:
            is_train = False
        else:
            is_train = True

        movie_id = str(movie_id).zfill(2)

        input_array, gt_array = load_image_from_dir(path)
        crop_seed = [np.random.randint(0, 100) for i in range(crop_num)]

        for num_id, (input_img, gt_img) in enumerate(zip(input_array, gt_array)):

            # TODO: change indexing from 0 to len-n_stack
            id_flag = (
                False
                if (
                    num_id <= n_stack // 2
                    or num_id >= (len(input_array) - n_stack // 2)
                )
                else True
            )
            num_id = str(num_id).zfill(3)

            input_array_aug, gt_array_aug = augmentation(input_img, gt_img, crop_seed)

            for aug_id, (input_img, gt_img) in enumerate(
                zip(input_array_aug, gt_array_aug)
            ):

                aug_id = str(aug_id).zfill(3)
                img_id = movie_id + aug_id + num_id

                np.save(
                    os.path.join(output_dir, "input_data/{}.npy".format(img_id)),
                    input_img,
                )
                np.save(
                    os.path.join(output_dir, "gt_data/{}.npy".format(img_id)), gt_img
                )

                if id_flag == True:
                    if is_train == True:
                        train_IDs.append(img_id)
                    else:
                        test_IDs.append(img_id)

    print("train_ids:", len(train_IDs), "test_ids:", len(test_IDs))

    f = open(os.path.join(output_dir, "id_list/train_id.pickle"), "wb")
    pickle.dump(train_IDs, f)

    f = open(os.path.join(output_dir, "id_list/test_id.pickle"), "wb")
    pickle.dump(test_IDs, f)


if __name__ == "__main__":
    main(
        data_path="data/DeepVideoDeblurring_Dataset/quantitative_datasets",
        output_dir="data",
        n_stack=5,  # number of frames in a clip
        crop_num=10,  # number of random crops
    )
