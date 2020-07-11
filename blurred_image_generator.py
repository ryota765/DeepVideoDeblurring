import glob
import os
import argparse

import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Parameter for training.")

    parser.add_argument(
        "--movie_path", type=str, help="path to target movie"
    )
    parser.add_argument(
        "--surround_num", type=int, help="number of frames to use for bluuring", default=7
    )
    parser.add_argument(
        "--ext", type=str, help="extension for output image data", default="jpg"
    )
    parser.add_argument(
        "--save_dir", type=str, help="directory for output", default="blur_data/frame"
    )

    return parser.parse_args()


def blur_image_generator(frame_list):
    surround_num = len(frame_list)
    
    gt_img = frame_list[surround_num//2]
    
    frame_array = np.array(frame_list)
    input_img = np.mean(frame_array, axis=0).astype(np.uint8)
    
    return gt_img, input_img


def movie_2_blurred_frame(movie_path, surround_num, ext, save_dir):
    
    cap = cv2.VideoCapture(movie_path)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    digit = len(str(frame_num // surround_num))
    frame_count = 0
    surround_frame_list = []
    
    gt_save_path = os.path.join(save_dir, 'GT')
    input_save_path = os.path.join(save_dir, 'input')
    
    os.makedirs(gt_save_path, exist_ok=True)
    os.makedirs(input_save_path, exist_ok=True)

    for i in range(frame_num):
        ret, frame = cap.read()
        if ret:
            surround_frame_list.append(frame)
            
            if frame_count % surround_num == 6:
                assert len(surround_frame_list) == surround_num
                gt_img, input_img = blur_image_generator(surround_frame_list)
                
                cv2.imwrite(os.path.join(gt_save_path, '{}.{}'.format(str(frame_count//surround_num).zfill(digit), ext)), gt_img)
                cv2.imwrite(os.path.join(input_save_path, '{}.{}'.format(str(frame_count//surround_num).zfill(digit), ext)), input_img)
                
                surround_frame_list = []
            
            frame_count += 1
    return


if __name__ == '__main__':

    args = parse_args()

    movie_2_blurred_frame(
        movie_path=args.movie_path,
        surround_num=args.surround_num,
        ext=args.ext,
        save_dir=args.save_dir,
    )