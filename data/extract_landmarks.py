"""
extract landmarks from data.csv
and save them as a npy file
"""
from typing import Union, List

import cv2
import mediapipe as mp
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description='landmark extractor')
parser.add_argument('--data', required=True, help='data csv file generated from train_test_split')
parser.add_argument('--normalized', type=int, default=1, help='1 or 0, if 1 record normalized coordinates (default=1)')
parser.add_argument('--exclude_z', type=int, default=1, help='1 or 0, if 1 record x,y coordinates only')
args = parser.parse_args()

mp_hands = mp.solutions.hands


def get_landmarks(img: np.ndarray, normalized, exclude_z) -> Union[None, List[tuple]]:
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5) as hands:
        img: np.ndarray = cv2.flip(img, 1)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # BGR to RGB

        image_height, image_width, _ = img.shape
        normalize_func = lambda coor: (coor[0] * image_width, coor[1] * image_height, *coor[2:])

        print('Handedness:', results.multi_handedness)
        results = results.multi_hand_landmarks
        if results is None:
            return None
        landmarks = results[0].landmark

        landmarks_list = []
        for coor in landmarks:
            coor = (coor.x, coor.y) if exclude_z else (coor.x, coor.y, coor.z)
            coor = coor if normalized else normalize_func(coor)
            landmarks_list.append(coor)

        return landmarks_list


def extract_landmarks(file_path: str, normalized=False, exclude_z=True) -> None:
    dir_path, file_name = file_path.rsplit('/', 1)
    file_name = file_name.rsplit('.', 1)[0]

    img: np.ndarray = cv2.imread(file_path)

    if img is None:  # is video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f'could not read file {file_path}')

        landmarks_collection = []  # (# of frames, 21, 2 or 3)
        while cap.isOpened():
            ret, frame = cap.read()  # BGR
            if not ret:
                break

            landmarks = get_landmarks(frame, normalized, exclude_z)
            if landmarks is None:
                continue
            landmarks_collection.append(landmarks)

        cap.release()
        np.save(Path(dir_path).joinpath(file_name + '.npy'), np.array(landmarks_collection))
    else:  # is img
        landmarks = get_landmarks(img, normalized, exclude_z)
        if landmarks is None:
            return
        np.save(Path(dir_path).joinpath(file_name + '.npy'), landmarks)


df = pd.read_csv(args.data)

for file_path in df['path']:
    extract_landmarks(file_path=file_path, normalized=args.normalized, exclude_z=args.exclude_z)
