"""
extract landmarks from data.csv
and save them as a npy file
"""
import cv2
import mediapipe as mp
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='landmark extractor')
parser.add_argument('--data', required=True, help='data csv file generated from train_test_split')
parser.add_argument('--normalized', default='y', help='y or n, if y record normalized coordinates (default=y)')
parser.add_argument('--exclude_z', default='y', help='y or n, if y record x,y coordinates only')
args = parser.parse_args()

mp_hands = mp.solutions.hands


def get_landmarks(img, normalized):
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7) as hands:
        img = cv2.flip(img, 1)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # BGR to RGB

        if not normalized:
            image_height, image_width, _ = img.shape
            for hand_landmarks in results.multi_hand_landmarks:  # per hand
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )

        print('Handedness:', results.multi_handedness)
        return results.multi_hand_landmarks


def extract_landmarks(file_path: str, normalized=False) -> None:
    dir_path, file_name = file_path.rsplit('/', 1)
    print(dir_path, file_name)  # TODO: logger
    file_name = file_name.rsplit('.', 1)[0]

    img = cv2.imread(file_path)

    if img is None:  # is video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f'could not read file {file_path}')

        landmarks = []
        while cap.isOpened():
            ret, frame = cap.read()  # BGR
            if not ret:
                break

            results = get_landmarks(frame, normalized)
            if results is None:
                continue
            landmarks.append(results.multi_handedness[0])

        cap.release()
        np.save(file_path + file_name + '.npy', np.array(landmarks))
    else:  # is img
        results = get_landmarks(img, normalized)
        if results is None:
            return
        np.save(file_path + file_name + '.npy', results.multi_handedness[0])


df = pd.read_csv(args.data)

for s in df['path']:
    extract_landmarks(file_path=..., normalized=False)
