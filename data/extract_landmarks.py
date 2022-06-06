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


def extract_landmarks(file_path: str, normalized=True):
    dir_path, file_name = file_path.rsplit('/', 1)
    print(dir_path, file_name)  # TODO: logger
    file_name = file_name.rsplit('.', 1)[0]

    img = cv2.imread(file_path)
    mp_hands = mp.solutions.hands

    if img is None:
        # is video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f'could not read file {file_path}')

        while cap.isOpened():
            ret, frame = cap.read()  # BGR

            if not ret:
                break

            # MEDIAPIPE
            with mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=0.7) as hands:
                img = cv2.flip(frame, 1)
                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # BGR to RGB

                print('Handedness:', results.multi_handedness)
                if not results.multi_hand_landmarks:
                    continue

                np.save(file_path + file_name + '.npy', results.multi_handedness[0])

        cap.release()

        return ...
    else:
        # is img
        with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.7) as hands:
            img = cv2.flip(img, 1)
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # BGR to RGB
            print('Handedness:', results.multi_handedness)

            if not results.multi_hand_landmarks:
                return
            image_height, image_width, _ = img.shape
            annotated_image = img.copy()
            for hand_landmarks in results.multi_hand_landmarks:  # per hand
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )


    # Closes all the frames
    # cv2.destroyAllWindows()


df = pd.read_csv(args.data)

for s in df['path']:

