import argparse
from pathlib import Path
import cv2
import mediapipe as mp

BASE_DIR = Path(__file__).parent

parser = argparse.ArgumentParser(description='landmark extractor')
parser.add_argument('--train', required=True, help='train csv file')
parser.add_argument('--test', required=True, help='test csv file')
parser.add_argument('--normalized', type=int, default=1, help='0 or 1, if 1 record normalized coordinates (default=1)')
parser.add_argument('--save_as', default='npy', help='npy or csv')
args = parser.parse_args()



vid_path = BASE_DIR.joinpath('dataset/1/1/22.mov').__str__()
cap = cv2.VideoCapture(vid_path)  # 0 if webcam
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if not cap.isOpened():
    raise IOError(f'could not read file {vid_path}')

while cap.isOpened():
    # Capture frame-by-frame
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
        image_height, image_width, _ = img.shape
        annotated_image = img.copy()
        for hand_landmarks in results.multi_hand_landmarks:  # per hand
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('annotated_image', annotated_image)  # display img for progress

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
