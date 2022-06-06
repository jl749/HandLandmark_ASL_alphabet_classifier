"""
python train_test_split.py --data_dir {path}

looks up the data path recursively and create train/test csv files

data_dir should contain data like the following format
e.g) **/class_name/1.jpg
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.extract_landmarks import extract_landmarks

parser = argparse.ArgumentParser(description='train test split')
# TODO: seed
parser.add_argument('--data_dir', required=True, help='dataset folder path')
args = parser.parse_args()

IMG_EXTENSIONS = ['jpg', 'jpeg']  # TODO: 4channel img
VID_EXTENSIONS = ['mov', 'avi', 'mp4']

dataset_dir = Path(args.data_dir).resolve()

all_files = list()
for e in IMG_EXTENSIONS + VID_EXTENSIONS:
    all_files += list(dataset_dir.rglob(f'*.{e}'))

data_dict = {'path': [], 'class': []}
for p in all_files:
    class_name = p.parent.stem
    data_dict['path'].append(p.__str__())
    data_dict['class'].append(class_name)

df = pd.DataFrame(data_dict)
df.to_csv('data.csv')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, shuffle=True, stratify=df['class'])
train_df.to_csv('train.csv')
test_df.to_csv('test.csv')

train_classes = train_df['class']
print(f'classes: {train_classes.unique()}\n'
      f'num_classes: {len(train_classes.unique())}\n\n'
      f'{train_classes.value_counts()}')
