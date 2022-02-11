import pandas as pd
import json
import os
import glob
from PIL import Image

data_dir = os.path.join('..', 'Remote_Sensing_Datasets/RSVQA_LR')

train_data_images = json.load(open(os.path.join(data_dir, 'LR_split_train_images.json')))["images"]
train_data_questions = json.load(open(os.path.join(data_dir, 'LR_split_train_questions.json')))["questions"]
train_data_answers = json.load(open(os.path.join(data_dir, 'LR_split_train_answers.json')))["answers"]

test_data_images = json.load(open(os.path.join(data_dir, 'LR_split_test_images.json')))["images"]
test_data_questions = json.load(open(os.path.join(data_dir, 'LR_split_test_questions.json')))["questions"]
test_data_answers = json.load(open(os.path.join(data_dir, 'LR_split_test_answers.json')))["answers"]

val_data_images = json.load(open(os.path.join(data_dir, 'LR_split_val_images.json')))["images"]
val_data_questions = json.load(open(os.path.join(data_dir, 'LR_split_val_questions.json')))["questions"]
val_data_answers = json.load(open(os.path.join(data_dir, 'LR_split_val_answers.json')))["answers"]

train_data_images = [x for x in train_data_images if x['active'] == True]
train_data_questions = [x for x in train_data_questions if x['active'] == True]
train_data_answers = [x for x in train_data_answers if x['active'] == True]

test_data_images = [x for x in test_data_images if x['active'] == True]
test_data_questions = [x for x in test_data_questions if x['active'] == True]
test_data_answers = [x for x in test_data_answers if x['active'] == True]

val_data_images = [x for x in val_data_images if x['active'] == True]
val_data_questions = [x for x in val_data_questions if x['active'] == True]
val_data_answers = [x for x in val_data_answers if x['active'] == True]

train_df = pd.DataFrame(train_data_questions)
train_df = train_df.merge(pd.DataFrame(train_data_answers), how='inner', suffixes=('_1', '_2'), left_on = 'id', right_on = 'question_id')
train_df = train_df.drop(['id_1', 'id_2','date_added_1','date_added_2','people_id_1','people_id_2','active_1','active_2','question_id','answers_ids'], axis=1)
train_df = train_df.rename(columns={"type": "category"})
train_df.insert(0, 'mode', 'train')

test_df = pd.DataFrame(test_data_questions)
test_df = test_df.merge(pd.DataFrame(test_data_answers), how='inner', suffixes=('_1', '_2'), left_on = 'id', right_on = 'question_id')
test_df = test_df.drop(['id_1', 'id_2','date_added_1','date_added_2','people_id_1','people_id_2','active_1','active_2','question_id','answers_ids'], axis=1)
test_df = test_df.rename(columns={"type": "category"})
test_df.insert(0, 'mode', 'test')

val_df = pd.DataFrame(val_data_questions)
val_df = val_df.merge(pd.DataFrame(val_data_answers), how='inner', suffixes=('_1', '_2'), left_on = 'id', right_on = 'question_id')
val_df = val_df.drop(['id_1', 'id_2','date_added_1','date_added_2','people_id_1','people_id_2','active_1','active_2','question_id','answers_ids'], axis=1)
val_df = val_df.rename(columns={"type": "category"})
val_df.insert(0, 'mode', 'val')

for aux in glob.glob(os.path.join(data_dir, 'Images_LR') + "/*.tif"):
    im = Image.open(aux)
    out = im.convert("RGB")
    out.save(aux[:-3] + "jpg", "JPEG", quality=100)    
    print(aux)

train_df.to_csv(os.path.join(data_dir, 'traindf.csv'), index=False)
val_df.to_csv(os.path.join(data_dir, 'valdf.csv'), index=False)
test_df.to_csv(os.path.join(data_dir, 'testdf.csv'), index=False)

