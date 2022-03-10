import pandas as pd
import json
import os
import glob
from PIL import Image

def area_func(x, cat):
    if cat == 'area':
        area = x.split('m2')[0]
        area = int(area)
        if area == 0:
            return '0'
        elif area >=1 and area <= 10:
            return 'between 1 and 10'
        elif area >= 11 and area <= 100:
            return 'between 11 and 100'
        elif area >= 101 and area <= 1000:
            return 'between 101 and 1000'
        elif area > 1000:
            return 'more than 1000'
    else:
        return x

data_dir = os.path.join('..', 'Remote_Sensing_Datasets/RSVQA_HR/')

test_data_images = json.load(open(os.path.join(data_dir, 'USGS_split_test_phili_images.json')))["images"]
test_data_questions = json.load(open(os.path.join(data_dir, 'USGS_split_test_phili_questions.json')))["questions"]
test_data_answers = json.load(open(os.path.join(data_dir, 'USGS_split_test_phili_answers.json')))["answers"]


test_data_images = [x for x in test_data_images if x['active'] == True]
test_data_questions = [x for x in test_data_questions if x['active'] == True]
test_data_answers = [x for x in test_data_answers if x['active'] == True]

test_df = pd.DataFrame(test_data_questions)
test_df = test_df.merge(pd.DataFrame(test_data_answers), how='inner', suffixes=('_1', '_2'), left_on = 'id', right_on = 'question_id')
test_df = test_df.drop(['id_1', 'id_2','date_added_1','date_added_2','people_id_1','people_id_2','active_1','active_2','question_id','answers_ids'], axis=1)
test_df = test_df.rename(columns={"type": "category"})
test_df.insert(0, 'mode', 'test')



categories = test_df['category'].unique()
print('writing')
if 'area' in categories:
    print('Processing area category')
    test_df['answer'] = test_df.apply(lambda x: area_func(x['answer'], x['category']), axis=1)

test_df.to_csv(os.path.join(data_dir, 'testdf_phili.csv'), index=False)

