#!/bin/sh

#rm -rf datasets test-results trained-model

#mkdir test-results
#mkdir datasets
#mkdir trained_model

#cd datasets
#wget https://cloud.sylvainlobry.com/s/4Qg5AXX8YfCswmX/download
#unzip download
#rm -rf download
#cd RSVQA_LR
#unzip Images_LR.zip
#rm -rf Images_LR.zip
#cd ../..

#cd datasets
#wget https://cloud.sylvainlobry.com/s/f7NpYQKqx4bZStx/download
#unzip download
#rm -rf download
#cd ..

#cd datasets
#mkdir RSVQA_BEN
#cd RSVQA_BEN
#wget "https://zenodo.org/record/5084904/files/Images.zip?download=1"
#unzip "Images.zip?download=1"
#rm -rf "Images.zip?download=1"
#wget "https://zenodo.org/record/5084904/files/RSVQAxBEN_split_test_answers.json?download=1"
#mv "RSVQAxBEN_split_test_answers.json?download=1" RSVQAxBEN_split_test_answers.json
#wget "https://zenodo.org/record/5084904/files/RSVQAxBEN_split_test_images.json?download=1"
#mv "RSVQAxBEN_split_test_images.json?download=1" RSVQAxBEN_split_test_images.json
#wget "https://zenodo.org/record/5084904/files/RSVQAxBEN_split_test_questions.json?download=1"
#mv "RSVQAxBEN_split_test_questions.json?download=1" RSVQAxBEN_split_test_questions.json
#wget "https://zenodo.org/record/5084904/files/RSVQAxBEN_split_train_answers.json?download=1"
#mv "RSVQAxBEN_split_train_answers.json?download=1" RSVQAxBEN_split_train_answers.json
#wget "https://zenodo.org/record/5084904/files/RSVQAxBEN_split_train_images.json?download=1"
#mv "RSVQAxBEN_split_train_images.json?download=1" RSVQAxBEN_split_train_images.json
#wget "https://zenodo.org/record/5084904/files/RSVQAxBEN_split_train_questions.json?download=1"
#mv "RSVQAxBEN_split_train_questions.json?download=1" RSVQAxBEN_split_train_questions.json
#wget "https://zenodo.org/record/5084904/files/RSVQAxBEN_split_val_answers.json?download=1"
#mv "RSVQAxBEN_split_val_answers.json?download=1" RSVQAxBEN_split_val_answers.json
#wget "https://zenodo.org/record/5084904/files/RSVQAxBEN_split_val_images.json?download=1"
#mv "RSVQAxBEN_split_val_images.json?download=1" RSVQAxBEN_split_val_images.json
#wget "https://zenodo.org/record/5084904/files/RSVQAxBEN_split_val_questions.json?download=1"
#mv "RSVQAxBEN_split_val_questions.json?download=1" RSVQAxBEN_split_val_questions.json
#cd ../..

## TODO: Get images associated to the different sources for this dataset.
#cd datasets
#wget https://github.com/spectralpublic/RSIVQA/archive/refs/heads/main.zip
#unzip main.zip
#mv RSIVQA-main/RSIVQA .
#rm -rf main.zip RSIVQA-main
#cd ..

#cd preprocess
#python3 rsvqa_lr_data.py
#python3 rsvqa_hr_data.py
#python3 rsvqa_ben_data.py
#python3 rsvqa_rsivqa_data.py
#cd ..

PYTHONPATH=. python3 rsvqa/train.py --run_name='vqa_run_name' --cnn_encoder='tf_efficientnetv2_m' --transformer_model='realformer' --data_dir="datasets/RSVQA_LR" --model_dir='trained-model' --batch_size=4 --num_vis=5 --hidden_size=768 --num_workers=16 --save_dir="test-results" --loss='ASLSingleLabel' --epochs=100
