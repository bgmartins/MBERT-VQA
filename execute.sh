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
#cd ..
#wget https://cloud.sylvainlobry.com/s/f7NpYQKqx4bZStx/download
#unzip download
#rm -rf download
#cd ..

#cd preprocess
#python3 rsvqa_lr_data.py
#cd ..

PYTHONPATH=. python3 rsvqa/train.py --run_name='vqa_run_name' --cnn_encoder='tf_efficientnetv2_m' --transformer_model='realformer' --data_dir="datasets/RSVQA_LR" --model_dir='trained-model' --batch_size=4 --num_vis=5 --hidden_size=768 --num_workers=16 --save_dir="test-results" --loss='ASLSingleLabel' --epochs=100
