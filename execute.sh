#!/bin/sh
rm -rf datasets test-results
mkdir datasets
cd datasets
wget https://cloud.sylvainlobry.com/s/4Qg5AXX8YfCswmX/download
unzip download
rm -rf download
cd RSVQA_LR
unzip Images_LR.zip
rm -rf Images_LR.zip
cd ..
wget https://cloud.sylvainlobry.com/s/f7NpYQKqx4bZStx/download
unzip download
rm -rf download
cd ..

mkdir test-results
python rsvqa/train.py --run_name='vqa_run_name' --cnn_encoder='tf_efficientnetv2_m' --transformer_model='realformer' --data_dir="RSVQA_LR" --use_pretrained --model_dir='path_to_pretrained_model' --batch_size=16 --num_vis=5 --hidden_size=768 --num_workers=16 --save_dir="test-results" --loss='ASLSingleLabel' --epochs=100
