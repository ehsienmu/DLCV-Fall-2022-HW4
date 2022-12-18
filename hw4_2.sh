#!/bin/bash
# python3 .py $1 $2 $3
# wget  -O hw4_2_r10921a36.pt
python3 p2_test.py --input_csv_file $1 --input_dir $2 --output_file $3 --model_file hw4_2_r10921a36.pt
# python3 p1_test.py --input_csv_file ./hw4_data/office/val.csv --input_dir ./hw4_data/office/val/ --model_file ./resnet_ckpt/best_resnet_model.pt --output_file ./offtest.csv
# TODO - run your inference Python3 code