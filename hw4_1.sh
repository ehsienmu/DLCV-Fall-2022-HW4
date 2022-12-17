#!/bin/bash
# python3 .py $1 $2
# wget  -O hw4_1_r10921a36.tar
python3 p1_main.py --config configs/nerf/dlcv.py --render_only --render_test_dlcv --dump_images --render_test_dlcv_json_path $1 --render_test_dlcv_output_path $2 --render_test_dlcv_model_file hw4_1_r10921a36.tar
# python3 p1_main.py --config configs/nerf/dlcv.py --render_only --render_test_dlcv --dump_images --render_test_dlcv_json_path /home/hsien/dlcv/hw4-ehsienmu/hw4_data/hotdog_copy/transforms_test.json --render_test_dlcv_output_path ./old_model --render_test_dlcv_model_file /home/hsien/dlcv/hw4-ehsienmu/old_logs/nerf_synthetic/dvgo_hotdog/fine_last.tar
# grade
# python3 grade.py ./old_model /home/hsien/dlcv/hw4-ehsienmu/hw4_data/hotdog/test/
# TODO - run your inference Python3 code