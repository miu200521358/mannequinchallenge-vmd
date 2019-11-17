@echo off
rem --- 
rem ---  映像データから深度推定を行う
rem --- 

cls

rem ---  カレントディレクトリを実行先に変更
cd /d %~dp0

rem ---  入力対象映像ファイルパス
set INPUT_VIDEO="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\heart\heart_mini2.mp4"
set OPENPOSE_JSON="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\heart\heart_mini2_20190203_164123\heart_mini2_json"
set NUMBER_PEOPLE_MAX=1
set PAST_DEPTH_PATH=

set INPUT_VIDEO="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\night\night_zoom.mp4"
set OPENPOSE_JSON="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\night\night_zoom_20190416_131417\night_zoom_json"
set NUMBER_PEOPLE_MAX=3
set PAST_DEPTH_PATH=

set INPUT_VIDEO="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\04\yoiyoi\yoiyoi_3388-3530.mp4"
set OPENPOSE_JSON="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\04\yoiyoi\yoiyoi_3388-3530_20191108_043201\yoiyoi_3388-3530_json"
set NUMBER_PEOPLE_MAX=4
set PAST_DEPTH_PATH="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\04\yoiyoi\yoiyoi_3388-3530_20191108_043201\yoiyoi_3388-3530_json_20191108_043201_depth"

set INPUT_VIDEO="E:/MMD/MikuMikuDance_v926x64/Work/201805_auto/04/yoiyoi/yoiyoi_3388-3530.mp4"
set OPENPOSE_JSON="E:/MMD/MikuMikuDance_v926x64/Work/201805_auto/04/yoiyoi/yoiyoi_3388-3530_20191108_043201/yoiyoi_3388-3530_json"
set NUMBER_PEOPLE_MAX=4
set PAST_DEPTH_PATH=

set INPUT_VIDEO="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\ivory\ivory.mp4"
set OPENPOSE_JSON="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\ivory\ivory_20191112_044054\ivory_json"
set PAST_DEPTH_PATH="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\ivory\ivory_20191112_044054\ivory_json_20191113_041232_depth"
set NUMBER_PEOPLE_MAX=3

set INPUT_VIDEO="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\goast\goast.mp4"
set OPENPOSE_JSON="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\goast\goast_20191112_021708\goast_json"
set PAST_DEPTH_PATH="E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\goast\goast_20191112_021708\goast_json_20191113_013115_depth"
set NUMBER_PEOPLE_MAX=5


set DEPTH_INTERVAL=20
set FRAME_END=1000
set REVERSE_SPECIFIC_LIST=
set ORDER_SPECIFIC_LIST=
set AVI_OUTPUT=yes
set VERBOSE=2

rem ---  python 実行
python predict_video.py --past_depth_path "%PAST_DEPTH_PATH%" --video_path %INPUT_VIDEO% --json_path %OPENPOSE_JSON% --interval %DEPTH_INTERVAL% --reverse_specific "%REVERSE_SPECIFIC_LIST%" --order_specific "%ORDER_SPECIFIC_LIST%" --avi_output %AVI_OUTPUT% --verbose %VERBOSE% --number_people_max %NUMBER_PEOPLE_MAX% --end_frame_no %FRAME_END% --input single_view --batchSize 1



