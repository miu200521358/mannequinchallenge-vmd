@echo off
rem --- 
rem ---  Perform depth estimation from video data
rem --- 

rem ---  Change the current directory to the execution destination
cd /d %~dp0

rem ---  Input target video file path
echo Please enter the full path of the file of the video analyzed with Openpose.
echo This setting is available only for half size alphanumeric characters, it is a required item.
set INPUT_VIDEO=
set /P INPUT_VIDEO=** Movie file path to be analyzed: 
rem echo INPUT_VIDEO：%INPUT_VIDEO%

IF /I "%INPUT_VIDEO%" EQU "" (
    ECHO Processing is suspended because the analysis target video file path is not set.
    EXIT /B
)

rem ---  Analysis result JSON directory path
echo Please enter the full path of JSON directory of analysis result of Openpose.
echo This setting is available only for half size alphanumeric characters, it is a required item.
set OPENPOSE_JSON=
set /P OPENPOSE_JSON=** Analysis result JSON directory path: 
rem echo OPENPOSE_JSON：%OPENPOSE_JSON%

IF /I "%OPENPOSE_JSON%" EQU "" (
    ECHO Analysis result Since JSON directory path is not set, processing is interrupted.
    EXIT /B
)

rem ---  深度推定結果ディレクトリパス
echo If you have already done depth estimation, please enter the full path of the depth estimation result directory. ({Video name}_json_{datetime}_depth)
echo This setting is available only for half size alphanumeric characters.
echo If there is a depth file in the directory, the depth estimation result is read from that file.
set PAST_DEPTH_PATH=
set /P PAST_DEPTH_PATH=** Depth estimation result directory path: 

rem ---rem ---  Depth estimation interval
rem ---echo --------------
set DEPTH_INTERVAL=20
rem ---echo Please enter the interval of the frame to be estimated depth.
rem ---echo The smaller the value, the finer the depth estimation. (It takes time to do so)
rem ---echo If nothing is entered and ENTER is pressed, processing is done at the interval of "%DEPTH_INTERVAL%".
rem ---set /P DEPTH_INTERVAL="** Depth estimation interval: "

rem ---  Maximum number of people in the image

echo --------------
echo Please enter the maximum number of people shown in the image.
echo If you do not enter anything and press ENTER, it will be analysis for one person.
echo If you specify only one person in the image of which the number of people is the same size, the analysis subject may jump.
set NUMBER_PEOPLE_MAX=1
set /P NUMBER_PEOPLE_MAX="** Maximum number of people shown in the image:"

rem ---  Frame to end analysis

echo --------------
echo Please enter the frame number to end analysis. (0 beginning)
echo When you adjust the reverse or order, 
echo you can finish the process and see the result without outputting to the end.
echo If nothing is input and ENTER is pressed, analysis is performed to the end.
set FRAME_END=-1
set /P FRAME_END="** Analysis end frame number: "

rem ---  反転指定リスト
echo --------------
set REVERSE_SPECIFIC_LIST=
echo Specify the frame number (0 starting) that is inverted by Openpose by mistake, the person INDEX order, and the contents of the inversion.
echo In the order that Openpose recognizes at 0F, INDEX is assigned as 0, 1, ....
echo Format: [{frame number}: Person who wants to specify reverse INDEX, {reverse content}]
echo {reverse content}: R: Whole body inversion, U: Upper body inversion, L: Lower body inversion, N: No inversion
echo 例）[10:1,R]　…　The whole person flips the first person in the 10th frame.
echo Since the contents are output in the above format in message.log when inverted output, please refer to that.
echo As in [10:1,R][30:0,U], multiple items can be specified in parentheses.
set /P REVERSE_SPECIFIC_LIST="** Reverse specification list: "

rem ---  順番指定リスト
echo --------------
set ORDER_SPECIFIC_LIST=
echo In the multi-person trace, please specify the person INDEX order after crossing.
echo In the case of a one-person trace, it is OK to leave it blank.
echo In the order that Openpose recognizes at 0F, INDEX is assigned as 0, 1, ....
echo Format: [{frame number}: index of first estimated person, index of first estimated person, ...]
echo 例）[10:1,0]　…　The order of the 10th frame is rearranged in the order of the first person from the left and the zeroth person.
echo The order in which messages are output in message.log is left in the above format, so please refer to it.
echo As in [10:1,0][30:0,1], multiple items can be specified in parentheses.
echo Also, in output_XXX.avi, colors are assigned to people in the estimated order. The right half of the body is red and the left half is the following color.
echo 0: green, 1: blue, 2: white, 3: yellow, 4: peach, 5: light blue, 6: dark green, 7: dark blue, 8: gray, 9: dark yellow, 10: dark peach, 11: dark light blue
set /P ORDER_SPECIFIC_LIST="** Ordered list: "

rem ---  MMD用AVI出力
echo --------------
echo Please output AVI for MMD or enter yes or no.
echo AVI for MMD outputs information in smaller size by putting person-specific index information on the Openpose result.
echo When nothing is input and ENTER is pressed, AVI for MMD is output.
set AVI_OUTPUT=yes
set /P AVI_OUTPUT="** AVI for MMD[yes/no]: "

rem ---  Presence of detailed log

echo --------------
echo Please output detailed logs or enter yes or no.
echo If nothing is entered and ENTER is pressed, normal log and depth estimation GIF are output.
echo If warn is specified, animation GIF is not output. (That is earlier)
set VERBOSE=2
set IS_DEBUG=no
set /P IS_DEBUG="** Detailed log[yes/no/warn]: "

IF /I "%IS_DEBUG%" EQU "yes" (
    set VERBOSE=3
)

IF /I "%IS_DEBUG%" EQU "warn" (
    set VERBOSE=1
)

rem ---  python 実行
python tensorflow/predict_video.py --model_path tensorflow/data/NYU_FCRN.ckpt --centerz_model_path tensorflow/data2/centerz-depth.ckpt --past_depth_path "%PAST_DEPTH_PATH%" --video_path %INPUT_VIDEO% --json_path %OPENPOSE_JSON% --interval %DEPTH_INTERVAL% --reverse_specific "%REVERSE_SPECIFIC_LIST%" --order_specific "%ORDER_SPECIFIC_LIST%" --avi_output %AVI_OUTPUT% --verbose %VERBOSE% --number_people_max %NUMBER_PEOPLE_MAX% --end_frame_no %FRAME_END% --input single_view --batchSize 1


