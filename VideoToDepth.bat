@echo off
rem --- 
rem ---  映像データから深度推定を行う
rem --- 

rem ---  カレントディレクトリを実行先に変更
cd /d %~dp0

rem ---  入力対象映像ファイルパス
echo Openposeで解析した映像のファイルのフルパスを入力して下さい。
echo この設定は半角英数字のみ設定可能で、必須項目です。
set INPUT_VIDEO=
set /P INPUT_VIDEO=■解析対象映像ファイルパス: 
rem echo INPUT_VIDEO：%INPUT_VIDEO%

IF /I "%INPUT_VIDEO%" EQU "" (
    ECHO 解析対象映像ファイルパスが設定されていないため、処理を中断します。
    EXIT /B
)

rem ---  解析結果JSONディレクトリパス
echo Openposeの解析結果のJSONディレクトリのフルパスを入力して下さい。({動画名}_json)
echo この設定は半角英数字のみ設定可能で、必須項目です。
set OPENPOSE_JSON=
set /P OPENPOSE_JSON=■解析結果JSONディレクトリパス: 
rem echo OPENPOSE_JSON：%OPENPOSE_JSON%

IF /I "%OPENPOSE_JSON%" EQU "" (
    ECHO 解析結果JSONディレクトリパスが設定されていないため、処理を中断します。
    EXIT /B
)

rem ---  深度推定結果ディレクトリパス
echo 既に深度推定を行った事がある場合、深度推定結果ディレクトリのフルパスを入力して下さい。({動画名}_json_{日時}_depth)
echo この設定は半角英数字のみ設定可能です。
echo ディレクトリ内に深度ファイルがある場合、深度推定結果をそのファイルから読み取ります。
set PAST_DEPTH_PATH=
set /P PAST_DEPTH_PATH=■深度推定結果ディレクトリパス: 

rem ---  深度推定間隔
rem echo --------------
set DEPTH_INTERVAL=20
rem echo 深度推定を行うフレームの間隔を数値で入力して下さい。
rem echo 値が小さいほど、細かく深度推定を行います。（その分、時間がかかります）
rem echo 何も入力せず、ENTERを押下した場合、「%DEPTH_INTERVAL%」間隔で処理します。
rem set /P DEPTH_INTERVAL="■深度推定間隔: "

rem ---  映像に映っている最大人数

echo --------------
echo 映像に映っている最大人数を入力して下さい。
echo 何も入力せず、ENTERを押下した場合、1人分の解析になります。
echo 複数人数が同程度の大きさで映っている映像で1人だけ指定した場合、解析対象が飛ぶ場合があります。
set NUMBER_PEOPLE_MAX=1
set /P NUMBER_PEOPLE_MAX="■映像に映っている最大人数: "

rem ---  解析を終了するフレーム

echo --------------
echo 解析を終了するフレームNoを入力して下さい。(0始まり)
echo 反転や順番を調整する際に、最後まで出力せずとも処理を終了して結果を見ることができます。
echo 何も入力せず、ENTERを押下した場合、最後まで解析します。
set FRAME_END=-1
set /P FRAME_END="■解析終了フレームNo: "

rem ---  反転指定リスト
echo --------------
set REVERSE_SPECIFIC_LIST=
echo Openposeが誤認識して反転しているフレーム番号(0始まり)、人物INDEX順番、反転の内容を指定してください。
echo Openposeが0F目で認識した順番に0, 1, とINDEXが割り当てられます。
echo フォーマット：［＜フレーム番号＞:反転を指定したい人物INDEX,<反転内容>］
echo <反転内容>: R: 全身反転, U: 上半身反転, L: 下半身反転, N: 反転なし
echo 例）[10:1,R]　…　10F目の1番目の人物を全身反転します。
echo message.logに上記フォーマットで、反転出力した場合にその内容を出力しているので、それを参考にしてください。
echo [10:1,R][30:0,U]のように、カッコ単位で複数件指定可能です。
set /P REVERSE_SPECIFIC_LIST="■反転指定リスト: "

rem ---  順番指定リスト
echo --------------
set ORDER_SPECIFIC_LIST=
echo 複数人数トレースで、交差後の人物INDEX順番を指定してください。1人トレースの場合は空欄のままで大丈夫です。
echo Openposeが0F目で認識した順番に0, 1, とINDEXが割り当てられます。
echo フォーマット：［＜フレーム番号＞:0番目に推定された人物のインデックス,1番目に推定された人物のインデックス, …］
echo 例）[10:1,0]　…　10F目は、左から1番目の人物、0番目の人物の順番に並べ替えます。
echo message.logに上記フォーマットで、どのような順番で出力したかを残しているので、それを参考にしてください。
echo [10:1,0][30:0,1]のように、カッコ単位で複数件指定可能です。
echo また、output_XXX.aviでは、推定された順番に人物に色が割り当てられています。体の右半分は赤、左半分は以下の色になります。
echo 0:緑, 1:青, 2:白, 3:黄, 4:桃, 5:水色, 6:濃緑, 7:濃青, 8:灰色, 9:濃黄, 10:濃桃, 11:濃水色
set /P ORDER_SPECIFIC_LIST="■順番指定リスト: "

rem ---  MMD用AVI出力
echo --------------
echo MMD用AVIを出すか、yes か no を入力して下さい。
echo MMD用AVIは、Openposeの結果に、人物INDEX別情報を乗せて、サイズ小さめで出力します。
echo 何も入力せず、ENTERを押下した場合、MMD用AVIを出力します。
set AVI_OUTPUT=yes
set /P AVI_OUTPUT="■MMD用AVI[yes/no]: "

rem ---  詳細ログ有無
echo --------------
echo 詳細なログを出すか、yes か no を入力して下さい。
echo 何も入力せず、ENTERを押下した場合、通常ログと深度推定GIFを出力します。
echo warn と指定すると、アニメーションGIFも出力しません。（その分早いです）
set VERBOSE=2
set IS_DEBUG=no
set /P IS_DEBUG="■詳細ログ[yes/no/warn]: "

IF /I "%IS_DEBUG%" EQU "yes" (
    set VERBOSE=3
)

IF /I "%IS_DEBUG%" EQU "warn" (
    set VERBOSE=1
)

rem ---  python 実行
python predict_video.py --past_depth_path "%PAST_DEPTH_PATH%" --video_path %INPUT_VIDEO% --json_path %OPENPOSE_JSON% --interval %DEPTH_INTERVAL% --reverse_specific "%REVERSE_SPECIFIC_LIST%" --order_specific "%ORDER_SPECIFIC_LIST%" --avi_output %AVI_OUTPUT% --verbose %VERBOSE% --number_people_max %NUMBER_PEOPLE_MAX% --end_frame_no %FRAME_END%  --input single_view --batchSize 1



