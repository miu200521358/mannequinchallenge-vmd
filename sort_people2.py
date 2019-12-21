import os
import numpy as np
import logging
import cv2
import shutil
import json
import copy
import sys
import re
from matplotlib import pyplot as plt
import imageio
from collections import Counter

# ファイル出力ログ用
file_logger = logging.getLogger("message").getChild(__name__)
logger = logging.getLogger("__main__").getChild(__name__)

level = {0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG}

# 人物ソート
def sort(cnt, _display_idx, _iidx, sorted_idxs, now_str, interval, subdir, json_path, json_size, number_people_max, reverse_specific_dict, order_specific_dict, start_json_name, start_frame, pred_multi_ary, pred_multi_z_ary, pred_multi_xy_ary, pred_multi_frame_ary, frame_imgs, max_conf_ary, max_conf_color_ary, org_width, org_height, past_data, past_depths, past_depths_z, png_lib, verbose):

    # 該当シーンのJSONデータを読み込む
    file_name = re.sub(r'\d{12}', "{0:012d}".format(cnt), start_json_name)
    _file = os.path.join(json_path, file_name)
    try:
        data = json.load(open(_file))
        # 過去データ上書きしないデータも保持
        org_data = json.load(open(_file))
    except Exception as e:
        logger.warning("JSON読み込み失敗のため、空データ読み込み, %s %s", _file, e)
        data = json.load(open("json/all_empty_keypoints.json"))
        org_data = json.load(open("json/all_empty_keypoints.json"))

    for i in range(len(data["people"]), number_people_max):
        # 足りない分は空データを埋める
        data["people"].append(json.load(open("json/one_keypoints.json")))
        org_data["people"].append(json.load(open("json/one_keypoints.json")))

    logger.info("人体別処理: iidx: %s file: %s ------", _iidx, file_name)

    # 並べ直したindex用配列反転有無
    is_all_reverses = [False for x in range(number_people_max)]
    # 並べ直したindex用配列反転有無(上半身のみ)
    is_upper_reverses = [False for x in range(number_people_max)]
    # 並べ直したindex用配列反転有無(下半身のみ)
    is_lower_reverses = [False for x in range(number_people_max)]
    # 現在反転中か否か(上半身)
    is_now_upper_reversed = [False for x in range(number_people_max)]
    # 現在反転中か否か(下半身)
    is_now_lower_reversed = [False for x in range(number_people_max)]

    # インデックス並び替え -------------------------
    # 開始時
    if _iidx == 0:
        # 前回のXYを保持
        past_data = data["people"]
        # 前回の深度を保持
        past_depths = pred_multi_ary[0]
        # 前回の深度(センターZ)を保持
        past_depths_z = pred_multi_z_ary[0]

        # 最初は左から順番に0番目,1番目と並べる
        # FIXME 初期表示時の左から順番に並べたい
        # first_sorted_idxs = sort_first_idxs(data["people"])
        # logger.info("first_sorted_idxs: %s", first_sorted_idxs)

        for pidx in range(number_people_max):
            # 最初はインデックスの通りに並べる
            sorted_idxs[_iidx][pidx] = pidx

        past_depth_idx = -1
        next_depth_idx = -1
    else:
        # 前回の深度
        past_depth_idx = _iidx - (_iidx % interval)
        # 次回の深度
        next_depth_idx = _iidx + interval - (_iidx % interval)
        if next_depth_idx >= json_size - start_frame:
            # 最後は同じ値をnextとして見る
            next_depth_idx = json_size - start_frame - 1

        if _iidx in order_specific_dict:
            # 順番指定リストに該当フレームがある場合
            for key_idx, person_idx in enumerate(order_specific_dict[_iidx]):
                # Openposeのデータの順番に応じたソート順を指定する
                sorted_idxs[_iidx][key_idx] = person_idx
                # 反転はさせない
                is_all_reverses[key_idx] = False
                is_upper_reverses[key_idx] = False
                is_lower_reverses[key_idx] = False
                # logger.info("_iidx: %s, _display_idx: %s, key_idx: %s, person_idx: %s", _iidx, _display_idx, key_idx, person_idx )

            file_logger.warning("※※{0:05d}F目 順番指定あり {2}".format(_iidx, _display_idx, order_specific_dict[_iidx]))
            # logger.info("_iidx: %s, _display_idx: %s, sorted_idxs[_iidx]: %s", _iidx, _display_idx, sorted_idxs[_iidx] )
        else:
            # 前回のXYと深度から近いindexを算出
            sorted_idxs[_iidx], is_all_reverses, is_upper_reverses, is_lower_reverses = calc_nearest_idxs(
                sorted_idxs[_iidx - 1], past_data, data["people"], pred_multi_ary[past_depth_idx], pred_multi_ary[next_depth_idx], max_conf_ary, max_conf_color_ary, frame_imgs[(_iidx - 1) % interval], frame_imgs[_iidx % interval])

    logger.info("＊＊_iidx: %s(%s), past_depth_idx: %s, next_depth_idx: %s, sorted_idxs: %s, all: %s, upper: %s, lower: %s", _iidx, _display_idx,
                past_depth_idx, next_depth_idx, sorted_idxs[_iidx], is_all_reverses, is_upper_reverses, is_lower_reverses)

    # 現在データ
    now_data = [[] for x in range(number_people_max)]
    # 過去を引き継いだ現在データ
    all_now_data = [[] for x in range(number_people_max)]
    # 過去を引き継いだ現在深度
    all_now_depths = [[] for x in range(number_people_max)]
    # 過去を引き継いだ現在深度のxy
    all_now_depths_xy = [[] for x in range(number_people_max)]
    # 過去を引き継いだ現在センターZ
    all_now_depths_z = [[] for x in range(number_people_max)]

    # インデックス出力 ------------------------------
    for pidx, sidx in enumerate(sorted_idxs[_iidx]):
        logger.debug("reverse_specific_dict _iidx: %s, pidx: %s, in: %s", _iidx, pidx, (_iidx in reverse_specific_dict and pidx in reverse_specific_dict[_iidx]))
        if _iidx in reverse_specific_dict and pidx in reverse_specific_dict[_iidx]:
            if 'R' in reverse_specific_dict[_iidx][pidx]:
                logger.debug("反転指定対象フレーム【R】: %s(%s) rsd: %s", _iidx, _display_idx, reverse_specific_dict[_iidx][pidx])

                # 全身反転
                is_all_reverses[pidx] = True
                is_upper_reverses[pidx] = False
                is_lower_reverses[pidx] = False
            elif 'U' in reverse_specific_dict[_iidx][pidx]:
                logger.debug("反転指定対象フレーム【U】: %s(%s) rsd: %s", _iidx, _display_idx, reverse_specific_dict[_iidx][pidx])

                # 上半身反転
                is_all_reverses[pidx] = False
                is_upper_reverses[pidx] = True
                is_lower_reverses[pidx] = False
            elif 'L' in reverse_specific_dict[_iidx][pidx]:
                logger.debug("反転指定対象フレーム【L】: %s(%s) rsd: %s", _iidx, _display_idx, reverse_specific_dict[_iidx][pidx])

                # 下半身反転
                is_all_reverses[pidx] = False
                is_upper_reverses[pidx] = False
                is_lower_reverses[pidx] = True
            elif 'N' in reverse_specific_dict[_iidx][pidx]:
                logger.debug("反転指定対象フレーム【N】: %s(%s) rsd: %s", _iidx, _display_idx, reverse_specific_dict[_iidx][pidx])

                # 反転なし
                is_all_reverses[pidx] = False
                is_upper_reverses[pidx] = False
                is_lower_reverses[pidx] = False
            else:
                logger.warning("反転指定対象フレーム【EMPTY】: %s(%s) rsd: %s", _iidx, _display_idx, reverse_specific_dict[_iidx][pidx])

                # 反転なし
                is_all_reverses[pidx] = False
                is_upper_reverses[pidx] = False
                is_lower_reverses[pidx] = False
            
            if is_all_reverses[pidx]:
                # 現在の反転状況(全身反転)
                is_now_upper_reversed[pidx] = True
                is_now_lower_reversed[pidx] = True
            else:
                # 現在の反転状況(上下別々反転)
                is_now_upper_reversed[pidx] = is_upper_reverses[pidx]
                is_now_lower_reversed[pidx] = is_lower_reverses[pidx]

            logger.debug("_iidx: %s(%s), upper: %s, lower: %s", _iidx, _display_idx, is_now_upper_reversed, is_now_lower_reversed)
        else:
            # 現在データ(sidxで振り分け済み)
            now_sidx_data = data["people"][sidx]["pose_keypoints_2d"]

            if _iidx > 0:
                # とりあえず何らかのデータがある場合
                # 過去データ
                past_pidx_data = past_data[sorted_idxs[_iidx - 1][pidx]]["pose_keypoints_2d"]

                for o in range(0,len(now_sidx_data),3):
                    oidx = int(o/3)
                    if now_sidx_data[o] == now_sidx_data[o+1] == 0 and oidx in [1,2,3,4,5,6,7,8,9,10,11,12,13]:
                        logger.debug("過去PU: pidx: %s, sidx:%s, o: %s, ns: %s, pp: %s, np: %s, ps: %s", pidx, sidx, oidx, now_sidx_data[o], past_pidx_data[o], data["people"][pidx]["pose_keypoints_2d"][o], past_data[sidx]["pose_keypoints_2d"][o])
                        logger.debug("sidx: %s, now_sidx_data: %s", sidx, now_sidx_data)
                        # XもYも0の場合、過去から引っ張ってくる
                        # 反転対応済みのINDEXに設定する
                        now_sidx_data[o] = past_pidx_data[o]
                        now_sidx_data[o+1] = past_pidx_data[o+1]
                        now_sidx_data[o+2] = past_pidx_data[o+2] - 0.1

                logger.debug("反転再チェック: %s(%s) ----------------------------", _iidx, _display_idx)

                # 前回のXYと深度から近いindexを算出
                # 埋まってない部分を補完して、改めて反転再算出
                # 既に並べ終わってるので、少し底上げして厳しめにチェックする
                _, is_retake_all_reverses, is_retake_upper_reverses, is_retake_lower_reverses = \
                    calc_nearest_idxs([0], [past_data[pidx]], [data["people"][sidx]], [pred_multi_ary[past_depth_idx][sidx]], [pred_multi_ary[next_depth_idx][sidx]], None, max_conf_color_ary, frame_imgs[(_iidx - 1) % interval], frame_imgs[_iidx % interval], 0.03)

                is_all_reverses[pidx] = is_retake_all_reverses[0]
                is_upper_reverses[pidx] = is_retake_upper_reverses[0]
                is_lower_reverses[pidx] = is_retake_lower_reverses[0]
                
                logger.debug("＊＊反転再チェック: _iidx: %s, pidx: %s, all: %s, upper: %s, lower: %s", _iidx, pidx, is_all_reverses[pidx], is_upper_reverses[pidx], is_lower_reverses[pidx])

                if is_all_reverses[pidx]:
                    logger.debug("全身判定 true")
                    # 全身反転の場合
                    if is_upper_reverses[pidx] != is_lower_reverses[pidx]:
                        logger.debug("全身判定 上半身・下半身違いでクリア")
                        # 上半身と下半身で反転が違う場合、反転クリア
                        is_now_upper_reversed[pidx] = False
                        is_now_lower_reversed[pidx] = False
                    else:
                        # 反転状況が同じ場合は、反転採用
                        is_now_upper_reversed[pidx] = True
                        is_now_lower_reversed[pidx] = True
                else:
                    is_now_upper_reversed[pidx] = is_upper_reverses[pidx]
                    is_now_lower_reversed[pidx] = is_lower_reverses[pidx]
            else:
                # 反転対象外の場合、クリア
                is_now_upper_reversed[pidx] = False
                is_now_lower_reversed[pidx] = False
                
            logger.info("＊＊反転確定：pidx: %s, is_now_upper_reversed: %s, is_now_lower_reversed: %s", pidx, is_now_upper_reversed[pidx], is_now_lower_reversed[pidx])

            # # トレース失敗の場合、クリア
            # if (is_all_reverse == False and (is_upper_reverse or (is_upper_reverse == False and is_now_upper_reversed[pidx] ))) and (targetdata[2*3] == 0 or targetdata[3*3] == 0 or targetdata[5*3] == 0 or targetdata[6*3] == 0) :
            #     logger.debug("上半身ひじまでのトレース失敗のため、上半身反転フラグクリア %s(%s) data: %s", _iidx, _display_idx, targetdata)
            #     is_upper_reverses[pidx] = False
            #     is_now_upper_reversed[pidx] = False

            # if (is_all_reverse == False or (is_lower_reverse or (is_lower_reverse == False and is_now_lower_reversed[pidx] ))) and (targetdata[8*3] == 0 or targetdata[9*3] == 0 or targetdata[11*3] == 0 or targetdata[12*3] == 0) :
            #     logger.debug("下半身ひざまでのトレース失敗のため、下半身反転フラグクリア %s(%s) data: %s", _iidx, _display_idx, targetdata)
            #     is_lower_reverses[pidx] = False
            #     is_now_lower_reversed[pidx] = False

            logger.debug("_iidx: %s(%s), sidx: %s, pidx: %s, upper: %s, lower: %s", _iidx, _display_idx, sidx, pidx, is_now_upper_reversed[pidx], is_now_lower_reversed[pidx])

            logger.debug("is_now_upper_reversed: %s, is_now_lower_reversed: %s", is_now_upper_reversed, is_now_lower_reversed)

    # 反転判定が終わった後、出力処理
    for pidx, sidx in enumerate(sorted_idxs[_iidx]):
        # 指定ありの場合、メッセージ追加
        reverse_specific_str = ""
        if _iidx in reverse_specific_dict and pidx in reverse_specific_dict[_iidx]:
            reverse_specific_str = "【指定】"

        if is_now_upper_reversed[pidx] and is_now_lower_reversed[pidx]:
            file_logger.warning("※※{0:05d}F目 {2}番目の人物、全身反転 [{0}:{2},R]{3}".format( _iidx, _display_idx, pidx, reverse_specific_str))
        elif is_now_upper_reversed[pidx] and is_now_lower_reversed[pidx] == False :
            file_logger.warning("※※{0:05d}F目 {2}番目の人物、上半身反転 [{0}:{2},U]{3}".format( _iidx, _display_idx, pidx, reverse_specific_str))
        elif is_now_upper_reversed[pidx] == False and is_now_lower_reversed[pidx]:
            file_logger.warning("※※{0:05d}F目 {2}番目の人物、下半身反転 [{0}:{2},L]{3}".format( _iidx, _display_idx, pidx, reverse_specific_str))
        else:
            if len(reverse_specific_str) > 0:
                file_logger.warning("※※{0:05d}F目 {2}番目の人物、反転なし [{0}:{2},N]{3}".format( _iidx, _display_idx, pidx, reverse_specific_str))

        # 一旦空データを読む
        outputdata = json.load(open("json/empty_keypoints.json"))
        # 一旦空データを読む
        all_outputdata = json.load(open("json/empty_keypoints.json"))

        # 過去の上書きがない元データ
        org_sidx_data = org_data["people"][sidx]["pose_keypoints_2d"]

        # 出力用深度（とりあえず全部0）
        outputdepths = [0 for x in range(18)]
        # 出力用深度センターZ（とりあえず全部0）
        outputdepths_z = [0 for x in range(18)]
        # 出力用深度XY(X,Yの配列が入る)
        outputdepths_xy = [[] for x in range(18)]

        for o in range(0,len(outputdata["people"][0]["pose_keypoints_2d"]),3):
            # デフォルトのXINDEX
            oidx = int(o/3)

            if is_now_upper_reversed[pidx] and is_now_lower_reversed[pidx]:
                oidx = OPENPOSE_REVERSE_ALL[oidx]
            elif is_now_upper_reversed[pidx] and is_now_lower_reversed[pidx] == False:
                # 反転している場合、反転INDEX(上半身)
                oidx = OPENPOSE_REVERSE_UPPER[oidx]
            elif is_now_upper_reversed[pidx] == False and is_now_lower_reversed[pidx]:
                # 反転している場合、反転INDEX(下半身)
                oidx = OPENPOSE_REVERSE_LOWER[oidx]
            
            # 出力データはオリジナルデータのみコピー
            outputdata["people"][0]["pose_keypoints_2d"][o] = org_sidx_data[oidx*3]
            outputdata["people"][0]["pose_keypoints_2d"][o+1] = org_sidx_data[oidx*3+1]
            outputdata["people"][0]["pose_keypoints_2d"][o+2] = org_sidx_data[oidx*3+2]

            # 過去引継データもとりあえずオリジナルデータコピー
            all_outputdata["people"][0]["pose_keypoints_2d"][o] = org_sidx_data[oidx*3]
            all_outputdata["people"][0]["pose_keypoints_2d"][o+1] = org_sidx_data[oidx*3+1]
            all_outputdata["people"][0]["pose_keypoints_2d"][o+2] = org_sidx_data[oidx*3+2]

            if _iidx % interval == 0:
                # 深度元データ
                outputdepths[oidx] = pred_multi_ary[_iidx][sidx][oidx]
                outputdepths_z[oidx] = pred_multi_z_ary[_iidx][sidx][oidx]
                # logger.info("_iidx: %s, sidx: %s, oidx: %s, len(pred_multi_xy_ary[_iidx]): %s, len(pred_multi_xy_ary[_iidx][sidx]: %s", _iidx, sidx, oidx, len(pred_multi_xy_ary[_iidx]), len(pred_multi_xy_ary[_iidx][sidx]))
                if len(pred_multi_xy_ary[_iidx][sidx]) > 0:
                    outputdepths_xy[oidx] = pred_multi_xy_ary[_iidx][sidx][oidx]

        logger.debug("outputdata %s", outputdata["people"][0]["pose_keypoints_2d"])

        # 出力順番順に並べなおしてリストに設定
        now_data[sidx] = outputdata
        all_now_data[sidx] = all_outputdata

        if _iidx % interval == 0:
            all_now_depths[sidx] = outputdepths
            all_now_depths_z[sidx] = outputdepths_z
            all_now_depths_xy[sidx] = outputdepths_xy

    # 詰め直し
    now_sorted_datas = {}
    now_sorted_all_datas = {}
    now_sorted_all_depths = {}
    now_sorted_all_depths_xy = {}
    now_sorted_all_depths_z = {}
    for pidx, sidx in enumerate(sorted_idxs[_iidx]):
        # 現在データ
        now_sorted_datas[sidx] = now_data[pidx]["people"][0]
        # 現在データ（過去引継）
        now_sorted_all_datas[sidx] = all_now_data[pidx]["people"][0]
        # 現在深度
        now_sorted_all_depths[sidx] = all_now_depths[pidx]
        # 現在深度XY
        now_sorted_all_depths_xy[sidx] = all_now_depths_xy[pidx]
        # 現在深度センターZ
        now_sorted_all_depths_z[sidx] = all_now_depths_z[pidx]

    # 過去データからの引継
    for pidx, sidx in enumerate(sorted_idxs[_iidx]):
        now_sorted_data = now_sorted_datas[pidx]["pose_keypoints_2d"]
        now_sorted_all_data = now_sorted_all_datas[pidx]["pose_keypoints_2d"]
        past_sorted_data = past_data[pidx]["pose_keypoints_2d"]

        logger.debug("＊＊＊ iidx: %s(%s) pidx: %s, sidx: %s, np: %s, pp: %s", _iidx, _display_idx, pidx, sidx, now_sorted_all_data[1*3], past_sorted_data[1*3])

        for o in range(0,len(now_sorted_all_data),3):
            if now_sorted_all_data[o] == 0 and now_sorted_all_data[o+1] == 0 and int(o/3) in [1,2,3,4,5,6,7,8,9,11,12,13,16,17]:
                # 値がない場合、過去引継ぎデータは過去データをコピーする
                logger.debug("＊＊＊過去データ引継 iidx: %s(%s) pidx: %s, sidx: %s, np: %s, pp: %s", _iidx, _display_idx, pidx, sidx, now_sorted_all_data[o], past_sorted_data[o])
                now_sorted_all_data[o] = past_sorted_data[o]
                now_sorted_all_data[o+1] = past_sorted_data[o+1]
                now_sorted_all_data[o+2] = 0.3

            if now_sorted_all_data[o] > org_width or now_sorted_all_data[o] < 0 \
                or now_sorted_all_data[o+1] > org_height or now_sorted_all_data[o+1] < 0 :
                # 画像範囲外のデータが取れた場合、とりあえず0を入れ直す
                now_sorted_data[o] = 0
                now_sorted_data[o+1] = 0
                now_sorted_data[o+2] = 0

                now_sorted_all_data[o] = 0
                now_sorted_all_data[o+1] = 0
                now_sorted_all_data[o+2] = 0

        # 深度が0の場合、過去深度をコピーする
        if _iidx % interval == 0:
            now_sorted_one_depths = now_sorted_all_depths[pidx]
            now_sorted_one_depths_z = now_sorted_all_depths_z[pidx]
            past_sorted_depths = past_depths[pidx]
            past_sorted_depths_z = past_depths_z[pidx]

            for didx, d in enumerate(now_sorted_one_depths):
                if d == 0:
                    logger.debug("depth copy iidx: %s(%s) pidx: %s, sidx: %s, p: %s", _iidx, _display_idx, pidx, sidx, past_sorted_depths)
                    now_sorted_one_depths[didx] = past_sorted_depths[didx]

            for didx, d in enumerate(now_sorted_one_depths_z):
                if d == 0:
                    logger.debug("depth copy iidx: %s(%s) pidx: %s, sidx: %s, p: %s", _iidx, _display_idx, pidx, sidx, past_sorted_depths_z)
                    now_sorted_one_depths_z[didx] = past_sorted_depths_z[didx]

        # if _iidx > 0 and _iidx + 1 < len(openpose_2d):
        #     # まだ次フレームがある場合、足異常チェック
        #     _next_file = os.path.join(json_path, openpose_filenames[_op_idx+1])
        #     if not os.path.isfile(_next_file): raise Exception("No file found!!, {0}".format(_next_file))
        #     try:
        #         next_data = json.load(open(_next_file))
        #     except Exception as e:
        #         logger.warning("JSON読み込み失敗のため、空データ読み込み, %s %s", _next_file, e)
        #         next_data = json.load(open("json/all_empty_keypoints.json"))

        #     for i in range(len(next_data["people"]), number_people_max):
        #         # 足りない分は空データを埋める
        #         next_data["people"].append(json.load(open("json/one_keypoints.json")))

        #     # 足異常チェック(リセットは行わない)
        #     is_result_oneside, is_result_crosses = calc_leg_irregular([0], [past_data[pidx]], [now_sorted_datas[pidx]], next_data["people"], 1, False)
        #     logger.debug("足異常再チェック %s, %s, 片寄せ: %s, 交差: %s", _iidx, pidx, is_result_oneside, is_result_crosses)

        #     if True in is_result_oneside or True in is_result_crosses:
        #         if True in is_result_oneside:
        #             # 片寄せの可能性がある場合、前回データをコピー
        #             file_logger.warning("※※{0:05d}F目 {2}番目の人物、片寄せ可能性あり".format( _iidx, _display_idx, pidx))

        #         if True in is_result_crosses:
        #             # 交差の可能性がある場合、前回データをコピー
        #             file_logger.warning("※※{0:05d}F目 {2}番目の人物、交差可能性あり".format( _iidx,  _display_idx, pidx))

        #         for _lval in [8,9,10,11,12,13]:
        #             # logger.info("足異常:過去データコピー iidx: %s(%s) pidx: %s, sidx: %s, nn: %s, pn: %s, now: %s, past: %s", _iidx, _display_idx, pidx, sidx, org_sidx_data[1*3], past_pidx_data[1*3], org_sidx_data, past_pidx_data)

        #             # 信頼度は半分
        #             conf = past_pidx_data[_lval*3+2]/2

        #             # 出力用データ
        #             now_sorted_data[_lval*3] = past_pidx_data[_lval*3]
        #             now_sorted_data[_lval*3+1] = past_pidx_data[_lval*3+1]
        #             now_sorted_data[_lval*3+2] = conf if 0 < conf < 0.3 else 0.3

        #             # 過去引継データ
        #             now_sorted_all_datas[_lval*3] = past_pidx_data[_lval*3]
        #             now_sorted_all_datas[_lval*3+1] = past_pidx_data[_lval*3+1]
        #             now_sorted_all_datas[_lval*3+2] = conf if 0 < conf < 0.3 else 0.3

    # 首の位置が一番よく取れてるので、首の位置を出力する
    display_nose_pos = {}
    for pidx, sidx in enumerate(sorted_idxs[_iidx]):
        # データがある場合、そのデータ
        display_nose_pos[sidx] = [now_data[pidx]["people"][0]["pose_keypoints_2d"][1*3], now_data[pidx]["people"][0]["pose_keypoints_2d"][1*3+1]]

        # インデックス対応分のディレクトリ作成
        idx_path = '{0}/{1}_{3}_idx{2:02d}/json/{4}'.format(os.path.dirname(json_path), os.path.basename(json_path), sidx+1, now_str, file_name)
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        
        # 出力
        # json.dump(data, open(idx_path,'w'), indent=4)
        json.dump(now_data[pidx], open(idx_path,'w'), indent=4)

        if _iidx % interval == 0:
            # 深度データ
            depth_idx_path = '{0}/{1}_{3}_idx{2:02d}/depth.txt'.format(os.path.dirname(json_path), os.path.basename(json_path), pidx+1, now_str)
            # 追記モードで開く
            depthf = open(depth_idx_path, 'a')
            # 深度データを文字列化する
            # logger.debug("pred_multi_ary[_idx]: %s", pred_multi_ary[_idx])
            # logger.debug("pred_multi_ary[_idx][sidx]: %s", pred_multi_ary[_idx][sidx])
            # logger.info("all_now_depths pidx: %s, :%s", pidx, all_now_depths[pidx])
            pred_str_ary = [ str(x) for x in now_sorted_all_depths[pidx] ]

            # 一行分を追記
            depthf.write("{0}, {1}\n".format(_display_idx, ','.join(pred_str_ary)))
            depthf.close()

            # ------------------
            # 深度データ(センターZ)
            depthz_idx_path = '{0}/{1}_{3}_idx{2:02d}/depth_z.txt'.format(os.path.dirname(json_path), os.path.basename(json_path), pidx+1, now_str)
            # 追記モードで開く
            depthzf = open(depthz_idx_path, 'a')
            # 深度データを文字列化する
            # logger.debug("pred_multi_ary[_idx]: %s", pred_multi_ary[_idx])
            # logger.debug("pred_multi_ary[_idx][sidx]: %s", pred_multi_ary[_idx][sidx])
            # logger.info("all_now_depths pidx: %s, :%s", pidx, all_now_depths[pidx])
            pred_z_str_ary = [ str(x) for x in now_sorted_all_depths_z[pidx] ]

            # 一行分を追記
            depthzf.write("{0}, {1}\n".format(_display_idx, ','.join(pred_z_str_ary)))
            depthzf.close()

    # 深度画像保存 -----------------------
    if _iidx % interval == 0 and level[verbose] <= logging.INFO and len(pred_multi_frame_ary[_iidx]) > 0:
        # Plot result
        plt.cla()
        plt.clf()
        ii = plt.imshow(pred_multi_frame_ary[_iidx], interpolation='nearest')
        plt.colorbar(ii)

        # 散布図のようにして、出力に使ったポイントを明示
        DEPTH_COLOR = ["#33FF33", "#3333FF", "#FFFFFF", "#FFFF33", "#FF33FF", "#33FFFF", "#00FF00", "#0000FF", "#666666", "#FFFF00", "#FF00FF", "#00FFFF"]
        for pidx, sidx in enumerate(sorted_idxs[_iidx]):
            for pred_joint in now_sorted_all_depths_xy[pidx]:
                plt.scatter(pred_joint[0], pred_joint[1], s=5, c=DEPTH_COLOR[pidx])

        plotName = "{0}/depth_{1:012d}.png".format(subdir, cnt)
        plt.savefig(plotName)
        logger.debug("Save: {0}".format(plotName))

        png_lib.append(imageio.imread(plotName))

        # # アニメーションGIF用に区間分保持
        for mm in range(interval - 1):
            png_lib.append(imageio.imread(plotName))

        plt.close()

    file_logger.warning("＊＊{0:05d}F目の出力順番: [{0}:{2}], 位置: {3}".format(_iidx, _display_idx, ','.join(map(str, sorted_idxs[_iidx])), sorted(display_nose_pos.items()) ))

    # 今回全データを返す
    return all_now_data, all_now_depths, all_now_depths_z



# 0F目を左から順番に並べた人物INDEXを取得する
def sort_first_idxs(now_datas):
    most_common_idxs = []
    th = 0.3

    # 最終的な左からのINDEX
    result_nearest_idxs = [-1 for x in range(len(now_datas))]

    # 比較対象INDEX(最初は0(左端)を起点とする)
    target_x = [ 0 for x in range(int(len(now_datas[0]["pose_keypoints_2d"]))) ]

    # 人数分チェック
    for _idx in range(len(now_datas)):
        now_nearest_idxs = []
        # 関節位置Xでチェック
        for o in range(0,len(now_datas[0]["pose_keypoints_2d"]),3):
            is_target = True
            x_datas = []
            for _pnidx in range(len(now_datas)):
                if _pnidx not in result_nearest_idxs:
                    # 人物のうち、まだ左から並べられていない人物だけチェック対象とする
    
                    x_data = now_datas[_pnidx]["pose_keypoints_2d"][o]
                    x_conf = now_datas[_pnidx]["pose_keypoints_2d"][o+2]

                    if x_conf > th and is_target:
                        # 信頼度が一定以上あって、これまでも追加されている場合、追加
                        x_datas.append(x_data)
                    else:
                        # 一度でも信頼度が満たない場合、チェック対象外
                        is_target = False
                else:
                    # 既に並べられている人物の場合、比較対象にならない値を設定する
                    x_datas.append(sys.maxsize)
            
            # logger.info("sort_first_idxs: _idx: %s, x_datas: %s, is_target: %s", _idx, x_datas, is_target)

            if is_target:
                # 最終的に対象のままである場合、ひとつ前の人物に近い方のINDEXを取得する
                now_nearest_idxs.append(get_nearest_idx(x_datas, target_x[o]))

        # logger.info("sort_first_idxs: _idx: %s, now_nearest_idxs: %s", _idx, now_nearest_idxs)

        if len(now_nearest_idxs) > 0:
            # チェック対象件数がある場合、最頻出INDEXをチェックする
            most_common_idxs = Counter(now_nearest_idxs).most_common()
            logger.debug("sort_first_idxs: _idx: %s, most_common_idxs: %s", _idx, most_common_idxs)
            # 最頻出INDEX
            result_nearest_idxs[_idx] = most_common_idxs[0][0]
            # 次の比較元として、再頻出INDEXの人物を対象とする
            target_x = now_datas[most_common_idxs[0][0]]["pose_keypoints_2d"]

    logger.debug("sort_first_idxs: result_nearest_idxs: %s", result_nearest_idxs)

    if -1 in result_nearest_idxs:
        # 不採用になって判定できなかったデータがある場合
        for _nidx, _nval in enumerate(result_nearest_idxs):
            if _nval == -1:
                # 該当値が-1(判定不可）の場合
                for _cidx in range(len(now_datas)):
                    logger.debug("_nidx: %s, _nval: %s, _cidx: %s, _cidx not in nearest_idxs: %s", _nidx, _nval, _cidx, _cidx not in result_nearest_idxs)
                    # INDEXを頭から順に見ていく（正0, 正1 ... 正n, 逆0, 逆1 ... 逆n)
                    if _cidx not in result_nearest_idxs:
                        # 該当INDEXがリストに無い場合、設定
                        result_nearest_idxs[_nidx] = _cidx
                        break

    return result_nearest_idxs

# 前回のXYから片足寄せであるか判断する
def calc_leg_oneside(past_sorted_idxs, past_data, now_data, is_oneside_reset=False):
    # ひざと足首のペア
    LEG_IDXS = [[9,12],[10,13]]

    # 過去のX位置データ
    is_past_oneside = False
    for _pidx, _idx in enumerate(past_sorted_idxs):
        past_xyc = past_data[_idx]["pose_keypoints_2d"]

        for _lidx, _lvals in enumerate(LEG_IDXS):
            logger.debug("past _idx: %s, _lidx: %s, %sx: %s, %sx: %s, %sy: %s, %sy:%s", _idx, _lidx, _lvals[0], past_xyc[_lvals[0]*3], _lvals[1], past_xyc[_lvals[1]*3], _lvals[0], past_xyc[_lvals[0]*3+1], _lvals[1], past_xyc[_lvals[1]*3+1])
            
            if past_xyc[_lvals[0]*3] > 0 and past_xyc[_lvals[1]*3] > 0 and past_xyc[_lvals[0]*3+1] > 0 and past_xyc[_lvals[1]*3+1] > 0 \
                and abs(past_xyc[_lvals[0]*3] - past_xyc[_lvals[1]*3]) < 10 and abs(past_xyc[_lvals[0]*3+1] - past_xyc[_lvals[1]*3+1]) < 10:
                logger.debug("過去片寄せ: %s(%s), (%s,%s), (%s,%s)", _pidx, _lidx, past_xyc[_lvals[0]*3], past_xyc[_lvals[1]*3], past_xyc[_lvals[0]*3+1], past_xyc[_lvals[1]*3+1] )
                # 誰かの足が片寄せっぽいならば、FLG＝ON
                is_past_oneside = True

    is_leg_onesides = [ False for x in range(len(now_data)) ]
    # 今回のX位置データ
    for _idx in range(len(now_data)):
        now_xyc = now_data[_idx]["pose_keypoints_2d"]

        is_now_oneside_cnt = 0
        for _lidx, _lvals in enumerate(LEG_IDXS):
            logger.debug("now _idx: %s, _lidx: %s, %sx: %s, %sx: %s, %sy: %s, %sy:%s", _idx, _lidx, _lvals[0], now_xyc[_lvals[0]*3], _lvals[1], now_xyc[_lvals[1]*3], _lvals[0], now_xyc[_lvals[0]*3+1], _lvals[1], now_xyc[_lvals[1]*3+1])

            if now_xyc[_lvals[0]*3] > 0 and now_xyc[_lvals[1]*3] > 0 and now_xyc[_lvals[0]*3+1] > 0 and now_xyc[_lvals[1]*3+1] > 0 \
                and abs(now_xyc[_lvals[0]*3] - now_xyc[_lvals[1]*3]) < 10 and abs(now_xyc[_lvals[0]*3+1] - now_xyc[_lvals[1]*3+1]) < 10:
                # 両ひざ、両足首のX位置、Y位置がほぼ同じである場合
                logger.debug("現在片寄せ: %s(%s), (%s,%s), (%s,%s)", _idx, _lidx, now_xyc[_lvals[0]*3], now_xyc[_lvals[1]*3], now_xyc[_lvals[0]*3+1], now_xyc[_lvals[1]*3+1] )
                is_now_oneside_cnt += 1
        
        if is_now_oneside_cnt == len(LEG_IDXS) and is_past_oneside == False:
            # フラグを立てる
            is_leg_onesides[_idx] = True

            for _lidx, _lval in enumerate([8,9,10,11,12,13]):
                # リセットFLG＝ONの場合、足の位置を一旦全部クリア
                if is_oneside_reset:
                    now_xyc[_lval*3] = 0
                    now_xyc[_lval*3+1] = 0
                    now_xyc[_lval*3+2] = 0

    return is_leg_onesides


# 前回のXYから足関節が異常であるか判断する
def calc_leg_irregular(past_sorted_idxs, past_data, now_data, next_data, people_size, is_reset=False):

    now_sotred_data = [ [] for x in range(people_size) ]
    for _idx in range(people_size):
        # 過去の人物に近い現在INDEXを取得（上半身のみで判定）
        most_common_idxs = calc_upper_most_common_idxs(people_size, past_data, now_data[_idx])

        for mci in range(len(most_common_idxs)):
            now_idx = most_common_idxs[mci][0]
            # logger.debug("mci: %s, now_idx: %s", mci, now_idx)
            # logger.debug("now_sotred_data[now_idx]: %s", now_sotred_data[now_idx])
            # logger.debug("len(now_sotred_data[now_idx]): %s", len(now_sotred_data[now_idx]))
            if len(now_sotred_data[now_idx]) == 0:
                # まだ未設定の場合、ソート済みデータリストの該当INDEX箇所に設定
                now_sotred_data[now_idx] = now_data[_idx]
                break

    # 現在の人物分のデータを用意する
    next_sotred_data = [ [] for x in range(people_size) ]
    for _idx in range(people_size):
        # 現在の人物に近い未来INDEXを取得（上半身のみで判定）
        most_common_idxs = calc_upper_most_common_idxs(people_size, now_sotred_data, next_data[_idx])

        logger.debug("next most_common_idxs: %s, next_data[_idx]: %s", most_common_idxs, next_data[_idx])
        
        for mci in range(len(most_common_idxs)):
            next_idx = most_common_idxs[mci][0]
            # logger.debug("mci: %s, next_idx: %s", mci, next_idx)
            # logger.debug("next_sotred_data[next_idx]: %s", next_sotred_data[next_idx])
            # logger.debug("len(next_sotred_data[next_idx]): %s", len(next_sotred_data[next_idx]))
            if len(next_sotred_data[next_idx]) == 0:
                # まだ未設定の場合、ソート済みデータリストの該当INDEX箇所に設定
                next_sotred_data[next_idx] = next_data[_idx]
                break

    # logger.debug("past_data: %s", past_data)
    # logger.debug("now_data: %s", now_data)
    # logger.debug("now_sotred_data: %s", now_sotred_data)
    # logger.debug("next_data: %s", next_data)
    # logger.debug("next_sotred_data: %s", next_sotred_data)

    # ひざと足首のペア
    LEG_IDXS = [[9,12],[10,13]]

    is_leg_crosses = [ False for x in range(people_size) ]
    is_leg_onesides = [ False for x in range(len(now_data)) ]
    for _idx, (past_d, now_d, next_d) in enumerate(zip(past_data, now_sotred_data, next_sotred_data)):
        # logger.debug("past_d: %s", past_d)
        # logger.debug("now_d: %s", now_d)
        # logger.debug("next_d: %s", next_d)
        past_xyc = past_d["pose_keypoints_2d"]
        now_xyc = now_d["pose_keypoints_2d"]
        next_xyc = next_d["pose_keypoints_2d"]

        is_now_cross_cnt = 0
        is_now_oneside_cnt = 0
        for _lidx, _lvals in enumerate(LEG_IDXS):
            _lrightx = _lvals[0]*3
            _lleftx = _lvals[1]*3

            logger.debug("past _idx: %s, _lidx: %s, %sx: %s, %sx: %s, %sy: %s, %sy:%s", _idx, _lidx, _lvals[0], past_xyc[_lrightx], _lvals[1], past_xyc[_lleftx], _lvals[0], past_xyc[_lrightx+1], _lvals[1], past_xyc[_lleftx+1])
            logger.debug("now _idx: %s, _lidx: %s, %sx: %s, %sx: %s, %sy: %s, %sy:%s", _idx, _lidx, _lvals[0], now_xyc[_lrightx], _lvals[1], now_xyc[_lleftx], _lvals[0], now_xyc[_lrightx+1], _lvals[1], now_xyc[_lleftx+1])
            logger.debug("next _idx: %s, _lidx: %s, %sx: %s, %sx: %s, %sy: %s, %sy:%s", _idx, _lidx, _lvals[0], next_xyc[_lrightx], _lvals[1], next_xyc[_lleftx], _lvals[0], next_xyc[_lrightx+1], _lvals[1], next_xyc[_lleftx+1])

            # logger.debug("abs(past_xyc[_lrightx] - now_xyc[_lrightx]): %s, abs(past_xyc[_lrightx] - now_xyc[_lleftx]: %s, :%s", abs(past_xyc[_lrightx] - now_xyc[_lrightx]), abs(past_xyc[_lrightx] - now_xyc[_lleftx]), abs(past_xyc[_lrightx] - now_xyc[_lrightx]) > abs(past_xyc[_lrightx] - now_xyc[_lleftx]))
            # logger.debug("abs(past_xyc[_lleftx] - now_xyc[_lleftx]): %s, abs(past_xyc[_lleftx] - now_xyc[_lrightx]): %s, :%s", abs(past_xyc[_lleftx] - now_xyc[_lleftx]), abs(past_xyc[_lleftx] - now_xyc[_lrightx]), abs(past_xyc[_lrightx] - next_xyc[_lrightx]) < abs(past_xyc[_lrightx] - next_xyc[_lleftx]))
            # logger.debug("abs(past_xyc[_lrightx] - next_xyc[_lrightx]): %s, abs(past_xyc[_lrightx] - next_xyc[_lleftx]): %s, :%s", abs(past_xyc[_lrightx] - next_xyc[_lrightx]), abs(past_xyc[_lrightx] - next_xyc[_lleftx]), abs(past_xyc[_lleftx] - now_xyc[_lleftx]) > abs(past_xyc[_lleftx] - now_xyc[_lrightx]))
            # logger.debug("abs(past_xyc[_lleftx] - next_xyc[_lleftx]): %s, abs(past_xyc[_lleftx] - next_xyc[_lrightx]): %s, :%s", abs(past_xyc[_lleftx] - next_xyc[_lleftx]), abs(past_xyc[_lleftx] - next_xyc[_lrightx]), abs(past_xyc[_lleftx] - next_xyc[_lleftx]) < abs(past_xyc[_lleftx] - next_xyc[_lrightx]))

            if now_xyc[_lrightx] > 0 and now_xyc[_lleftx] > 0 and past_xyc[_lrightx] > 0 and past_xyc[_lleftx] > 0 and next_xyc[_lrightx] > 0 and next_xyc[_lleftx] > 0 :
                if abs(past_xyc[_lrightx] - now_xyc[_lrightx]) > abs(past_xyc[_lrightx] - now_xyc[_lleftx]) and \
                    abs(past_xyc[_lrightx] - next_xyc[_lrightx]) < abs(past_xyc[_lrightx] - next_xyc[_lleftx]) and \
                    abs(past_xyc[_lleftx] - now_xyc[_lleftx]) > abs(past_xyc[_lleftx] - now_xyc[_lrightx]) and \
                    abs(past_xyc[_lleftx] - next_xyc[_lleftx]) < abs(past_xyc[_lleftx] - next_xyc[_lrightx]) :
                    # 過去と現在で、反対方向の足の位置の方が近く、かつ過去と未来で、同じ方向の足の位置が近い場合、現在のみ交差しているとみなす
                    logger.info("！！足データ交差あり: %s(%s), nowx:(%s,%s), pastx:(%s,%s), nextx:(%s,%s)", _idx, _lidx, now_xyc[_lrightx], now_xyc[_lleftx], past_xyc[_lrightx], past_xyc[_lleftx], next_xyc[_lrightx], next_xyc[_lleftx] )
                    is_now_cross_cnt += 1
                else:
                    logger.debug("××足データ交差なし: %s(%s), nowx:(%s,%s), pastx:(%s,%s), nextx:(%s,%s)", _idx, _lidx, now_xyc[_lrightx], now_xyc[_lleftx], past_xyc[_lrightx], past_xyc[_lleftx], next_xyc[_lrightx], next_xyc[_lleftx] )

                if abs(now_xyc[_lrightx] - now_xyc[_lleftx]) < 10 and abs(now_xyc[_lrightx+1] - now_xyc[_lleftx+1]) < 10 \
                    and abs(past_xyc[_lrightx] - past_xyc[_lleftx]) > 10 and abs(past_xyc[_lrightx+1] - past_xyc[_lleftx+1]) > 10:
                    # 両ひざ、両足首のX位置、Y位置がほぼ同じである場合
                    logger.info("！！足データ片寄せあり: %s(%s), nowx:(%s,%s), pastx:(%s,%s), nextx:(%s,%s)", _idx, _lidx, now_xyc[_lrightx], now_xyc[_lleftx], past_xyc[_lrightx], past_xyc[_lleftx], next_xyc[_lrightx], next_xyc[_lleftx] )
                    is_now_oneside_cnt += 1
                else:
                    logger.debug("××足データ片寄せなし: %s(%s), nowx:(%s,%s), pastx:(%s,%s), nextx:(%s,%s)", _idx, _lidx, now_xyc[_lrightx], now_xyc[_lleftx], past_xyc[_lrightx], past_xyc[_lleftx], next_xyc[_lrightx], next_xyc[_lleftx] )

        if is_now_cross_cnt > 0:
            # フラグを立てる
            is_leg_crosses[_idx] = True

            for _lidx, _lval in enumerate([8,9,10,11,12,13]):
                # リセットFLG＝ONの場合、足の位置を一旦全部クリア
                if is_reset:
                    now_xyc[_lval*3] = 0
                    now_xyc[_lval*3+1] = 0
                    now_xyc[_lval*3+2] = 0

        if is_now_oneside_cnt == len(LEG_IDXS):
            # フラグを立てる
            is_leg_onesides[_idx] = True

            for _lidx, _lval in enumerate([8,9,10,11,12,13]):
                # リセットFLG＝ONの場合、足の位置を一旦全部クリア
                if is_reset:
                    now_xyc[_lval*3] = 0
                    now_xyc[_lval*3+1] = 0
                    now_xyc[_lval*3+2] = 0

    return is_leg_onesides, is_leg_crosses

def calc_upper_most_common_idxs(people_size, past_datas, now_datas):

    if people_size == 1:
        return [(0, 1)]

    # 過去データの上半身関節で、現在データと最も近いINDEXのリストを生成
    now_nearest_idxs = []
    most_common_idxs = []

    # logger.debug("calc_upper_most_common_idxs now_datas: %s", now_datas)
    # logger.debug("calc_upper_most_common_idxs past_datas: %s", past_datas)

    # # 位置データ(全身＋手足)
    # for _idx in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,2,3,4,5,6,7,8,9,10,11,12,13]:
    # 位置データ(上半身X)
    for _idx in [0,1,2,3,4,5,6,7,8,11,16,17]:
        one_data = now_datas["pose_keypoints_2d"][_idx*3]
        past_person = []
        for p in past_datas:
            # logger.debug("p: %s, c: %s", p, c)
            if _idx < len(p["pose_keypoints_2d"]):
                pdata = p["pose_keypoints_2d"][_idx*3]
                
                past_person.append(pdata) 

        # 今回データがないものはチェック対象外
        if len(past_person) > 0 and 0 not in past_person and one_data > 0:
            logger.debug("upper: %s, one_data %s", past_person, one_data)
            now_nearest_idxs.append(get_nearest_idx(past_person, one_data))
        else:
            # logger.debug("%s:: past_person対象外: %s, x_data %s", dimensional, past_person, x_data)
            pass

    if len(now_nearest_idxs) > 0:
        most_common_idxs = Counter(now_nearest_idxs).most_common()

    # 頻出で振り分けた後、件数が足りない場合（全部どれか1つに寄せられている場合)
    if len(most_common_idxs) < people_size:
        # logger.debug("頻出カウント不足: len(most_common_idxs): %s, len(conf_idxs): %s ", len(most_common_idxs), len(conf_idxs))
        for c in range(people_size):
            is_existed = False
            for m, mci in enumerate(most_common_idxs):
                if c == most_common_idxs[m][0]:
                    is_existed = True
                    break
            
            if is_existed == False:
                # 存在しないインデックスだった場合、追加                 
                most_common_idxs.append( (c, 0) )
    
    logger.debug("upper: most_common_idxs: %s, now_nearest_idxs: %s", most_common_idxs, now_nearest_idxs)

    return most_common_idxs


# 左右反転させたINDEX
OPENPOSE_REVERSE_ALL = {
    0: 0,
    1: 1,
    2: 5,
    3: 6,
    4: 7,
    5: 2,
    6: 3,
    7: 4,
    8: 11,
    9: 12,
    10: 13,
    11: 8,
    12: 9,
    13: 10,
    14: 15,
    15: 14,
    16: 17,
    17: 16,
    18: 18
}

# 上半身のみ左右反転させたINDEX
OPENPOSE_REVERSE_UPPER = {
    0: 0,
    1: 1,
    2: 5,
    3: 6,
    4: 7,
    5: 2,
    6: 3,
    7: 4,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 15,
    15: 14,
    16: 17,
    17: 16,
    18: 18
}

# 下半身のみ左右反転させたINDEX
OPENPOSE_REVERSE_LOWER = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 11,
    9: 12,
    10: 13,
    11: 8,
    12: 9,
    13: 10,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18
}

# 通常INDEX
OPENPOSE_NORMAL = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18
}

# 前回のXYと深度から近いindexを算出
def calc_nearest_idxs(past_sorted_idxs, past_data, now_data, past_pred_ary, now_pred_ary, max_conf_ary, max_conf_color_ary, past_frame, now_frame, limit_correction=0.0):
    # logger.debug("past_data: %s", past_data)
    
    # 前回の人物データ(前回のソート順に対応させる)
    # 左右反転もチェックするので、2倍。
    past_x_ary = [[] for x in range(len(past_data) * 2)]
    past_y_ary = [[] for x in range(len(past_data) * 2)]
    past_conf_ary = [[] for x in range(len(past_data) * 2)]
    # 下半身だけ回転しているパターン用
    past_lower_x_ary = [[] for x in range(len(past_data) * 2)]
    past_lower_y_ary = [[] for x in range(len(past_data) * 2)]
    past_lower_conf_ary = [[] for x in range(len(past_data) * 2)]
    # 上半身だけ回転しているパターン用
    past_upper_x_ary = [[] for x in range(len(past_data) * 2)]
    past_upper_y_ary = [[] for x in range(len(past_data) * 2)]
    past_upper_conf_ary = [[] for x in range(len(past_data) * 2)]
    # 過去画像の色情報リスト
    past_colors = [[] for x in range(len(past_data))]
    # 過去の首位置リスト
    past_necks = [0 for x in range(len(past_data))]
    for _idx, _idxv in enumerate(past_sorted_idxs):
        # logger.debug("past_data[_idx]: %s", past_data[_idx])

        past_xyc = past_data[_idx]["pose_keypoints_2d"]

        # logger.debug("_idx: %s, past_xyc: %s", _idx, past_xyc)
        # 正データ
        for o in range(0,len(past_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            # 全身反転用
            past_x_ary[_idx].append(past_xyc[o])
            past_y_ary[_idx].append(past_xyc[o+1])
            past_conf_ary[_idx].append(past_xyc[o+2])

            # 下半身反転用
            past_lower_x_ary[_idx].append(past_xyc[o])
            past_lower_y_ary[_idx].append(past_xyc[o+1])
            past_lower_conf_ary[_idx].append(past_xyc[o+2])

            # 上半身反転用
            past_upper_x_ary[_idx].append(past_xyc[o])
            past_upper_y_ary[_idx].append(past_xyc[o+1])
            past_upper_conf_ary[_idx].append(past_xyc[o+2])

            # 色情報
            if 0 < int(past_xyc[o+1]) < past_frame.shape[0] and 0 < int(past_xyc[o]) < past_frame.shape[1]:
                past_colors[_idx].append(past_frame[int(past_xyc[o+1]),int(past_xyc[o])])
                
                # 最高信頼度が書き換えられたら、色値も書き換える
                if max_conf_ary is not None:
                    if max_conf_ary[_idx][int(o/3)] < past_xyc[o+2]:
                        max_conf_color_ary[_idx][int(o/3)] = past_frame[int(past_xyc[o+1]),int(past_xyc[o])]
            else:
                # logger.warn("_idx: %s, o: %s, int(past_xyc[o+1]): %s, int(past_xyc[o]): %s", _idx, o, int(past_xyc[o+1]), int(past_xyc[o]))
                past_colors[_idx].append(np.array([0,0,0]))
            
            # 首位置
            if int(o/3) == 1:
                past_necks[_idx] = past_xyc[o]
        # 反転データ
        for o in range(0,len(past_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            past_x_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_ALL[int(o/3)]*3])
            past_y_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_ALL[int(o/3)]*3+1])
            # 反転は信頼度を下げる
            past_conf_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_ALL[int(o/3)]*3+2] - 0.1)
        # 下半身反転データ
        for o in range(0,len(past_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            past_lower_x_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_LOWER[int(o/3)]*3])
            past_lower_y_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_LOWER[int(o/3)]*3+1])
            # 反転は信頼度を下げる
            past_lower_conf_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_LOWER[int(o/3)]*3+2] - 0.1)
        # 上半身反転データ
        for o in range(0,len(past_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            past_upper_x_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_UPPER[int(o/3)]*3])
            past_upper_y_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_UPPER[int(o/3)]*3+1])
            # 反転は信頼度を下げる
            past_upper_conf_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_UPPER[int(o/3)]*3+2] - 0.1)
    
    logger.debug("max_conf_color_ary: %s", max_conf_color_ary)
    logger.debug("past_x: %s", np.array(past_x_ary)[:,1])

    # logger.debug("past_x_ary: %s", past_x_ary)
    # logger.debug("past_y_ary: %s", past_y_ary)

    # 今回の人物データ
    # 全身左右反転もチェックするので、2倍。
    now_x_ary = [[] for x in range(len(now_data) * 2)]
    now_y_ary = [[] for x in range(len(now_data) * 2)]
    now_conf_ary = [[] for x in range(len(now_data) * 2)]
    # 下半身だけ回転しているパターン用
    now_lower_x_ary = [[] for x in range(len(now_data) * 2)]
    now_lower_y_ary = [[] for x in range(len(now_data) * 2)]
    now_lower_conf_ary = [[] for x in range(len(now_data) * 2)]
    # 上半身だけ回転しているパターン用
    now_upper_x_ary = [[] for x in range(len(now_data) * 2)]
    now_upper_y_ary = [[] for x in range(len(now_data) * 2)]
    now_upper_conf_ary = [[] for x in range(len(now_data) * 2)]
    # 現在画像の色情報リスト
    now_colors = [[] for x in range(len(now_data))]
    # 現在の首X位置リスト
    now_necks = [0 for x in range(len(now_data))]
    for _idx in range(len(now_data)):
        now_xyc = now_data[_idx]["pose_keypoints_2d"]
        # logger.debug("_idx: %s, now_xyc: %s", _idx, now_xyc)
        # 正データ
        for o in range(0,len(now_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            now_x_ary[_idx].append(now_xyc[o])
            now_y_ary[_idx].append(now_xyc[o+1])
            now_conf_ary[_idx].append(now_xyc[o+2])

            # 下半身反転用
            now_lower_x_ary[_idx].append(now_xyc[o])
            now_lower_y_ary[_idx].append(now_xyc[o+1])
            now_lower_conf_ary[_idx].append(now_xyc[o+2])

            # 上半身反転用
            now_upper_x_ary[_idx].append(now_xyc[o])
            now_upper_y_ary[_idx].append(now_xyc[o+1])
            now_upper_conf_ary[_idx].append(now_xyc[o+2])

            # 色情報
            if 0 <= int(now_xyc[o+1]) < now_frame.shape[0] and 0 <= int(now_xyc[o]) < now_frame.shape[1]:
                now_colors[_idx].append(now_frame[int(now_xyc[o+1]),int(now_xyc[o])])
            else:
                now_colors[_idx].append(np.array([0,0,0]))

            # 首位置
            if int(o/3) == 1:
                now_necks[_idx] = now_xyc[o]
        # 反転データ
        for o in range(0,len(now_xyc),3):
            # logger.debug("_idx: %s, rev_idx: %s, o: %s, len(now_x_ary): %s, len(now_xyc): %s, OPENPOSE_REVERSE_ALL[o]: %s", _idx, _idx + len(now_data), o, len(now_x_ary), len(now_xyc), OPENPOSE_REVERSE_ALL[int(o/3)])
            now_x_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_ALL[int(o/3)]*3])
            now_y_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_ALL[int(o/3)]*3+1])
            # 反転は信頼度をすこし下げる
            now_conf_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_ALL[int(o/3)]*3+2] - 0.1)
        # 下半身反転データ
        for o in range(0,len(now_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            now_lower_x_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_LOWER[int(o/3)]*3])
            now_lower_y_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_LOWER[int(o/3)]*3+1])
            now_lower_conf_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_LOWER[int(o/3)]*3+2] - 0.1)
        # 上半身反転データ
        for o in range(0,len(now_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            now_upper_x_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_UPPER[int(o/3)]*3])
            now_upper_y_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_UPPER[int(o/3)]*3+1])
            now_upper_conf_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_UPPER[int(o/3)]*3+2] - 0.1)

    logger.debug("now_x: %s", np.array(now_x_ary)[:,1])

    # 過去の深度データ
    past_pred = []
    for _pidx, _idx in enumerate(past_sorted_idxs):
        past_pred.append(past_pred_ary[_idx])

    # logger.debug("past_pred: %s,", past_pred)
    # logger.debug("org_past_conf: %s,", org_past_conf)

    # 信頼度の高い順に人物インデックスを割り当てていく
    avg_conf_ary = []
    for con in now_conf_ary:
        # 体幹ほど重みをつけて平均値を求める
        avg_conf_ary.append(np.average(np.array(con), weights=[0.5,2.0,0.5,0.3,0.1,0.5,0.3,0.1,0.8,0.3,0.1,0.8,0.3,0.1,0.1,0.1,0.1,0.1]))
    
    # 信頼度の低い順のインデックス番号
    conf_idxs = np.argsort(avg_conf_ary)
    logger.debug("avg_conf_ary: %s", avg_conf_ary)
    logger.debug("conf_idxs: %s", conf_idxs)

    # # 信頼度の高い順に人物インデックスを割り当てていく
    # normal_avg_conf_ary = []
    # for con in now_conf_ary[0:len(now_data)]:
    #     # 体幹ほど重みをつけて平均値を求める
    #     normal_avg_conf_ary.append(np.average(np.array(con), weights=[0.5,0.8,0.5,0.3,0.1,0.5,0.3,0.1,0.8,0.3,0.1,0.8,0.3,0.1,0.1,0.1,0.1,0.1]))
    
    # # 信頼度の低い順のインデックス番号
    # normal_conf_idxs = np.argsort(normal_avg_conf_ary)

    # conf_idxs = [-1 for x in range(len(now_conf_ary))]
    # for _ncidx in range(len(normal_conf_idxs)):
    #     # 正データ
    #     conf_idxs[_ncidx] = normal_conf_idxs[_ncidx]+len(now_data)
    #     # 反転データ
    #     conf_idxs[_ncidx+len(now_data)] = normal_conf_idxs[_ncidx]

    # logger.debug("normal_avg_conf_ary: %s", normal_avg_conf_ary)
    # logger.debug("normal_conf_idxs: %s", normal_conf_idxs)
    # logger.debug("conf_idxs: %s", conf_idxs)

    nearest_idxs = [-1 for x in range(len(conf_idxs))]
    is_upper_reverses = [False for x in range(len(conf_idxs))]
    is_lower_reverses = [False for x in range(len(conf_idxs))]
    most_common_idxs = []

    # logger.debug("past_pred_ary: %s", past_pred_ary)
    # logger.debug("now_pred_ary: %s", now_pred_ary)
    # XY正の判定用
    XY_LIMIT = 0.73 + limit_correction
    # XY上半身・下半身のみ反転用。やや厳しめ
    REV_LIMIT = 0.83 + limit_correction
    # 深度判定用。甘め
    D_LIMIT = 0.61 + limit_correction
    # 色度判定用。甘め
    C_LIMIT = 0.61 + limit_correction

    logger.debug("XY_LIMIT: %s, REV_LIMIT: %s, C_LIMIT: %s", XY_LIMIT, REV_LIMIT, C_LIMIT)

    # 複数人数のソートであるか
    is_multi_sort = len(past_sorted_idxs) > 1

    # 首位置がほとんど同じものは優先採用
    for ncidx in range(len(now_necks)):
        for pcidx in range(len(past_necks)):
            if abs(past_necks[pcidx] - now_necks[ncidx]) < 3:
                # 首位置がほとんど動いていない場合、優先採用
                logger.debug("首優先採用: ncidx: %s, now: %s, pcidx: %s, past: %s", ncidx, now_necks[ncidx], pcidx, past_necks[pcidx])
                nearest_idxs[ncidx] = pcidx
                break

    # 信頼度の低い順の逆順(信頼度降順)に人物を当てはめていく
    cidx = len(conf_idxs) - 1
    cidxcnt = 0
    while cidx >= 0 and cidxcnt < len(conf_idxs):
        now_conf_idx = conf_idxs[cidx]
        now_x = now_x_ary[now_conf_idx]
        now_y = now_y_ary[now_conf_idx]
        now_conf = now_conf_ary[now_conf_idx]

        logger.debug("cidx: %s, now_conf_idx: %s ----------------------------------------", cidx, now_conf_idx )
        logger.debug("now_x: %s", now_x)

        # 過去データの当該関節で、現在データと最も近いINDEXのリストを生成
        now_nearest_idxs, most_common_idxs, is_y = calc_most_common_idxs( is_multi_sort, conf_idxs, now_x, now_y, now_conf, past_x_ary, past_y_ary, past_lower_conf_ary, OPENPOSE_NORMAL, XY_LIMIT)

        sum_most_common_idxs, most_common_per, same_frame_per, top_frame, second_frame, is_top = \
            get_most_common_frames(now_nearest_idxs, most_common_idxs, conf_idxs)
        
        logger.debug("len(now_nearest_idxs): %s, all_size: %s, per: %s", len(now_nearest_idxs), (len(now_x) + ( 0 if is_y == False else len(now_y) )), len(now_nearest_idxs) / (len(now_x) + ( 0 if is_y == False else len(now_y) )))
        logger.info("now_nearest_idxs: %s, most_common_per: %s", now_nearest_idxs, most_common_per)

        if most_common_per < XY_LIMIT or len(now_nearest_idxs) / (len(now_x) + ( 0 if is_y == False else len(now_y) )) < 0.25:
            # 再頻出が指定未満、チェック対象件数が指定未満、のいずれかの場合、
            # 上半身と下半身で回転が違っている可能性あり。

            logger.info("下半身反転データチェック cidx: %s, now_conf_idx: %s", cidx, now_conf_idx)
            # 下半身だけ反転しているデータで比較する
            now_lower_x = now_lower_x_ary[now_conf_idx]
            now_lower_y = now_lower_y_ary[now_conf_idx]
            now_lower_conf = now_lower_conf_ary[now_conf_idx]

            lower_now_nearest_idxs, lower_most_common_idxs, is_y = calc_most_common_idxs(is_multi_sort, conf_idxs, now_lower_x, now_lower_y, now_lower_conf, past_lower_x_ary, past_lower_y_ary, past_conf_ary, OPENPOSE_REVERSE_LOWER, REV_LIMIT )

            sum_lower_most_common_idxs, lower_most_common_per, lower_same_frame_per, lower_top_frame, lower_second_frame, is_top_lower = \
                get_most_common_frames(lower_now_nearest_idxs, lower_most_common_idxs, conf_idxs)
            logger.info("lower_most_common_per: %s, most_common_per: %s", lower_most_common_per, most_common_per)

            if lower_most_common_per > REV_LIMIT and lower_most_common_per > most_common_per:
                # # 下半身反転データも同じINDEXで、より精度が高い場合、採用
                if (now_x[2] == 0 or now_x[3] == 0 or now_x[5] == 0 or now_x[6] == 0):
                    # 上半身がない場合、全身反転とする
                    now_nearest_idxs = []
                    for lnni in lower_now_nearest_idxs:
                        now_nearest_idxs.append(lnni + len(now_data))

                    most_common_idxs = Counter(now_nearest_idxs).most_common()

                    for c in range(len(conf_idxs)):
                        is_existed = False
                        for m, mci in enumerate(most_common_idxs):
                            if c == most_common_idxs[m][0]:
                                is_existed = True
                                break
                        
                        if is_existed == False:
                            # 存在しないインデックスだった場合、追加
                            most_common_idxs.append( (c, 0) )
                    logger.info("＊下半身→全身反転データ採用: now_nearest_idxs: %s, most_common_idxs: %s", now_nearest_idxs, most_common_idxs)
                else:
                    now_nearest_idxs = lower_now_nearest_idxs
                    most_common_idxs = lower_most_common_idxs
                    is_lower_reverses[now_conf_idx] = True
                    logger.info("＊下半身反転データ採用: lower_now_nearest_idxs: %s, lower_most_common_idxs: %s, is_lower_reverses: %s", lower_now_nearest_idxs, lower_most_common_idxs, is_lower_reverses)
            else:
                # 信頼度が最後のものはチェックしない
                # 精度が高くない場合、上半身反転データチェック
                logger.info("上半身反転データチェック cidx: %s, now_conf_idx: %s", cidx, now_conf_idx)

                # 上半身だけ反転しているデータで比較する
                now_upper_x = now_upper_x_ary[now_conf_idx]
                now_upper_y = now_upper_y_ary[now_conf_idx]
                now_upper_conf = now_upper_conf_ary[now_conf_idx]

                upper_now_nearest_idxs, upper_most_common_idxs, is_y = calc_most_common_idxs(is_multi_sort, conf_idxs, now_upper_x, now_upper_y, now_upper_conf, past_upper_x_ary, past_upper_y_ary, past_upper_conf_ary, OPENPOSE_REVERSE_UPPER, REV_LIMIT)

                sum_upper_most_common_idxs, upper_most_common_per, upper_same_frame_per, upper_top_frame, upper_second_frame, is_top_upper = \
                    get_most_common_frames(upper_now_nearest_idxs, upper_most_common_idxs, conf_idxs)
                logger.info("upper_most_common_per: %s, most_common_per: %s", upper_most_common_per, most_common_per)

                if upper_most_common_per > REV_LIMIT and upper_most_common_per > most_common_per:
                    # 上半身反転データも同じINDEXで、より精度が高い場合、採用
                    if (now_x[8] == 0 or now_x[9] == 0 or now_x[11] == 0 or now_x[12] == 0):
                        # 下半身がない場合、全身反転とする
                        now_nearest_idxs = []
                        for unni in upper_now_nearest_idxs:
                            now_nearest_idxs.append(unni + len(now_data))

                        most_common_idxs = Counter(now_nearest_idxs).most_common()

                        for c in range(len(conf_idxs)):
                            is_existed = False
                            for m, mci in enumerate(most_common_idxs):
                                if c == most_common_idxs[m][0]:
                                    is_existed = True
                                    break
                            
                            if is_existed == False:
                                # 存在しないインデックスだった場合、追加
                                most_common_idxs.append( (c, 0) )

                        logger.info("＊上半身→全身反転データ採用: now_nearest_idxs: %s, most_common_idxs: %s", now_nearest_idxs, most_common_idxs)
                    else:
                        now_nearest_idxs = upper_now_nearest_idxs
                        most_common_idxs = upper_most_common_idxs
                        is_upper_reverses[now_conf_idx] = True
                        logger.info("＊上半身反転データ採用: upper_now_nearest_idxs: %s, upper_most_common_idxs: %s, is_upper_reverses: %s", upper_now_nearest_idxs, upper_most_common_idxs, is_upper_reverses)
                else:
                    # logger.debug("most_common_idxs: %s, lower_most_common_idxs: %s, upper_most_common_idxs: %s", most_common_idxs, lower_most_common_idxs, upper_most_common_idxs )

                    # # TOP1.2で上位を占めているか
                    # logger.info("再検査:: same_frame_per: %s, len(now_x): %s, top: %s, second: %s, is_top: %s", same_frame_per, int(len(conf_idxs)/2), top_frame, second_frame, is_top)

                    # if is_top:
                    #     logger.info("全身TOP2の最頻出同一枠のため全身採用: same_frame_per: %s, top: %s, second: %s", same_frame_per, most_common_idxs[1][0] % len(now_data), most_common_idxs[1][0] % len(now_data))
                    #     is_upper_reverses[now_conf_idx] = False
                    #     is_lower_reverses[now_conf_idx] = False
                    # else:
                    # 下半身反転も上半身反転もダメな場合、深度チェック
                    logger.info("深度データチェック cidx: %s, now_conf_idx: %s", cidx, now_conf_idx)
                    # 深度データは反転保持していないので、半分にする
                    now_depth = now_pred_ary[int(now_conf_idx % len(now_data))]

                    depth_now_nearest_idxs, depth_most_common_idxs = calc_depth_most_common_idxs(conf_idxs, now_depth, now_conf, past_pred, past_conf_ary, now_nearest_idxs)

                    sum_depth_most_common_idxs, depth_most_common_per, depth_same_frame_per, depth_top_frame, depth_second_frame, is_top_depth = \
                        get_most_common_frames(depth_now_nearest_idxs, depth_most_common_idxs, conf_idxs)
                        
                    logger.info("depth_most_common_per: %s, most_common_per: %s", depth_most_common_per, most_common_per)
                    
                    if depth_most_common_per > D_LIMIT and depth_most_common_per > most_common_per:
                        now_nearest_idxs = depth_now_nearest_idxs
                        most_common_idxs = depth_most_common_idxs
                        logger.info("＊深度データ採用: depth_now_nearest_idxs: %s, depth_most_common_idxs: %s", depth_now_nearest_idxs, depth_most_common_idxs)
                    else:
                        # 下半身反転も上半身反転も深度推定ダメな場合、色チェック
                        logger.info("色データチェック cidx: %s, now_conf_idx: %s", cidx, now_conf_idx)
                        # 色データは反転保持していないので、半分にする
                        now_color = now_colors[int(now_conf_idx % len(now_data))]

                        color_now_nearest_idxs, color_most_common_idxs = calc_color_most_common_idxs(conf_idxs, now_color, now_conf, max_conf_color_ary, past_conf_ary, now_nearest_idxs)

                        sum_color_most_common_idxs = 0
                        for lmci_data in color_most_common_idxs:
                            sum_color_most_common_idxs += lmci_data[1]

                        sum_most_common_idxs = 0
                        for mci_data in most_common_idxs:
                            sum_most_common_idxs += mci_data[1]

                        color_most_common_per = 0 if sum_color_most_common_idxs == 0 else color_most_common_idxs[0][1] / sum_color_most_common_idxs
                        most_common_per = 0 if sum_most_common_idxs == 0 else most_common_idxs[0][1] / sum_most_common_idxs
                        logger.info("color_most_common_per: %s, most_common_per: %s", color_most_common_per, most_common_per)
                        
                        # color_most_common_perの下限は甘め
                        if color_most_common_per > C_LIMIT and color_most_common_per > most_common_per:
                            now_nearest_idxs = color_now_nearest_idxs
                            most_common_idxs = color_most_common_idxs
                            is_upper_reverses[now_conf_idx] = False
                            is_lower_reverses[now_conf_idx] = False
                            logger.info("＊色データ採用: color_now_nearest_idxs: %s, color_most_common_idxs: %s", color_now_nearest_idxs, color_most_common_idxs)
                        else:
                            # どのパターンも採用できなかった場合、採用なしで次にいく
                            logger.info("採用なし")
                            now_nearest_idxs = [0]
                            most_common_idxs = [(0,0)]

                            # # 深度データも駄目だったので、とりあえずこれまでの中でもっとも確率の高いのを採用する
                            # if most_common_idxs[0][0] in nearest_idxs and lower_most_common_per > most_common_per:
                            #     now_nearest_idxs = lower_now_nearest_idxs
                            #     most_common_idxs = lower_most_common_idxs
                            #     is_lower_reverses[now_conf_idx] = True
                            #     logger.info("＊深度データ不採用→下半身反転データ採用: lower_now_nearest_idxs: %s, lower_most_common_idxs: %s, is_lower_reverses: %s", lower_now_nearest_idxs, lower_most_common_idxs, is_lower_reverses)
                            # elif most_common_idxs[0][0] in nearest_idxs and upper_most_common_per > most_common_per:
                            #     now_nearest_idxs = upper_now_nearest_idxs
                            #     most_common_idxs = upper_most_common_idxs
                            #     is_upper_reverses[now_conf_idx] = True
                            #     logger.info("＊深度データ不採用→上半身反転データ採用: upper_now_nearest_idxs: %s, upper_most_common_idxs: %s, is_upper_reverses: %s", upper_now_nearest_idxs, upper_most_common_idxs, is_upper_reverses)
                            # else:
                            #     logger.info("＊深度データ不採用→全身データ採用: upper_now_nearest_idxs: %s, upper_most_common_idxs: %s, is_upper_reverses: %s", upper_now_nearest_idxs, upper_most_common_idxs, is_upper_reverses)
            
        logger.debug("cidx: %s, most_common_idx: %s", cidx, most_common_idxs)
        
        is_passed = False
        # 最も多くヒットしたINDEXを処理対象とする
        for cmn_idx in range(len(most_common_idxs)):
            # 入れようとしているINDEXが、採用枠（前半）か不採用枠（後半）か
            if now_conf_idx < len(now_data):
                # 採用枠(前半)の場合
                check_ary = nearest_idxs[0: len(now_data)]
            else:
                # 不採用枠(後半)の場合
                check_ary = nearest_idxs[len(now_data): len(now_data)*2]
            
            logger.debug("nearest_idxs: %s, most_common_idxs[cmn_idx][0]: %s, check_ary: %s", nearest_idxs, most_common_idxs[cmn_idx][0], check_ary )

            is_idx_existed = False
            for ca in check_ary:
                logger.debug("ca: %s, ca / len(now): %s, most / len(now): %s", ca, ca % len(now_data), most_common_idxs[cmn_idx][0] % len(now_data))
                if ca >= 0 and ca % len(now_data) == most_common_idxs[cmn_idx][0] % len(now_data):
                    # 同じ枠に既に同じINDEXの候補が居る場合、TRUE
                    is_idx_existed = True
                    break

            if most_common_idxs[cmn_idx][0] in nearest_idxs or is_idx_existed:
                # 同じINDEXが既にリストにある場合
                # もしくは入れようとしているINDEXが反対枠の同じ並び順にいるか否か
                # logger.info("次点繰り上げ cmn_idx:%s, val: %s, nearest_idxs: %s", cmn_idx, most_common_idxs[cmn_idx][0], nearest_idxs)
                # continue
                logger.info("既出スキップ cmn_idx:%s, val: %s, nearest_idxs: %s", cmn_idx, most_common_idxs[cmn_idx][0], nearest_idxs)
                # 既出の場合、これ以上チェックできないので、次にいく
                cidx -= 1
                break
            elif most_common_idxs[cmn_idx][1] > 0:
                # 同じINDEXがリストにまだない場合
                logger.info("採用 cmn_idx:%s, val: %s, nearest_idxs: %s", cmn_idx, most_common_idxs[cmn_idx][0], nearest_idxs)
                # 採用の場合、cidx減算
                is_passed = True
                cidx -= 1
                break
            else:
                logger.info("再頻出ゼロ cmn_idx:%s, val: %s, nearest_idxs: %s", cmn_idx, most_common_idxs[cmn_idx][0], nearest_idxs)
                # 最頻出がない場合、これ以上チェックできないので、次にいく
                cidx -= 1
                break

        logger.info("結果: near: %s, cmn_idx: %s, val: %s, most_common_idxs: %s", now_conf_idx, cmn_idx, most_common_idxs[cmn_idx][0], most_common_idxs)

        if is_passed:
            # 信頼度の高いINDEXに該当する最多ヒットINDEXを設定
            nearest_idxs[now_conf_idx] = most_common_idxs[cmn_idx][0]
        
        # 現在のループ回数は必ず加算
        cidxcnt += 1

        logger.debug("now_conf_idx: %s, cidx: %s, cidxcnt: %s, nearest_idxs: %s ---------------------", now_conf_idx, cidx, cidxcnt, nearest_idxs)

    logger.debug("nearest_idxs: %s", nearest_idxs)

    if -1 in nearest_idxs:
        # 不採用になって判定できなかったデータがある場合
        for _nidx, _nval in enumerate(nearest_idxs):
            if _nval == -1:
                # 該当値が-1(判定不可）の場合
                for _cidx in range(len(conf_idxs)):
                    logger.debug("_nidx: %s, _nval: %s, _cidx: %s, _cidx not in nearest_idxs: %s", _nidx, _nval, _cidx, _cidx not in nearest_idxs)
                    # INDEXを頭から順に見ていく（正0, 正1 ... 正n, 逆0, 逆1 ... 逆n)
                    if _cidx not in nearest_idxs:

                        # 入れようとしているINDEXが、採用枠（前半）か不採用枠（後半）か
                        if now_conf_idx < len(now_data):
                            # 採用枠(前半)の場合
                            check_ary = nearest_idxs[len(now_data): len(now_data)*2]
                        else:
                            # 不採用枠(後半)の場合
                            check_ary = nearest_idxs[0: len(now_data)]
                        
                        logger.debug("nearest_idxs: %s, _cidx: %s, check_ary: %s", nearest_idxs, _cidx, check_ary )

                        is_idx_existed = False
                        for ca in check_ary:
                            logger.debug("ca: %s, ca / len(now): %s, _cidx / len(now): %s", ca, ca % len(now_data), _cidx % len(now_data))
                            if ca >= 0 and ca % len(now_data) == _cidx % len(now_data):
                                # 同じ枠に既に同じINDEXの候補が居る場合、TRUE
                                is_idx_existed = True
                                break

                        if is_idx_existed == False:
                            # 該当INDEXがリストに無い場合、設定
                            nearest_idxs[_nidx] = _cidx
                            break

    logger.debug("is_upper_reverses: %s, is_lower_reverses: %s", is_upper_reverses, is_lower_reverses)
    logger.debug("past_sorted_idxs: %s nearest_idxs(retake): %s", past_sorted_idxs, nearest_idxs)

    # 最終的に人数分だけ残したINDEXリスト
    result_nearest_idxs = [-1 for x in range(len(now_data))]
    result_is_all_reverses = [False for x in range(len(now_data))]
    result_is_upper_reverses = [False for x in range(len(now_data))]
    result_is_lower_reverses = [False for x in range(len(now_data))]
    for _ridx in range(len(now_data)):
        # # 反転の可能性があるので、人数で割った余りを設定する
        sidx = int(nearest_idxs[_ridx] % len(now_data))

        if _ridx < len(now_data):
            # 自分より前に、自分と同じINDEXが居る場合、次のINDEXを引っ張り出す
            s = 1
            while sidx in result_nearest_idxs[0:_ridx+1]:
                newsidx = int(nearest_idxs[_ridx+s] % len(now_data))
                logger.info("INDEX重複のため、次点繰り上げ: %s, sidx: %s, newsidx: %s", _ridx, sidx, newsidx)
                sidx = newsidx
                s += 1

        result_nearest_idxs[_ridx] = sidx
        result_is_upper_reverses[sidx] = is_upper_reverses[_ridx]
        result_is_lower_reverses[sidx] = is_lower_reverses[_ridx]

        idx_target = OPENPOSE_NORMAL
        if result_is_upper_reverses[sidx] and result_is_lower_reverses[sidx]:
            # 全身反転
            idx_target = OPENPOSE_REVERSE_ALL
        elif result_is_upper_reverses[sidx] and result_is_lower_reverses[sidx] == False:
            # 反転している場合、反転INDEX(上半身)
            idx_target = OPENPOSE_REVERSE_UPPER
        elif result_is_upper_reverses[sidx] == False and result_is_lower_reverses[sidx]:
            # 反転している場合、反転INDEX(下半身)
            idx_target = OPENPOSE_REVERSE_LOWER

        # 上下の左右が合っているか
        if is_match_left_right(now_x, idx_target) == False:
            # 上下の左右があってない場合、とりあえず反転クリア
            result_is_upper_reverses[sidx] = False
            result_is_lower_reverses[sidx] = False

        result_is_all_reverses[sidx] = True if nearest_idxs[_ridx] >= len(now_data) and is_upper_reverses[_ridx] == False and is_lower_reverses[_ridx] == False else False

    logger.info("result_nearest_idxs: %s, all: %s, upper: %s, lower: %s", result_nearest_idxs, result_is_all_reverses, result_is_upper_reverses, result_is_lower_reverses)

    return result_nearest_idxs, result_is_all_reverses, result_is_upper_reverses, result_is_lower_reverses

# 上半身と下半身で左右の方向が合っているか
def is_match_left_right(now_x, idx_target):

    shoulder = now_x[idx_target[2]] > 0 and now_x[idx_target[5]] > 0 and (now_x[idx_target[2]] - now_x[idx_target[5]]) < 0
    elbow = now_x[idx_target[3]] > 0 and now_x[idx_target[6]] > 0 and (now_x[idx_target[3]] - now_x[idx_target[6]]) < 0
    hip = now_x[idx_target[8]] > 0 and now_x[idx_target[11]] > 0 and (now_x[idx_target[8]] - now_x[idx_target[11]]) < 0
    knee = now_x[idx_target[9]] > 0 and now_x[idx_target[12]] > 0 and (now_x[idx_target[9]] - now_x[idx_target[12]]) < 0

    if shoulder == elbow == hip == knee:
        logger.debug("方向統一: shoulder: %s, elbow: %s, hip: %s, knee: %s, x: %s", shoulder, elbow, hip, knee, now_x)
    else:
        if shoulder == elbow and shoulder != hip and hip == knee:
            logger.debug("上下で方向ずれあり: shoulder: %s, elbow: %s, hip: %s, knee: %s, x: %s", shoulder, elbow, hip, knee, now_x)
            return False
        else:
            logger.debug("上下バラバラ: shoulder: %s, elbow: %s, hip: %s, knee: %s, x: %s", shoulder, elbow, hip, knee, now_x)

    # 明示的なずれでなければ、とりあえずTRUE
    return True


# 過去データと現在データを比較して、頻出インデックス算出
def calc_most_common_idxs(is_multi_sort, conf_idxs, now_x, now_y, now_confs, past_x_ary, past_y_ary, past_conf_ary, idx_target, limit_th):
    # 過去データの当該関節で、現在データと最も近いINDEXのリストを生成
    now_nearest_idxs = []
    most_common_idxs = []
    th = 0.3

    # X方向の頻出インデックス(th高めで過去データを含まない) ------------------
    now_nearest_idxs, most_common_idxs = \
        calc_one_dimensional_most_common_idxs("x", is_multi_sort, conf_idxs, now_x, now_confs, past_x_ary, past_conf_ary, now_nearest_idxs, idx_target, 0.3)

    sum_most_common_idxs, most_common_per, same_frame_per, top_frame, second_frame, is_top = \
        get_most_common_frames(now_nearest_idxs, most_common_idxs, conf_idxs)

    if (most_common_per > limit_th or same_frame_per > limit_th) and len(now_nearest_idxs) >= len(now_x) * 0.2:
        # 一定件数以上で、上限を満たしている場合、結果を返す
        return now_nearest_idxs, most_common_idxs, False

    # X方向の頻出インデックス(th低めで過去データを含む) ------------------
    now_nearest_idxs, most_common_idxs = \
        calc_one_dimensional_most_common_idxs("x", is_multi_sort, conf_idxs, now_x, now_confs, past_x_ary, past_conf_ary, now_nearest_idxs, idx_target, 0.0)

    sum_most_common_idxs, most_common_per, same_frame_per, top_frame, second_frame, is_top = \
        get_most_common_frames(now_nearest_idxs, most_common_idxs, conf_idxs)

    if (most_common_per > limit_th or same_frame_per > limit_th) and len(now_nearest_idxs) >= len(now_x) * 0.2:
        # 一定件数以上で、上限を満たしている場合、結果を返す
        return now_nearest_idxs, most_common_idxs, False

    # Y方向の頻出インデックス(th高めで過去データを含まない) ------------------
    now_nearest_idxs, most_common_idxs = \
        calc_one_dimensional_most_common_idxs("y", is_multi_sort,  conf_idxs, now_y, now_confs, past_y_ary, past_conf_ary, now_nearest_idxs, idx_target, 0.3)

    if (most_common_per > limit_th or same_frame_per > limit_th) and len(now_nearest_idxs) >= len(now_y) * 0.2:
        # 一定件数以上で、上限を満たしている場合、結果を返す
        return now_nearest_idxs, most_common_idxs, True

    # Y方向の頻出インデックス(th低めで過去データを含む) ------------------
    now_nearest_idxs, most_common_idxs = \
        calc_one_dimensional_most_common_idxs("y", is_multi_sort,  conf_idxs, now_y, now_confs, past_y_ary, past_conf_ary, now_nearest_idxs, idx_target, 0.0)

    return now_nearest_idxs, most_common_idxs, True

# 一方向だけの頻出インデックス算出
def calc_one_dimensional_most_common_idxs(dimensional, is_multi_sort, conf_idxs, now_datas, now_confs, past_datas, past_confs, now_nearest_idxs, idx_target, th):
    logger.debug("calc_one_dimensional_most_common_idxs: %s, th=%s", dimensional, th)

    # この前の頻出は引き継がない
    now_nearest_idxs = []
    # 過去データの当該関節で、現在データと最も近いINDEXのリストを生成
    most_common_idxs = []

    # 判定対象は全身
    TARGET_IDX = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

    if is_multi_sort == True:
        # 複数人数トレースの場合、体幹中心にソートする
        TARGET_IDX = [1,2,3,5,6,8,9,10,11,12,13,1,1,1]

    # 位置データ(+体幹)
    for _idx in TARGET_IDX:
        one_data = now_datas[_idx]
        past_person = []
        zero_cnt = 0
        for p, c in zip(past_datas, past_confs):
            # logger.debug("p: %s, c: %s", p, c)
            if _idx < len(p):
                if c[idx_target[_idx]] > th:
                    # 信頼度が一定以上の場合、判定対象
                    past_person.append(p[idx_target[_idx]]) 
                else:
                    past_person.append(0)

                if past_person[-1] == 0:
                    zero_cnt += 1

        if len(past_person) > 0 and len(past_person) > zero_cnt and one_data > 0 and now_confs[_idx] > th:
            logger.debug("_idx: %s, %s: %s, one_data %s", _idx, dimensional, past_person, one_data)
            now_nearest_idxs.append(get_nearest_idx(past_person, one_data))
        else:
            logger.debug("×対象外 _idx: %s, %s: %s, one_data %s", _idx, dimensional, past_person, one_data)
            pass

    if len(now_nearest_idxs) > 0:
        most_common_idxs = Counter(now_nearest_idxs).most_common()
    
    if is_multi_sort == True:
        # 複数人数トレースの場合、全体の中心もチェックする
        
        past_persons_avg = []
        for p, c in zip(past_datas, past_confs):
            p_sum = 0
            p_cnt = 0
            for _idx in TARGET_IDX:
                if _idx < len(p):
                    if c[idx_target[_idx]] > th:
                        # 信頼度が一定以上の場合、判定対象
                        p_sum += p[idx_target[_idx]]
                        p_cnt += 1

            # 平均値を求める
            if p_cnt > 0:
                past_persons_avg.append(p_sum / p_cnt)
            else:
                past_persons_avg.append(0)

            if past_persons_avg[-1] == 0:
                zero_cnt += 1

        now_avg = 0
        n_sum = 0
        n_cnt = 0
        for _idx in TARGET_IDX:
            if now_confs[_idx] > th:
                # 信頼度が一定以上の場合、判定対象
                n_sum += now_datas[_idx]
                n_cnt += 1

        # 平均値を求める
        if n_cnt > 0:
            now_avg = n_sum / n_cnt

        # TOPの枠
        top_frame = -1 if len(most_common_idxs) <= 0 else most_common_idxs[0][0] % int(len(conf_idxs)/2)

        # 多めに求める
        for cnt in range(3):
            if len(past_persons_avg) > 0 and len(past_persons_avg) > zero_cnt and now_avg > 0 and now_confs[_idx] > th:
                logger.debug("avg _idx: %s, %s: %s, one_data %s", _idx, dimensional, past_persons_avg, now_avg)
                avg_nearest_idx = get_nearest_idx(past_persons_avg, now_avg)

                # 現在の枠
                now_frame = avg_nearest_idx % int(len(conf_idxs)/2)

                if top_frame == now_frame:
                    # TOPの枠と、現在の枠が同じ場合、TOPの枠を設定する
                    now_nearest_idxs.append(most_common_idxs[0][0])
                else:
                    now_nearest_idxs.append(avg_nearest_idx)
            else:
                logger.debug("×avg対象外 _idx: %s, %s: %s, one_data %s", _idx, dimensional, past_persons_avg, now_avg)
                pass

    if len(now_nearest_idxs) > 0:
        most_common_idxs = Counter(now_nearest_idxs).most_common()

    # 頻出で振り分けた後、件数が足りない場合（全部どれか1つに寄せられている場合)
    if len(most_common_idxs) < len(conf_idxs):
        # logger.debug("頻出カウント不足: len(most_common_idxs): %s, len(conf_idxs): %s ", len(most_common_idxs), len(conf_idxs))
        for c in range(len(conf_idxs)):
            is_existed = False
            for m, mci in enumerate(most_common_idxs):
                if c == most_common_idxs[m][0]:
                    is_existed = True
                    break
            
            if is_existed == False:
                # 存在しないインデックスだった場合、追加                 
                most_common_idxs.append( (c, 0) )
    
    logger.debug("%s:: len(most_common_idxs): %s, len(conf_idxs): %s, len(now_nearest_idxs): %s, dimensional,len(now_datas): %s", dimensional, len(most_common_idxs), len(conf_idxs), len(now_nearest_idxs), len(now_datas))
    logger.debug("%s:: now_nearest_idxs: %s, most_common_idxs: %s", dimensional, now_nearest_idxs, most_common_idxs)

    return now_nearest_idxs, most_common_idxs

def get_most_common_frames(now_nearest_idxs, most_common_idxs, conf_idxs):

    top_frame = most_common_idxs[0][0] % int(len(conf_idxs)/2)

    # 同じ枠と合わせた割合を計算する
    sum_most_common_idxs = 0
    same_frames_most_common_idxs = 0
    for smidx in range(len(most_common_idxs)):
        now_frame = most_common_idxs[smidx][0] % int(len(conf_idxs)/2)

        if top_frame == now_frame:
            same_frames_most_common_idxs += most_common_idxs[smidx][1]
        
        sum_most_common_idxs += most_common_idxs[smidx][1]

    logger.debug("sum_most_common_idxs: %s, same_frames_most_common_idxs: %s", sum_most_common_idxs, same_frames_most_common_idxs)
    
    # 同じ枠の割合
    same_frame_per = 0 if sum_most_common_idxs == 0 else same_frames_most_common_idxs / sum_most_common_idxs

    # 同じ枠と合わせた割合を計算する
    smidx = 1
    while smidx < len(most_common_idxs):
        # logger.debug("smidx: %s, most_common_idxs[1][1]: %s, most_common_idxs[smidx][1]: %s", smidx, most_common_idxs[1][1], most_common_idxs[smidx][1])
        if most_common_idxs[1][1] == most_common_idxs[smidx][1]:
            # ２位と３位以下が同率の場合
            second_frame = most_common_idxs[1][0] % int(len(conf_idxs)/2)
            third_frame = most_common_idxs[smidx][0] % int(len(conf_idxs)/2)
            # 1位と同じ枠を採用
            second_frame = third_frame if top_frame == third_frame else second_frame
            
            # logger.debug("smidx: %s, top_frame: %s, second_frame: %s, third_frame: %s, most_common_idxs: %s", smidx, top_frame, second_frame, third_frame, most_common_idxs)

            smidx += 1
        else:
            second_frame = most_common_idxs[1][0] % int(len(conf_idxs)/2)
            break

    most_common_per = 0 if sum_most_common_idxs == 0 else (most_common_idxs[0][1]) / sum_most_common_idxs

    logger.debug("top_frame: %s, second_frame: %s, sum_most_common_idxs: %s, most_common_per: %s", top_frame, second_frame, sum_most_common_idxs, most_common_per)

    # TOP1だけで7割か、同じ枠で7.3割か
    is_top = most_common_idxs[0][1] > 0.7 or same_frame_per > 0.73
    logger.debug("same_frame_per: %s, len(now_datas): %s, top: %s, second: %s, is_top: %s", same_frame_per, int(len(conf_idxs)/2), top_frame, second_frame, is_top)

    return sum_most_common_idxs, most_common_per, same_frame_per, top_frame, second_frame, is_top


# 色差データで人物判定
def calc_color_most_common_idxs(conf_idxs, now_clr, now_conf, past_color_ary, past_conf_ary, now_nearest_idxs):
    # XYの頻出は引き継がない
    now_nearest_idxs = []
    most_common_idxs = []
    th = 0.1

    # 色データ(全身)
    for c_idx in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
        if c_idx < len(now_clr):
            c_data = now_clr[c_idx]
            past_colors = []
            for p in past_color_ary:
                if c_idx < len(p):
                    # logger.debug("c_idx: %s, p[c_idx]: %s, c_data: %s", c_idx, p[c_idx], c_data)
                    past_colors.append(p[c_idx]) 

            # 今回データがないものはチェック対象外
            if len(past_colors) > 0 and (c_data > 0).all():
                logger.debug("_idx: %s, c: %s, one_data %s", c_idx, past_colors, c_data)
                now_nearest_idxs.append(get_nearest_idx_ary(past_colors, c_data))
            else:
                logger.debug("past_colors対象外: %s, c_data %s", past_colors, c_data)

    if len(now_nearest_idxs) > 0:
        most_common_idxs = Counter(now_nearest_idxs).most_common()

    logger.debug("c:: now_nearest_idxs: %s, most_common_idxs: %s, ", now_nearest_idxs, most_common_idxs)

    # 頻出で振り分けた後、件数が足りない場合（全部どれか1つに寄せられている場合)
    if len(most_common_idxs) < len(conf_idxs):
        # logger.debug("頻出カウント不足: len(most_common_idxs): %s, len(conf_idxs): %s ", len(most_common_idxs), len(conf_idxs))
        for c in range(len(conf_idxs)):
            is_existed = False
            for m, mci in enumerate(most_common_idxs):
                if c == most_common_idxs[m][0]:
                    is_existed = True
                    break
            
            if is_existed == False:
                # 存在しないインデックスだった場合、追加                 
                most_common_idxs.append( (c, 0) )
    
    return now_nearest_idxs, most_common_idxs


# 深度データで人物判定
def calc_depth_most_common_idxs(conf_idxs, now_depth, now_conf, past_depth_ary, past_conf_ary, now_nearest_idxs):
    # XYの頻出は引き継がない
    now_nearest_idxs = []
    most_common_idxs = []
    th = 0.1

    # 深度データ(末端除く)
    for d_idx in [0,1,2,3,5,6,8,9,11,12,14,15,16,17]:
        if d_idx < len(now_depth):
            d_data = now_depth[d_idx]
            past_depths = []
            for p in past_depth_ary:
                if d_idx < len(p):
                    # logger.debug("d_idx: %s, p[d_idx]: %s, c[d_idx]: %s", d_idx, p[d_idx], c[d_idx])
                    past_depths.append(p[d_idx]) 

            # 今回データがないものはチェック対象外
            if len(past_depths) > 0 and 0 not in past_depths and d_data > 0:
                logger.debug("past_depths: %s, d_data %s", past_depths, d_data)
                now_nearest_idxs.append(get_nearest_idx(past_depths, d_data))
            else:
                logger.debug("past_depths対象外: %s, d_data %s", past_depths, d_data)

    if len(now_nearest_idxs) > 0:
        most_common_idxs = Counter(now_nearest_idxs).most_common()

    logger.debug("d:: now_nearest_idxs: %s, most_common_idxs: %s, ", now_nearest_idxs, most_common_idxs)

    # logger.debug("past_depth_ary: %s", past_depth_ary)

    # past_depths = []
    # for p in past_depth_ary:
    #     past_sum_depths = []
    #     logger.debug("now_depth: %s", now_depth)
    #     for d_idx in range(len(now_depth)):
    #         logger.debug("d_idx: %s", d_idx)
    #         past_sum_depths.append(p[d_idx]) 

    #     logger.debug("past_sum_depths: %s", past_sum_depths)

    #     # 重み付けした平均値を求める
    #     past_depths.append(np.average(np.array(past_sum_depths), weights=[0.1,0.8,0.5,0.3,0.1,0.5,0.3,0.1,0.8,0.3,0.1,0.8,0.3,0.1,0.1,0.1,0.1,0.1]))
        
    #     # 今回データがないものはチェック対象外
    #     # if len(past_depths) > 0 and 0 not in past_depths and d_data > 0 and now_conf[d_idx] > th:
    #     #     logger.debug("[limbs] past_depths: %s, d_data %s", past_depths, d_data)
    #     #     now_nearest_idxs.append(get_nearest_idx(past_depths, d_data))

    # if len(now_nearest_idxs) > 0:
    #     most_common_idxs = Counter(now_nearest_idxs).most_common()

    # logger.debug("d:: now_nearest_idxs: %s, most_common_idxs: %s", now_nearest_idxs, most_common_idxs)

    # 頻出で振り分けた後、件数が足りない場合（全部どれか1つに寄せられている場合)
    if len(most_common_idxs) < len(conf_idxs):
        # logger.debug("頻出カウント不足: len(most_common_idxs): %s, len(conf_idxs): %s ", len(most_common_idxs), len(conf_idxs))
        for c in range(len(conf_idxs)):
            is_existed = False
            for m, mci in enumerate(most_common_idxs):
                if c == most_common_idxs[m][0]:
                    is_existed = True
                    break
            
            if is_existed == False:
                # 存在しないインデックスだった場合、追加                 
                most_common_idxs.append( (c, 0) )
    
    return now_nearest_idxs, most_common_idxs


def get_nearest_idx(target_list, num):
    """
    概要: リストからある値に最も近い値のINDEXを返却する関数
    @param target_list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値のINDEX
    """

    # logger.debug(target_list)
    # logger.debug(num)

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(target_list) - num).argmin()
    return idx


def get_nearest_idx_ary(target_list, num_ary):
    """
    概要: リストからある値に最も近い値のINDEXを返却する関数
    @param target_list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値のINDEX
    """

    # logger.debug(target_list)
    # logger.debug(num)

    target_list2 = []
    for t in target_list:
        # 現在との色の差を絶対値で求めて、10の位で四捨五入する
        target_list2.append(np.round(np.abs(t - num_ary), decimals=-1))

    # logger.debug("num_ary: %s", num_ary)
    # logger.debug("target_list: %s", target_list)
    # logger.debug("target_list2: %s", target_list2)

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idxs = np.asarray(target_list2).argmin(axis=0)
    # logger.debug("np.asarray(target_list2).argmin(axis=0): %s", idxs)

    idx = np.argmax(np.bincount(idxs))
    # logger.debug("np.argmax(np.bincount(idxs)): %s", idx)

    return idx