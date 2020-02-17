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
from PIL import Image

# ファイル出力ログ用
file_logger = logging.getLogger("message").getChild(__name__)
logger = logging.getLogger("__main__").getChild(__name__)

level = {0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG}

# 人物ソート
def exec(pred_depth_ary, pred_depth_support_ary, pred_conf_ary, pred_conf_support_ary, pred_image_ary, video_path, now_str, subdir, json_path, json_size, number_people_max, reverse_specific_dict, order_specific_dict, start_json_name, start_frame, end_frame_no, org_width, org_height, png_lib, scale, prev_sorted_idxs, verbose):

    logger.warn("人物ソート開始 ---------------------------")

    # 前回情報
    past_pattern_datas = [{} for x in range(number_people_max)]
    past_sorted_idxs = [x for x in range(number_people_max)]
    # 判断材料の範囲
    dimensional_range = {"x": {"min": 0, "max": 0}, "y": {"min": 0, "max": 0}, "depth": {"min": 0, "max": 0}, "depth_support": {"min": 0, "max": 0}}

    if len(prev_sorted_idxs) > 0:
        # 過去データ流用の場合、流用
        past_sorted_idxs, past_pattern_datas = load_sorted_idxs(json_path, now_str, number_people_max, prev_sorted_idxs, start_json_name, start_frame, pred_depth_ary[len(prev_sorted_idxs) - 1])

    cnt = 0
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        # 動画から1枚キャプチャして読み込む
        flag, frame = cap.read()  # Capture frame-by-frame

        # 深度推定のindex
        _idx = cnt - start_frame
        _display_idx = cnt

        # 開始フレームより前は飛ばす
        if start_frame > cnt or len(prev_sorted_idxs) + start_frame > cnt:
            cnt += 1
            continue

        # 終わったフレームより後は飛ばす
        # 明示的に終わりが指定されている場合、その時も終了する
        if flag == False or cnt >= json_size + start_frame or (end_frame_no > 0 and _idx >= end_frame_no):
            break
        
        # 開始シーンのJSONデータを読み込む
        file_name = "{0}.json".format(cnt)
        _file = os.path.join(json_path, file_name)
        try:
            data = json.load(open(_file))
        except Exception as e:
            logger.warning("JSON読み込み失敗のため、空データ読み込み, %s %s", _file, e)
            data = json.load(open("json/all_empty_keypoints.json"))

        for i in range(len(data["people"]), number_people_max):
            # 足りない分は空データを埋める
            data["people"].append(json.load(open("json/one_keypoints.json")))

        logger.debug("＊＊＊人体別処理: iidx: %s file: %s --------", _idx, file_name)

        # フレームイメージをオリジナルのサイズで保持(色差用)
        frame_img = np.array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), dtype=np.float32)

        # 判断材料の範囲に最大値設定
        dimensional_range["x"]["max"] = frame_img.shape[1]
        dimensional_range["y"]["max"] = frame_img.shape[0]

        # 前フレームと出来るだけ近い位置のINDEX順番を計算する
        sorted_idxs, now_pattern_datas = calc_sort_and_direction(_idx, reverse_specific_dict, order_specific_dict, number_people_max, past_pattern_datas, data, pred_depth_ary[_idx], pred_depth_support_ary[_idx], pred_conf_ary[_idx], pred_conf_support_ary[_idx], frame_img, past_sorted_idxs, dimensional_range)

        # 出力する
        output_sorted_data(_idx, _display_idx, number_people_max, sorted_idxs, now_pattern_datas, json_path, now_str, file_name, reverse_specific_dict, order_specific_dict)

        # 画像保存
        save_image(_idx, pred_image_ary, frame_img, number_people_max, sorted_idxs, now_pattern_datas, subdir, cnt, png_lib, scale, verbose)

        # 今回分を前回分に置き換え
        past_pattern_datas = now_pattern_datas
        # 前回のソート順を保持
        past_sorted_idxs = sorted_idxs

        # インクリメント
        cnt += 1

# 前フレームと出来るだけ近い位置のINDEX順番を計算する
def calc_sort_and_direction(_idx, reverse_specific_dict, order_specific_dict, number_people_max, past_pattern_datas, data, pred_depth, pred_depth_support, pred_conf, pred_conf_support, frame_img, past_sorted_idxs, dimensional_range):
    # ソート順
    sorted_idxs = [-1 for x in range(number_people_max)]

    if _idx == 0:
        # 今回情報
        now_pattern_datas = [{} for x in range(number_people_max)]

        # 最初はインデックスの通りに並べる
        for _pidx in range(number_people_max):
            sorted_idxs[_pidx] = _pidx

        # パターンはノーマルで生成        
        for _pidx in range(number_people_max):
            # パターン別のデータ
            now_pattern_datas[_pidx] = {"eidx": _pidx, "pidx": _pidx, "sidx": _pidx, "in_idx": _pidx, "pattern": OPENPOSE_NORMAL["pattern"], 
                "x": np.zeros(18), "y": np.zeros(18), "conf": np.zeros(18), "fill": [False for x in range(18)], "depth": np.zeros(18), 
                "depth_support": np.zeros(17), "conf_support": np.zeros(17), "color": [None for x in range(18)], "x_avg": 0, "conf_avg": 0}

            # 1人分の関節位置データ
            now_xyc = data["people"][_pidx]["pose_keypoints_2d"]

            for o in range(0,len(now_xyc),3):
                oidx = int(o/3)
                now_pattern_datas[_pidx]["x"][oidx] = now_xyc[OPENPOSE_NORMAL[oidx]*3]
                now_pattern_datas[_pidx]["y"][oidx] = now_xyc[OPENPOSE_NORMAL[oidx]*3+1]
                now_pattern_datas[_pidx]["conf"][oidx] = now_xyc[OPENPOSE_NORMAL[oidx]*3+2]
                now_pattern_datas[_pidx]["depth"][oidx] = pred_depth[_pidx][OPENPOSE_NORMAL[oidx]]

                # 色情報
                if 0 <= int(now_xyc[o+1]) < frame_img.shape[0] and 0 <= int(now_xyc[o]) < frame_img.shape[1]:
                    now_pattern_datas[_pidx]["color"][oidx] = frame_img[int(now_xyc[o+1]),int(now_xyc[o])]
                else:
                    now_pattern_datas[_pidx]["color"][oidx] = np.array([0,0,0])

            # 深度補佐データ
            now_pattern_datas[_pidx]["depth_support"] = pred_depth_support[_pidx]
            now_pattern_datas[_pidx]["conf_support"] = pred_conf_support[_pidx]

        # 前回データはそのまま
        return sorted_idxs, now_pattern_datas
    else:
        # ソートのための準備
        pattern_datas = prepare_sort(_idx, number_people_max, data, pred_depth, pred_depth_support, pred_conf, pred_conf_support, frame_img, past_sorted_idxs, past_pattern_datas)

        # 順番が指定されている場合、適用
        order_sorted_idxs = None
        if _idx in order_specific_dict:
            order_sorted_idxs = order_specific_dict[_idx]

        # 再頻出INDEXを算出
        return calc_sort_and_direction_frame(_idx, reverse_specific_dict, number_people_max, past_pattern_datas, pattern_datas, order_sorted_idxs, past_sorted_idxs, frame_img, dimensional_range)

# ソート順と向きを求める
def calc_sort_and_direction_frame(_idx, reverse_specific_dict, number_people_max, past_pattern_datas, pattern_datas, order_sorted_idxs, past_sorted_idxs, frame_img, dimensional_range):
    # ソート結果
    sorted_idxs = [-1 for _ in range(number_people_max)]
    # sorted_in_idxs = [-1 for _ in range(number_people_max)]

    if number_people_max == 1:
        # 1人の場合はソート不要
        sorted_idxs = [0]
        sorted_in_idxs = [0]
    else:
        # 過去データの前回のINDEXに相当するデータを生成
        all_pattern_datas = [{} for x in range(len(past_pattern_datas))]
        all_past_pattern_datas = [{} for x in range(len(past_pattern_datas) * 4)]

        for _eidx in range(len(past_pattern_datas)):
            # 今回データ
            all_pattern_datas[_eidx] = copy.deepcopy(pattern_datas[_eidx*4])

            # 前回データの反転データを作成する
            for op_idx, op_idx_data in enumerate(OP_PATTERNS):
                ppd = copy.deepcopy(past_pattern_datas[_eidx])
                for _didx in range(len(ppd["conf"])):
                    for dimensional in ["x","y","depth"]:
                        ppd[dimensional][_didx] = past_pattern_datas[_eidx][dimensional][op_idx_data[_didx]]
                    ppd["conf"][_didx] = past_pattern_datas[_eidx]["conf"][op_idx_data[_didx]] * 0.9
                    ppd["fill"][_didx] = True

                # 前回データ
                all_past_pattern_datas[_eidx*4+op_idx] = ppd

        # 現在データを基準にソート順を求める
        sorted_idxs = [i for i in range(number_people_max)]
        # sorted_idxs = calc_sort_frame(_idx, number_people_max, all_pattern_datas, all_past_pattern_datas, dimensional_range, \
        #     [{"th": 0.4, "past_th": 0.0, "most_th": 0.7, "all_most_th": 0.5, "ppd_th": 0.5}, \
        #     {"th": 0.01, "past_th": 0.0, "most_th": 0.6, "all_most_th": 0.4, "ppd_th": 0.3}, \
        #     {"th": 0.01, "past_th": 0.0, "most_th": 0.55, "all_most_th": 0.3, "ppd_th": 0.0}, \
        #     {"th": 0.01, "past_th": 0.0, "most_th": 0.51, "all_most_th": 0.3, "ppd_th": 0.0}])
        
        logger.debug("_idx: %s, sorted_idxs: %s", _idx, sorted_idxs)

        if order_sorted_idxs:
            copy_sorted_idxs = copy.deepcopy(sorted_idxs)
            # 順番が指定されている場合、適用
            for _eidx, osi in enumerate(order_sorted_idxs):
                now_sidx = get_nearest_idxs(sorted_idxs, _eidx)[0]
                copy_sorted_idxs[now_sidx] = osi
            sorted_idxs = copy_sorted_idxs

    # 人物INDEXが定まったところで、向きを再確認する
    now_pattern_datas = calc_direction_frame(_idx, number_people_max, past_pattern_datas, pattern_datas, sorted_idxs, frame_img, 0.1, 0.4)

    logger.debug("now_pattern_datas: %s", now_pattern_datas)

    for _eidx, _sidx in enumerate(sorted_idxs):
        now_sidx = get_nearest_idxs(sorted_idxs, _eidx)[0]

        # 反転パターン
        if _idx in reverse_specific_dict:
            for _ridx in reverse_specific_dict[_idx]:
                if _ridx == _eidx:
                    pattern_type = 0
                    if reverse_specific_dict[_idx][_ridx] == "N":
                        pattern_type = 0
                    elif reverse_specific_dict[_idx][_ridx] == "R":
                        pattern_type = 1
                    elif reverse_specific_dict[_idx][_ridx] == "U":
                        pattern_type = 2
                    elif reverse_specific_dict[_idx][_ridx] == "L":
                        pattern_type = 3

                    now_pattern_datas[_eidx] = pattern_datas[now_sidx*4+pattern_type]

    # 過去データ引継
    if _idx > 0:
        for _eidx, (npd, ppd) in enumerate(zip(now_pattern_datas, past_pattern_datas)):
            logger.debug("npd: %s", npd)
            for _didx in range(len(npd["conf"])):
                if npd["conf"][_didx] < 0.1 and npd["conf"][_didx] < ppd["conf"][_didx]:
                    for dimensional in ["x","y","depth"]:
                        # 信頼度が全くない場合、過去データで埋める（信頼度は下げる）
                        npd[dimensional][_didx] = ppd[dimensional][_didx]
                    npd["conf"][_didx] = ppd["conf"][_didx] * 0.8
                    npd["fill"][_didx] = True

    return sorted_idxs, now_pattern_datas


# 指定された方向（x, y, depth, color）に沿って、向きを計算する
def calc_direction_frame(_idx, number_people_max, past_pattern_datas, pattern_datas, sorted_idxs, frame_img, th, most_th):
    # 今回のパターン結果
    now_pattern_datas = [[] for x in range(len(past_pattern_datas))]
    # 0F目のSINDEXリスト
    start_sidxs = [x for x in range(number_people_max)]

    for _eidx, _sidx in enumerate(sorted_idxs):
        # Openpose推定結果の0F目の該当INDEXの人物があるINDEXを取得する
        now_sidx = get_nearest_idxs(sorted_idxs, _eidx)[0]

        # 直近INDEX
        now_nearest_idxs = []
        # 最頻出INDEX
        most_common_idxs = []

        # 足が同一方向で算出されている場合、TRUE
        is_leg_same_direction = False
        # 足が付け根と反転している場合、TRUE
        is_leg_reverse = False
        # 前回データと今回データをチェック
        for pd in [past_pattern_datas[_eidx], pattern_datas[now_sidx*4]]:
            # 足が同一方向で算出されている場合
            for (lidx, ridx) in [(8,11), (9,12), (10,13)]:
                if abs(pd["x"][lidx] - pd["x"][ridx]) < frame_img.shape[1] / 100:
                    is_leg_same_direction = True

            # 足が付け根と反転している場合、TRUE
            if pd["conf"][8] >= 0.1 and pd["x"][11] >= 0.1 and pd["conf"][9] >= 0.1 and pd["x"][12] >= 0.1 and \
                np.sign(pd["x"][8] - pd["x"][11]) != np.sign(pd["x"][9] - pd["x"][12]):
                    is_now_leg_reverse = True

        if is_leg_same_direction or is_leg_reverse:
            # 足がほとんど同じ位置にあるか、足が付け根と反転している場合、正方向で取得する
            now_pattern_datas[_eidx] = pattern_datas[now_sidx*4]
            now_pattern_datas[_eidx]["sidx"] = now_sidx
        else:
            for _didx, dimensional in enumerate(["x","y"]):
                # 上半身と下半身で同じ回数分チェックできるよう調整
                jidx_rng = [(0,1),(1,1),(2,3),(3,1),(4,1),(5,3),(6,1),(7,1),(8,4),(9,2),(10,2),(11,4),(12,2),(13,2),(14,1),(15,1),(16,1),(17,1)]
                for (_jidx, jcnt) in jidx_rng:
                    # 前回の該当関節データ
                    for _ in range(jcnt):
                        is_check = True

                        # 今回の該当関節データリスト
                        now_per_joint_data = []
                        now_per_joint_conf = []
                        for _pidx, pt_data in enumerate(pattern_datas[now_sidx*4:now_sidx*4+4]):
                            if pt_data["conf"][_jidx] >= th:
                                # 該当辺の該当関節値を設定
                                if dimensional == "xy":
                                    now_per_joint_data.append(np.array([pt_data["x"][_jidx],pt_data["y"][_jidx]]))
                                else:
                                    now_per_joint_data.append(pt_data[dimensional][_jidx])
                                now_per_joint_conf.append(pt_data["conf"][_jidx])
                            else:
                                # 足りない場合、チェック対象外
                                is_check = False
                                break

                        if not is_check:
                            # 足りてない場合、あり得ない値
                            now_nearest_idxs.append(-1)
                            continue

                        if past_pattern_datas[_eidx]["conf"][_jidx] >= th:
                            # 信頼度が足りてる場合、前回のチェック対象関節値    
                            if dimensional == "xy":
                                # XY座標の場合は組合せでチェック
                                past_per_joint_value = np.array([past_pattern_datas[_eidx]["x"][_jidx],past_pattern_datas[_eidx]["y"][_jidx]])
                                nearest_idx = get_nearest_idx_ary(now_per_joint_data, past_per_joint_value, now_per_joint_conf, th)
                            else:
                                past_per_joint_value = past_pattern_datas[_eidx][dimensional][_jidx]
                                nearest_idx = get_nearest_idxs(now_per_joint_data, past_per_joint_value, now_per_joint_conf, th)

                            if 0 < len(nearest_idx) < len(now_per_joint_conf):
                                # 偏向して近似INDEXがある場合、そのまま追加
                                now_nearest_idxs.extend(nearest_idx)
                            elif 0 < len(nearest_idx) and len(nearest_idx) == len(now_per_joint_conf):
                                # 全部同じ数の場合、最小INDEXのみ追加
                                now_nearest_idxs.append(min(nearest_idx))
                        else:
                            if _jidx in [0,14,15]:
                                # 顔が足りていない場合、正面向きとみなす
                                now_nearest_idxs.append(0)
                            else:
                                # 足りてない場合、あり得ない値
                                now_nearest_idxs.append(-1)

                if len(now_nearest_idxs) > 0:
                    most_common_idxs = Counter(now_nearest_idxs).most_common()

                    most_common_cnt = 0
                    for mci in most_common_idxs:
                        most_common_cnt += mci[1]

                    # 件数の入ってるのだけ拾う
                    most_common_idxs = [mci for mci in most_common_idxs if mci[0] >= 0]

                    # 頻出で振り分けた後、件数が足りない場合（全部どれか1つに寄せられている場合)
                    if len(most_common_idxs) < len(past_pattern_datas):
                        for c in range(len(past_pattern_datas)):
                            is_existed = False
                            for m, mci in enumerate(most_common_idxs):
                                if c == most_common_idxs[m][0]:
                                    is_existed = True
                                    break
                            
                            if is_existed == False:
                                # 存在しないインデックスだった場合、追加                 
                                most_common_idxs.append( (c, 0) )
                    
                    if most_common_cnt > 0:
                        most_common_per = most_common_idxs[0][1] / most_common_cnt
                        if most_common_idxs[0][0] >= 0 and most_common_per >= most_th - (_didx * 0.1):
                            # 再頻出INDEXが有効で、再頻出INDEXの出現数が全体の既定割合を超えていれば終了
                            break
                            
            op_direction = most_common_idxs[0][0]

            if op_direction in [2, 3]:
                # 上半身または下半身のみ反転の場合
                pd = pattern_datas[now_sidx*4+op_direction]
                
                if np.sign(pd["x"][1] - np.median([x for x in pd["x"][2:4] if x > 0])) != np.sign(pd["x"][1] - np.median([x for x in pd["x"][8:10] if x > 0])) or \
                    np.sign(pd["x"][1] - np.median([x for x in pd["x"][5:7] if x > 0])) != np.sign(pd["x"][1] - np.median([x for x in pd["x"][11:13] if x > 0])):
                    # 上下で右半身と左半身の符号が違う場合、反転クリア
                    op_direction = 0
                
            now_pattern_datas[_eidx] = pattern_datas[now_sidx*4+op_direction]
            now_pattern_datas[_eidx]["sidx"] = now_sidx

    return now_pattern_datas


# 指定された方向（x, y, depth, color）に沿って、ソート順を計算する
def calc_sort_frame(_idx, number_people_max, pattern_datas, past_pattern_datas, dimensional_range, param_ths):
    # ソート結果
    sorted_idxs = [-1 for _ in range(number_people_max)]

    for pth in param_ths:
        # 最頻出INDEXの割合(自身のINDEXも持つ)
        all_most_common_per = []
        all_now_per_joint_data = []

        th = pth["th"]
        past_th = pth["past_th"]
        most_th = pth["most_th"]
        all_most_th = pth["all_most_th"]
        ppd_th = pth["ppd_th"]

        # 最終的に追加設定したいINDEX
        target_not_existed_idxs = [x for x in range(number_people_max) if x not in sorted_idxs]
        # 追加設定したいINDEXに相当する過去データ
        target_now_idxs = [(e, x, get_nearest_idxs(sorted_idxs, -1)[e]) for e, x in enumerate(target_not_existed_idxs)]
        # 現在対象となっている最大人数
        now_number_people_max = len(target_not_existed_idxs)

        for tp_idx in target_now_idxs:
            for op_idx in range(4):
                _eidx = tp_idx[1]*4+op_idx
                ppt_data = past_pattern_datas[_eidx]

                if np.median(ppt_data["conf"]) <= ppd_th:
                    # 過去データの信頼度が低い場合、処理スキップ
                    continue

                for _didx, dimensional in enumerate(["x", "y", "depth", "depth_support", "color"]):
                    # 直近INDEX
                    now_nearest_idxs = []
                    # 最頻出INDEX
                    most_common_idxs = []
                    # 再頻出INDEXの割合
                    most_common_per = 0

                    # 範囲を限定する（深度補佐は全部）
                    jidx_rng = [(0,1),(1,5),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),(8,3),(9,2),(10,2),(11,3),(12,2),(13,2),(14,1),(15,1),(16,1),(17,1)] if dimensional in ["x", "depth", "y"] else [(x, 1) for x in range(len(pt_data[dimensional]))]
                    for (_jidx, jcnt) in jidx_rng:
                        # 前回の該当関節データ
                        for _ in range(jcnt):
                            is_check = True

                            now_per_joint_data = []
                            now_per_joint_conf = []
                            for tn_idx in target_now_idxs:
                                pt_data = pattern_datas[tn_idx[2]]

                                # 信頼度が足りている場合、該当辺の該当関節値を設定
                                now_per_joint_conf.append(pt_data["conf"][_jidx])
                                if dimensional == "xy":
                                    now_per_joint_data.append(np.array([pt_data["x"][_jidx],pt_data["y"][_jidx]]))
                                elif dimensional == "xd":
                                    now_per_joint_data.append(np.array([pt_data["x"][_jidx],pt_data["depth"][_jidx]]))
                                elif dimensional == "xyd":
                                    now_per_joint_data.append(np.array([pt_data["x"][_jidx],pt_data["depth"][_jidx],pt_data["y"][_jidx]]))
                                else:
                                    now_per_joint_data.append(pt_data[dimensional][_jidx])

                            # 今回チェックしようとしている関節値
                            if dimensional == "xy":
                                per_joint_value = np.array([ppt_data["x"][_jidx],ppt_data["y"][_jidx]])
                            elif dimensional == "xd":
                                per_joint_value = np.array([ppt_data["x"][_jidx],ppt_data["depth"][_jidx]])
                            elif dimensional == "xyd":
                                per_joint_value = np.array([ppt_data["x"][_jidx],ppt_data["depth"][_jidx],ppt_data["y"][_jidx]])
                            else:
                                per_joint_value = ppt_data[dimensional][_jidx]

                            all_now_per_joint_data.append((now_per_joint_data, per_joint_value))

                            if dimensional in ["xyd", "xd", "xy", "color"]:
                                if ppt_data["conf"][_jidx] > past_th:
                                    # XY座標と色の場合は組合せでチェック
                                    now_nearest_idxs.extend(get_nearest_idx_ary(now_per_joint_data, per_joint_value, now_per_joint_conf, th))
                                else:
                                    # 足りてない場合、あり得ない値
                                    now_nearest_idxs.append(-1)
                            else:
                                if ppt_data["conf"][_jidx] > past_th:
                                    # 信頼度が足りてる場合、直近のINDEXを取得
                                    now_nearest_idxs.extend(get_nearest_idxs(now_per_joint_data, per_joint_value, now_per_joint_conf, th, dimensional_range[dimensional]["min"], dimensional_range[dimensional]["max"]))
                                    logger.debug("now_nearest_idxs: %s", now_nearest_idxs)
                                else:
                                    # 足りてない場合、あり得ない値
                                    now_nearest_idxs.append(-1)

                    if len(now_nearest_idxs) > 0:
                        most_common_idxs = Counter(now_nearest_idxs).most_common()
                        
                        most_common_cnt_all = 0
                        for mci in most_common_idxs:
                            most_common_cnt_all += mci[1]

                        # 件数の入ってるのだけ拾う
                        most_common_idxs = [mci for mci in most_common_idxs if mci[0] >= 0]
                        
                        # 頻出で振り分けた後、件数が足りない場合（全部どれか1つに寄せられている場合)
                        if len(most_common_idxs) < now_number_people_max:
                            for c in range(now_number_people_max):
                                is_existed = False
                                for m, mci in enumerate(most_common_idxs):
                                    if c == most_common_idxs[m][0]:
                                        is_existed = True
                                        break
                                
                                if is_existed == False:
                                    # 存在しないインデックスだった場合、追加                 
                                    most_common_idxs.append( (c, 0) )

                        most_common_cnt = 0
                        for mci in most_common_idxs:
                            most_common_cnt += mci[1]
                        
                        if most_common_cnt > 0 and most_common_cnt_all > 0:
                            most_common_per_all = most_common_idxs[0][1] / most_common_cnt_all
                            most_common_per = most_common_idxs[0][1] / most_common_cnt
                            if most_common_idxs[0][0] >= 0 and most_common_per >= most_th and most_common_per_all >= all_most_th:
                                pd = pattern_datas[target_now_idxs[most_common_idxs[0][0]][2]]

                                if pd["x"][1] == 0 or pd["y"][1] == 0:
                                    # 首が取れてない場合、全身取れてないので、スルー
                                    continue

                                # 過去データと現在データのX差分を保持(一番取れるneckで)
                                pd_x_diff = abs(ppt_data["x"][1] - pd["x"][1])
                                pd_y_diff = abs(ppt_data["y"][1] - pd["y"][1])

                                # 差があまりにも大きい場合、スルー
                                if pd_x_diff / (dimensional_range["x"]["max"] - dimensional_range["x"]["min"]) > 0.1 or \
                                    pd_y_diff / (dimensional_range["y"]["max"] - dimensional_range["y"]["min"]) > 0.1:
                                    continue
                                
                                # 全体サイズから差分を引いて、値が大きいほど近くなるように
                                pd_diff = ((dimensional_range["x"]["max"] - dimensional_range["x"]["min"]) - pd_x_diff) \
                                            + ((dimensional_range["y"]["max"] - dimensional_range["y"]["min"]) - pd_y_diff)

                                # 再頻出INDEXが有効で、再頻出INDEXの出現数が全体の既定割合を超えていれば終了
                                all_most_common_per.append({"_eidx": _eidx, "most_common_per": most_common_per, "most_common_per_all": most_common_per_all, "most_common_idxs":most_common_idxs, "ppd": ppt_data, "ppd_conf_median": np.median(ppt_data["conf"]), "pd": pd, "pd_conf_avg": np.mean(pd["conf"]), "pd_conf_median": np.median(pd["conf"]), "pd_diff": pd_diff, "dimensional": dimensional})
                                break
                        
        # 全体平均の信頼度降順
        sorted_common_median = sorted(all_most_common_per, key=lambda x: (x["pd_diff"], x["pd_conf_median"], x["ppd_conf_median"], x["most_common_per_all"], x["most_common_per"]), reverse=True)
        
        logger.debug("_idx: %s, sorted: %s", _idx, sorted_common_median)

        # 信頼度降順の人物INDEX
        # 1番目から順に埋めていく
        for _eidx, smc in enumerate(sorted_common_median):
            # past_idx = pattern_datas[mci[_midx][0]]["pidx"]
            most_idx = target_now_idxs[smc["most_common_idxs"][0][0]][2]
            most_cnt = smc["most_common_idxs"][0][1]
            now_idx = smc["_eidx"] // 4
            # past_pidx = pattern_datas[past_idx]["past_pidx"]
            if now_idx not in sorted_idxs and sorted_idxs[most_idx] == -1 and most_cnt > 0:
                # まだ設定されていないINDEXで、入れようとしている箇所が空で、かつ信頼度平均がリミット以上の場合、設定
                sorted_idxs[most_idx] = now_idx
                logger.debug("sorted_idxs: %s", sorted_idxs)
            
            if -1 not in sorted_idxs:
                # 埋まったら終了
                break
        if -1 not in sorted_idxs:
            # 埋まったら終了
            break

    # 最終的に追加設定したいINDEX
    target_not_existed_idxs = [x for x in range(number_people_max) if x not in sorted_idxs]

    if len(target_not_existed_idxs) > 0:
        # まだ値がない場合、まだ埋まってないのを先頭から
        _nidx = 0
        for _eidx in range(number_people_max):
            if sorted_idxs[_eidx] < 0:
                sorted_idxs[_eidx] = target_not_existed_idxs[_nidx]
                _nidx += 1

    return sorted_idxs

def get_nearest_idxs(target_list, num, conf_list=None, th=0, dim_min=0, dim_max=0):
    """
    概要: リストからある値に最も近い値のINDEXを返却する関数
    @param target_list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値のINDEXの配列（同じ値がヒットした場合、すべて返す）
    """

    target_conf_list = []
    target_num = num

    # if dim_min != dim_max:
    #     # 範囲が指定されている場合、その範囲内の一定区分ごとの値を取得する
    #     target_denominator = 1 / ((dim_max - dim_min) / 200)
    #     target_num = round(num / target_denominator)

    if conf_list:
        for t, c in zip(target_list, conf_list):
            if c >= th:
                # 信頼度を満たしている場合のみリスト追加
                # if dim_min != dim_max:
                #     # 範囲が指定されている場合、その区分
                #     target_conf_list.append( round(t / target_denominator) )
                # else:
                target_conf_list.append(t)
            else:
                # 要件を満たせない場合、あり得ない値
                target_conf_list.append(999999999999)
    else:
        target_conf_list = target_list
    
    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(target_conf_list) - target_num).argmin()

    result_idxs = []

    for i, v in enumerate(target_conf_list):
        # INDEXが該当していて、かつ値が有効な場合、結果として追加
        if v == target_conf_list[idx] and v != 999999999999:
            # if conf_list and dim_min != dim_max and len(result_idxs) > 0:
            #     # 範囲が指定されており、かつ既にINDEXがある場合、信頼度の高い方を採用する
            #     if conf_list[result_idxs[-1]] < conf_list[i]:
            #         result_idxs[-1] = i
            #     elif conf_list[result_idxs[-1]] == conf_list[i]:
            #         # まったく同じなら追加
            #         result_idxs.append(i)
            # else:
            result_idxs.append(i)
    
    return result_idxs

def get_nearest_idx_ary(target_list, num_ary, conf_list=None, th=0):

    target_conf_list = []

    if conf_list:
        for t, c in zip(target_list, conf_list):
            # 信頼度を満たしている場合のみリスト追加
            if c >= th:
                target_conf_list.append(t)
            else:
                # 要件を満たせない場合、あり得ない値
                target_conf_list.append(np.full(len(target_list[0]), 999999999999))
    else:
        target_conf_list = target_list

    target_list2 = []
    for t in target_conf_list:
        # 現在との差を絶対値で求める
        target_list2.append(np.abs(np.asarray(t) - np.asarray(num_ary)))

    # logger.debug("num_ary: %s", num_ary)
    # logger.debug("target_list: %s", target_list)
    # logger.debug("target_list2: %s", target_list2)

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idxs = np.asarray(target_list2).argmin(axis=0)
    # logger.debug("np.asarray(target_list2).argmin(axis=0): %s", idxs)

    idx = np.argmax(np.bincount(idxs))
    # logger.debug("np.argmax(np.bincount(idxs)): %s", idx)

    result_idxs = []

    for i, v in enumerate(target_list2):
        if (v == target_list2[idx]).all():
            result_idxs.append(i)

    return result_idxs





# 通常INDEX
OPENPOSE_NORMAL = {"pattern": "normal", 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17}
# 左右反転させたINDEX
OPENPOSE_REVERSE_ALL = {"pattern": "reverse", 0:0, 1:1, 2:5, 3:6, 4:7, 5:2, 6:3, 7:4, 8:11, 9:12, 10:13, 11:8, 12:9, 13:10, 14:15, 15:14, 16:17, 17:16}
# 上半身のみ左右反転させたINDEX
OPENPOSE_REVERSE_UPPER = {"pattern": "up_reverse", 0:0, 1:1, 2:5, 3:6, 4:7, 5:2, 6:3, 7:4, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:15, 15:14, 16:17, 17:16}
# 下半身のみ左右反転させたINDEX
OPENPOSE_REVERSE_LOWER = {"pattern": "low_reverse", 0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:11, 9:12, 10:13, 11:8, 12:9, 13:10, 14:14, 15:15, 16:16, 17:17}
# 反転あり情報リスト
OP_PATTERNS = [OPENPOSE_NORMAL, OPENPOSE_REVERSE_ALL, OPENPOSE_REVERSE_UPPER, OPENPOSE_REVERSE_LOWER]

# ソートのための準備
# 人物データを、通常・全身反転・上半身反転・下半身反転の4パターンに分ける
def prepare_sort(_idx, number_people_max, data, pred_depth, pred_depth_support, pred_conf, pred_conf_support, frame_img, past_sorted_idxs, past_pattern_datas):
    pattern_datas = [{} for x in range(number_people_max * 4)]

    for _eidx, _pidx in enumerate(past_sorted_idxs):
        for op_idx, op_idx_data in enumerate(OP_PATTERNS):
            in_idx = (_eidx * 4) + op_idx

            # パターン別のデータ
            pattern_datas[in_idx] = {"eidx": _eidx, "pidx": _pidx, "sidx": _pidx, "in_idx": in_idx, "pattern": op_idx_data["pattern"], 
                "x": np.zeros(18), "y": np.zeros(18), "conf": np.zeros(18), "fill": [False for x in range(18)], "depth": np.zeros(18), 
                "depth_support": np.zeros(17), "conf_support": np.zeros(17), "color": [None for x in range(18)], "x_avg": 0, "conf_avg": 0}

            # 1人分の関節位置データ
            now_xyc = data["people"][_eidx]["pose_keypoints_2d"]

            for o in range(0,len(now_xyc),3):
                oidx = int(o/3)
                pattern_datas[in_idx]["x"][oidx] = now_xyc[op_idx_data[oidx]*3]
                pattern_datas[in_idx]["y"][oidx] = now_xyc[op_idx_data[oidx]*3+1]
                
                # 信頼度調整値(キーと値が合ってない反転系は信頼度を少し下げる)
                conf_tweak = 1.0 if oidx == op_idx_data[oidx] else 0.9
                pattern_datas[in_idx]["conf"][oidx] = now_xyc[op_idx_data[oidx]*3+2] * conf_tweak

                # 深度情報
                pattern_datas[in_idx]["depth"][oidx] = pred_depth[_eidx][op_idx_data[oidx]]

                # 色情報
                if 0 <= int(now_xyc[o+1]) < frame_img.shape[0] and 0 <= int(now_xyc[o]) < frame_img.shape[1]:
                    pattern_datas[in_idx]["color"][oidx] = frame_img[int(now_xyc[o+1]),int(now_xyc[o])]
                else:
                    pattern_datas[in_idx]["color"][oidx] = frame_img[0,0]
                
            # 深度補佐データ
            pattern_datas[in_idx]["depth_support"] = pred_depth_support[_eidx]
            pattern_datas[in_idx]["conf_support"] = pred_conf_support[_eidx]

            # 有効なXの平均値
            x_ary = np.array(pattern_datas[in_idx]["x"])
            pattern_datas[in_idx]["x_avg"] = np.mean(x_ary[x_ary != 0])
            if np.isnan(pattern_datas[in_idx]["x_avg"]):
                pattern_datas[in_idx]["x_avg"] = 0 

            logger.debug(pattern_datas[in_idx])

        logger.debug(pattern_datas)

    # 信頼度が著しく低いモノをクリア
    for _lidx, lpd in enumerate(pattern_datas):
        if np.mean(lpd["conf"]) < 0.25 and np.median(lpd["conf"]) < 0.1:
            if (np.mean(lpd["conf"]) != 0 or np.median(lpd["conf"]) != 0) and lpd["in_idx"] % 4 == 0:
                file_logger.info("※※{0:05d}F目 信頼度低情報除外: eidx: {1}, 平均値: {2}, 中央値: {3}".format( _idx, lpd["eidx"], np.mean(lpd["conf"]), np.median(lpd["conf"]) ))
            # パターン別の初期データで再設定
            pattern_datas[_lidx] = {"eidx": lpd["eidx"], "pidx": lpd["pidx"], "sidx": lpd["sidx"], "in_idx": lpd["in_idx"], "pattern": lpd["pattern"], 
                "x": np.zeros(18), "y": np.zeros(18), "conf": np.zeros(18), "fill": [False for x in range(18)], "depth": np.zeros(18), 
                "depth_support": np.zeros(17), "conf_support": np.zeros(17), "color": [frame_img[0,0] for x in range(18)], "x_avg": 0}

    # ほぼ同じ位置で信頼度が低いモノをクリア
    for _lidx, lpd in enumerate(pattern_datas):
        for _hidx, hpd in enumerate(pattern_datas):
            if lpd["pidx"] != hpd["pidx"] and abs(lpd["x"][1] - hpd["x"][1]) < frame_img.shape[1] / 30 and np.mean(hpd["conf"]) > np.mean(lpd["conf"]) and np.mean(lpd["conf"]) < 0.25:
                if (np.mean(lpd["conf"]) != 0 or np.median(lpd["conf"]) != 0) and lpd["in_idx"] % 4 == 0:
                    file_logger.info("※※{0:05d}F目 重複データ除外: eidx: {1}, 平均値: {2}, 中央値: {3}".format( _idx, lpd["eidx"], np.mean(lpd["conf"]), np.median(lpd["conf"]) ))
                # パターン別の初期データで再設定
                pattern_datas[_lidx] = {"eidx": lpd["eidx"], "pidx": lpd["pidx"], "sidx": lpd["sidx"], "in_idx": lpd["in_idx"], "pattern": lpd["pattern"], 
                    "x": np.zeros(18), "y": np.zeros(18), "conf": np.zeros(18), "fill": [False for x in range(18)], "depth": np.zeros(18), 
                    "depth_support": np.zeros(17), "conf_support": np.zeros(17), "color": [frame_img[0,0] for x in range(18)], "x_avg": 0}
                break

    # 体幹のサイズが規格外の場合、クリア（変なのとってきた場合）
    for _lidx, lpd in enumerate(pattern_datas):
        sizeover_cnt = 0
        for _pidx, ppd in enumerate(past_pattern_datas):
            lp_datas = [[lpd["x"][1], lpd["y"][1]],[lpd["x"][8], lpd["y"][8]],[lpd["x"][11], lpd["y"][11]]]
            all_diffs = np.expand_dims(lp_datas, axis=1) - np.expand_dims(lp_datas, axis=0)
            lp_distance = np.sqrt(np.sum(all_diffs ** 2, axis=-1)) 
            pp_datas = [[ppd["x"][1], ppd["y"][1]],[ppd["x"][8], ppd["y"][8]],[ppd["x"][11], ppd["y"][11]]]
            all_diffs = np.expand_dims(pp_datas, axis=1) - np.expand_dims(pp_datas, axis=0)
            pp_distance = np.sqrt(np.sum(all_diffs ** 2, axis=-1)) 

            if np.median(lpd["conf"]) < 0.4 and lp_distance[0,1] > 0 and lp_distance[0,1] < pp_distance[0,1] * 0.6 and lp_distance[0,2] > 0 and lp_distance[0,2] < pp_distance[0,2] * 0.6:
                # 前の6割未満の場合、カウント
                sizeover_cnt += 1

        if sizeover_cnt == len(past_pattern_datas):
            # どのデータでもサイズ外であった場合
            if (np.mean(lpd["conf"]) != 0 or np.median(lpd["conf"]) != 0) and lpd["in_idx"] % 4 == 0:
                file_logger.info("※※{0:05d}F目 体幹サイズ外情報除外: eidx: {1}, 平均値: {2}, 中央値: {3}".format( _idx, lpd["eidx"], np.mean(lpd["conf"]), np.median(lpd["conf"]) ))
            # パターン別の初期データで再設定
            pattern_datas[_lidx] = {"eidx": lpd["eidx"], "pidx": lpd["pidx"], "sidx": lpd["sidx"], "in_idx": lpd["in_idx"], "pattern": lpd["pattern"], 
                "x": np.zeros(18), "y": np.zeros(18), "conf": np.zeros(18), "fill": [False for x in range(18)], "depth": np.zeros(18), 
                "depth_support": np.zeros(17), "conf_support": np.zeros(17), "color": [frame_img[0,0] for x in range(18)], "x_avg": 0}

    return pattern_datas

# ソート順に合わせてデータを出力する
def output_sorted_data(_idx, _display_idx, number_people_max, sorted_idxs, now_pattern_datas, json_path, now_str, file_name, reverse_specific_dict, order_specific_dict):
    # 指定ありの場合、メッセージ追加
    if _idx in order_specific_dict:
        file_logger.warning("※※{0:05d}F目、順番指定 [{0}:{2}]".format( _idx, _display_idx, ','.join(map(str, order_specific_dict[_idx]))))

    display_sorted_idx = [x for x in range(number_people_max)]
    # for _eidx, _sidx in enumerate(sorted_idxs):
    #     now_sidx = get_nearest_idxs(sorted_idxs, _eidx)[0]
    #     display_sorted_idx[now_sidx] = _eidx

    # ソート順
    order_path = '{0}/{1}_{2}_depth/order.txt'.format(os.path.dirname(json_path), os.path.basename(json_path), now_str)
    # 追記モードで開く
    orderf = open(order_path, 'a')
    # 一行分を追記
    orderf.writelines("{0}\n".format(','.join(map(str, sorted_idxs))))
    orderf.close()

    display_nose_pos = {}
    for _eidx, npd in enumerate(now_pattern_datas):
        # データがある場合、そのデータ
        if not npd["fill"][1]:
            display_nose_pos[_eidx] = [npd["x"][1], npd["y"][1]]
        else:
            display_nose_pos[_eidx] = [0, 0]

        # インデックス対応分のディレクトリ作成
        idx_path = '{0}/{1}_{3}_idx{2:02d}/json/{4}'.format(os.path.dirname(json_path), os.path.basename(json_path), _eidx+1, now_str, file_name)
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        
        output_data = {"people": [{"pose_keypoints_2d": []}]}
        for (npd_x, npd_y, npd_conf, npd_fill) in zip(npd["x"], npd["y"], npd["conf"], npd["fill"]):
            if not npd_fill:
                # 過去補填以外のみ通常出力
                output_data["people"][0]["pose_keypoints_2d"].append(np.round(npd_x, 4))
                output_data["people"][0]["pose_keypoints_2d"].append(np.round(npd_y, 4))
                output_data["people"][0]["pose_keypoints_2d"].append(np.round(npd_conf, 6))
            else:
                # 過去補填情報は無視
                output_data["people"][0]["pose_keypoints_2d"].append(0.0)
                output_data["people"][0]["pose_keypoints_2d"].append(0.0)
                output_data["people"][0]["pose_keypoints_2d"].append(0.0)

        # 指定ありの場合、メッセージ追加
        reverse_specific_str = ""
        if _idx in reverse_specific_dict and _eidx in reverse_specific_dict[_idx]:
            reverse_specific_str = "【指定】"

        if npd["pattern"] == "reverse":
            file_logger.warning("※※{0:05d}F目 {2}番目の人物、全身反転 [{0}:{2},R]{3}".format( _idx, _display_idx, _eidx, reverse_specific_str))
        elif npd["pattern"] == "up_reverse":
            file_logger.warning("※※{0:05d}F目 {2}番目の人物、上半身反転 [{0}:{2},U]{3}".format( _idx, _display_idx, _eidx, reverse_specific_str))
        elif npd["pattern"] == "low_reverse":
            file_logger.warning("※※{0:05d}F目 {2}番目の人物、下半身反転 [{0}:{2},L]{3}".format( _idx, _display_idx, _eidx, reverse_specific_str))
        else:
            if len(reverse_specific_str) > 0:
                file_logger.warning("※※{0:05d}F目 {2}番目の人物、反転なし [{0}:{2},N]{3}".format( _idx, _display_idx, _eidx, reverse_specific_str))

        # 出力
        json.dump(output_data, open(idx_path,'w'), indent=4)

        # 深度データ
        depth_idx_path = '{0}/{1}_{3}_idx{2:02d}/depth.txt'.format(os.path.dirname(json_path), os.path.basename(json_path), _eidx+1, now_str)
        # 追記モードで開く
        depthf = open(depth_idx_path, 'a')
        # 一行分を追記
        depthf.write("{0}, {1},{2}\n".format(_display_idx, ','.join([ str(x) for x in npd["depth"] ]), ','.join([ str(x) for x in npd["depth_support"] ]) ))
        depthf.close()

        # 信頼度データ
        conf_idx_path = '{0}/{1}_{3}_idx{2:02d}/conf.txt'.format(os.path.dirname(json_path), os.path.basename(json_path), _eidx+1, now_str)
        # 追記モードで開く
        conff = open(conf_idx_path, 'a')
        # 一行分を追記
        conff.write("{0}, {1},{2}\n".format(_display_idx, ','.join([ str(x) for x in npd["conf"] ]), ','.join([ str(x) for x in npd["conf_support"] ]) ))
        conff.close()

    file_logger.warning("＊＊{0:05d}F目の出力順番: [{0}:{2}], 位置: {3}".format(_idx, _display_idx, ','.join(map(str, display_sorted_idx)), sorted(display_nose_pos.items()) ))

# 深度画像を保存する
def save_image(_idx, pred_image_ary, frame_img, number_people_max, sorted_idxs, now_pattern_datas, subdir, cnt, png_lib, scale, verbose):
    # 深度画像保存 -----------------------
    if level[verbose] <= logging.INFO and len(pred_image_ary[_idx]) > 0:
        # Plot result
        plt.cla()
        plt.clf()
        ii = plt.imshow(pred_image_ary[_idx], interpolation='nearest')
        plt.colorbar(ii)

        # 散布図のようにして、出力に使ったポイントを明示
        DEPTH_COLOR = ["#33FF33", "#3333FF", "#FFFFFF", "#FFFF33", "#FF33FF", "#33FFFF", "#00FF00", "#0000FF", "#666666", "#FFFF00", "#FF00FF", "#00FFFF"]
        for _eidx, npd in enumerate(now_pattern_datas):
            for (npd_x, npd_y, npd_fill) in zip(npd["x"], npd["y"], npd["fill"]):
                if not npd_fill:
                    plt.scatter(npd_x * scale, npd_y * scale, s=5, c=DEPTH_COLOR[_eidx])

        plotName = "{0}/depth_{1:012d}.png".format(subdir, cnt)
        plt.savefig(plotName)
        logger.debug("Save: {0}".format(plotName))

        png_lib.append(imageio.imread(plotName))

        plt.close()

# 前回データを読み込む
def load_sorted_idxs(json_path, now_str, number_people_max, prev_sorted_idxs, start_json_name, start_frame, pred_depth):

    past_sorted_idxs = [x for x in range(number_people_max)]
    sorted_idxs = [x for x in range(number_people_max)]

    past_pattern_datas = [{} for x in range(number_people_max)]
    now_pattern_datas = [{} for x in range(number_people_max)]

    cnt = start_frame
    for one_sorted_idxs in prev_sorted_idxs:
        # 前回データを引き継いだINDEXを再生成
        for _eidx, _sidx in enumerate(past_sorted_idxs):
            # Openpose推定結果の0F目の該当INDEXの人物があるINDEXを取得する
            sorted_idxs[_sidx] = get_nearest_idxs(one_sorted_idxs, _eidx)[0]

        if cnt - start_frame >= len(prev_sorted_idxs) - min(100, len(prev_sorted_idxs)):
            for _eidx, _sidx in enumerate(sorted_idxs):
                file_name = re.sub(r'\d{12}', "{0:012d}".format(cnt), start_json_name)
                # インデックス対応分のディレクトリ
                idx_path = '{0}/{1}_{3}_idx{2:02d}/json/{4}'.format(os.path.dirname(json_path), os.path.basename(json_path), _eidx+1, now_str, file_name)
                
                try:
                    data = json.load(open(idx_path))
                except Exception as e:
                    logger.warning("JSON読み込み失敗のため、空データ読み込み, %s %s", idx_path, e)
                    data = json.load(open("json/all_empty_keypoints.json"))

                # パターン別のデータ
                now_pattern_datas[_eidx] = {"eidx": _eidx, "pidx": _sidx, "sidx": _sidx, "in_idx": _eidx, "pattern": OPENPOSE_NORMAL["pattern"], 
                    "x": np.zeros(18), "y": np.zeros(18), "conf": np.zeros(18), "fill": [False for x in range(18)], "depth": np.zeros(18), 
                    "depth_support": np.zeros(17), "conf_support": np.zeros(17), "color": [np.array([0,0,0]) for x in range(18)], "x_avg": 0, "conf_avg": 0}

                # 1人分の関節位置データ
                now_xyc = data["people"][0]["pose_keypoints_2d"]

                for o in range(0,len(now_xyc),3):
                    oidx = int(o/3)
                    now_pattern_datas[_eidx]["x"][oidx] = now_xyc[OPENPOSE_NORMAL[oidx]*3]
                    now_pattern_datas[_eidx]["y"][oidx] = now_xyc[OPENPOSE_NORMAL[oidx]*3+1]
                    now_pattern_datas[_eidx]["conf"][oidx] = now_xyc[OPENPOSE_NORMAL[oidx]*3+2]
                    now_pattern_datas[_eidx]["depth"][oidx] = pred_depth[_eidx][OPENPOSE_NORMAL[oidx]]

            # 過去データ引継
            if cnt - start_frame > len(prev_sorted_idxs) - min(100, len(prev_sorted_idxs)):
                for _eidx, (npd, ppd) in enumerate(zip(now_pattern_datas, past_pattern_datas)):
                    logger.debug("npd: %s", npd)
                    if np.mean(npd["conf"]) < 0.2 and np.median(npd["conf"]) < 0.1:
                        for _didx in range(len(npd["conf"])):
                            for dimensional in ["x","y","depth"]:
                                # 信頼度が全くない場合、過去データで埋める（信頼度は下げる）
                                npd[dimensional][_didx] = ppd[dimensional][_didx]
                            npd["conf"][_didx] = ppd["conf"][_didx] * 0.9
                            npd["fill"][_didx] = True

        past_sorted_idxs = copy.deepcopy(sorted_idxs)
        past_pattern_datas = copy.deepcopy(now_pattern_datas)
        cnt += 1

    return sorted_idxs, now_pattern_datas