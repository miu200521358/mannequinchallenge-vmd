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
def exec(pred_depth_ary, pred_depth_support_ary, pred_conf_ary, pred_conf_support_ary, pred_image_ary, video_path, now_str, subdir, json_path, json_size, number_people_max, reverse_specific_dict, order_specific_dict, start_json_name, start_frame, end_frame_no, org_width, org_height, png_lib, scale, verbose):

    logger.warn("人物ソート開始 ---------------------------")

    # 前回情報
    past_pattern_datas = [{} for x in range(number_people_max)]
    past_sorted_idxs = [x for x in range(number_people_max)]

    cnt = 0
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        # 動画から1枚キャプチャして読み込む
        flag, frame = cap.read()  # Capture frame-by-frame

        # 深度推定のindex
        _idx = cnt - start_frame
        _display_idx = cnt

        # 開始フレームより前は飛ばす
        if start_frame > cnt:
            cnt += 1
            continue

        # 終わったフレームより後は飛ばす
        # 明示的に終わりが指定されている場合、その時も終了する
        if flag == False or cnt >= json_size + start_frame or (end_frame_no > 0 and _idx >= end_frame_no):
            break

        # 開始シーンのJSONデータを読み込む
        file_name = re.sub(r'\d{12}', "{0:012d}".format(cnt), start_json_name)
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

        # 前フレームと出来るだけ近い位置のINDEX順番を計算する
        sorted_idxs, now_pattern_datas, normal_pattern_datas = calc_sort_and_direction(_idx, reverse_specific_dict, order_specific_dict, number_people_max, past_pattern_datas, data, pred_depth_ary[_idx], pred_depth_support_ary[_idx], pred_conf_ary[_idx], pred_conf_support_ary[_idx], frame_img, past_sorted_idxs)

        # 出力する
        output_sorted_data(_idx, _display_idx, number_people_max, sorted_idxs, now_pattern_datas, json_path, now_str, file_name, reverse_specific_dict, order_specific_dict)

        # 画像保存
        save_image(_idx, pred_image_ary, frame_img, number_people_max, sorted_idxs, now_pattern_datas, subdir, cnt, png_lib, scale, verbose)

        # 今回ノーマル分を前回分に置き換え
        past_pattern_datas = normal_pattern_datas
        # 前回のソート順を保持
        past_sorted_idxs = sorted_idxs

        # インクリメント
        cnt += 1

# 前フレームと出来るだけ近い位置のINDEX順番を計算する
def calc_sort_and_direction(_idx, reverse_specific_dict, order_specific_dict, number_people_max, past_pattern_datas, data, pred_depth, pred_depth_support, pred_conf, pred_conf_support, frame_img, past_sorted_idxs):
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
                "x": [0 for x in range(18)], "y": [0 for x in range(18)], "conf": [0 for x in range(18)], "fill": [False for x in range(18)], 
                "depth": [0 for x in range(18)], "depth_support": [], "conf_support": [], "color": [0 for x in range(18)]}

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
        return sorted_idxs, now_pattern_datas, now_pattern_datas
    else:
        # ソートのための準備
        pattern_datas = prepare_sort(_idx, number_people_max, data, pred_depth, pred_depth_support, pred_conf, pred_conf_support, frame_img, past_sorted_idxs)

        # 順番が指定されている場合、適用
        order_sorted_idxs = None
        if _idx in order_specific_dict:
            order_sorted_idxs = order_specific_dict[_idx]

        # 再頻出INDEXを算出
        return calc_sort_and_direction_frame(_idx, reverse_specific_dict, number_people_max, past_pattern_datas, pattern_datas, order_sorted_idxs, past_sorted_idxs)

# ソート順と向きを求める
def calc_sort_and_direction_frame(_idx, reverse_specific_dict, number_people_max, past_pattern_datas, pattern_datas, order_sorted_idxs, past_sorted_idxs):
    # ソート結果
    sorted_idxs = [-1 for _ in range(number_people_max)]
    # sorted_in_idxs = [-1 for _ in range(number_people_max)]

    if number_people_max == 1:
        # 1人の場合はソート不要
        sorted_idxs = [0]
        sorted_in_idxs = [0]
    else:
        # 信頼度降順の人物推定INDEXを求める
        sorted_all_most_common_per_all = calc_sort_frame(_idx, number_people_max, past_pattern_datas, pattern_datas, 0.01, 0.65)

        logger.debug("_idx: %s, sorted_all_most_common_per_all: %s", _idx, sorted_all_most_common_per_all)

        # 信頼度降順の人物INDEX
        # 1番目から順に埋めていく
        for (sorted_most_commons, pd_conf_avg_limit) in [(sorted_all_most_common_per_all, 0.6), (sorted_all_most_common_per_all, 0.3), (sorted_all_most_common_per_all, 0)]:
            for _midx in range(number_people_max):
                for _eidx, smc in enumerate(sorted_most_commons):
                    # past_idx = past_pattern_datas[mci[_midx][0]]["pidx"]
                    most_idx = smc["most_common_idxs"][_midx][0]
                    past_idx = smc["_eidx"] // 4
                    # past_pidx = past_pattern_datas[past_idx]["past_pidx"]
                    if most_idx not in sorted_idxs and sorted_idxs[past_idx] == -1 and smc["pd_conf_avg"] >= pd_conf_avg_limit:
                        # まだ設定されていないINDEXで、入れようとしている箇所が空で、かつ信頼度平均がリミット以上の場合、設定
                        sorted_idxs[past_idx] = most_idx
                    
                    if -1 not in sorted_idxs:
                        # 埋まったら終了
                        break
                if -1 not in sorted_idxs:
                    # 埋まったら終了
                    break

        existed_idxs = {}
        not_existed_idxs = []
        for _sidx in range(len(sorted_idxs)):
            if sorted_idxs[_sidx] >= 0:
                # ちゃんと値が入っていたら辞書保持
                existed_idxs[sorted_idxs[_sidx]] = _sidx
            else:
                not_existed_idxs.append(_sidx)

        # 値がない場合、まだ埋まってないのを先頭から
        _nidx = 0
        for _eidx in range(number_people_max):
            if _eidx not in existed_idxs:
                sorted_idxs[not_existed_idxs[_nidx]] = _eidx
                _nidx += 1
        
        if order_sorted_idxs:
            # 順番が指定されている場合、適用
            copy_sorted_idxs = copy.deepcopy(sorted_idxs)
            for _eidx, osi in enumerate(order_sorted_idxs):
                copy_sorted_idxs[_eidx] = sorted_idxs[osi]
            sorted_idxs = copy_sorted_idxs

        logger.debug("_idx: %s, sorted_idxs: %s", _idx, sorted_idxs)

    # 人物INDEXが定まったところで、向きを再確認する
    now_pattern_datas = calc_direction_frame(_idx, number_people_max, past_pattern_datas, pattern_datas, sorted_idxs, 0.1, 0.5)
    # ノーマルパターン結果
    normal_pattern_datas = [[] for x in range(len(past_pattern_datas))]

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

        # 前回用にノーマルパターン保持
        normal_pattern_datas[_eidx] = pattern_datas[now_sidx*4]
    
    # ノーマルパターンに過去データ引継
    if _idx > 0:
        for (npd, ppd) in zip(normal_pattern_datas, past_pattern_datas):
            for _didx in range(len(npd["conf"])):
                if npd["conf"][_didx] <= 0:
                    for dimensional in ["x","y","depth"]:
                        # 信頼度が全くない場合、過去データで埋める（信頼度は下げる）
                        npd[dimensional][_didx] = ppd[dimensional][_didx]
                    npd["conf"][_didx] = ppd["conf"][_didx] * 0.8
                    npd["fill"][_didx] = True

    for npd in now_pattern_datas:
        logger.debug("_idx: %s, now_pattern_datas: pidx: %s, in_idx: %s, pattern: %s", _idx, npd["pidx"], npd["in_idx"], npd["pattern"])

    return sorted_idxs, now_pattern_datas, normal_pattern_datas


# 指定された方向（x, y, depth, color）に沿って、向きを計算する
def calc_direction_frame(_idx, number_people_max, past_pattern_datas, pattern_datas, sorted_idxs, th, most_th):
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

        for _didx, dimensional in enumerate(["xy","x","y"]):
            for _jidx in range(14):
                # 今回の該当関節データリスト
                now_per_joint_data = []
                now_per_joint_conf = []
                for _pidx, pt_data in enumerate(pattern_datas[now_sidx*4:now_sidx*4+4]):
                    # 該当辺の該当関節値を設定
                    if dimensional == "xy":
                        now_per_joint_data.append(np.array([pt_data["x"][_jidx],pt_data["y"][_jidx]]))
                    else:
                        now_per_joint_data.append(pt_data[dimensional][_jidx])
                    now_per_joint_conf.append(pt_data["conf"][_jidx])

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
                    # 足りてない場合、あり得ない値
                    now_nearest_idxs.append(-1)

            if len(now_nearest_idxs) > 0:
                most_common_idxs = Counter(now_nearest_idxs).most_common()
                # 件数の入ってるのだけ拾う
                most_common_idxs = [mci for mci in most_common_idxs if mci[0] >= 0]

                most_common_cnt = 0
                for mci in most_common_idxs:
                    most_common_cnt += mci[1]

                if most_common_cnt > 0:
                    most_common_per = most_common_idxs[0][1] / most_common_cnt
                    if most_common_idxs[0][0] >= 0 and most_common_per >= most_th - (_didx * 0.1):
                        # 再頻出INDEXが有効で、再頻出INDEXの出現数が全体の既定割合を超えていれば終了
                        break
            
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
        
        now_pattern_datas[_eidx] = pattern_datas[now_sidx*4+most_common_idxs[0][0]]
        now_pattern_datas[_eidx]["sidx"] = now_sidx

    return now_pattern_datas


# 指定された方向（x, y, depth, color）に沿って、ソート順を計算する
def calc_sort_frame(_idx, number_people_max, past_pattern_datas, pattern_datas, th, most_th):
    # 最頻出INDEXの割合(自身のINDEXも持つ)
    all_most_common_per = []
    all_past_per_joint_data = []

    # # 信頼度降順に並べ直す
    # conf_in_idx_list = sorted(list(map(lambda x: (x["in_idx"], np.mean(x["conf"])), pattern_datas)), key=lambda x: x[1], reverse=True)

    for _eidx, pd in enumerate(pattern_datas):
        pd_confs = []

        for _jidx in range(18):
            # 今回の該当関節の信頼度を保持
            pd_confs.append(pd["conf"][_jidx])

        for _didx, dimensional in enumerate(["x", "depth", "y", "depth_support", "color"]):
            # 直近INDEX
            now_nearest_idxs = []
            # 最頻出INDEX
            most_common_idxs = []
            # 再頻出INDEXの割合
            most_common_per = 0

            # 範囲を限定する（深度補佐は全部）
            jidx_rng = [0,1,2,3,5,6,8,9,10,11,12,13,1,8,11,1,8,11] if dimensional != "depth_support" else range(len(pd[dimensional]))
            for _jidx in jidx_rng:
                # 前回の該当関節データ
                past_per_joint_data = []
                past_per_joint_conf = []
                for ppt_data in past_pattern_datas:
                    past_per_joint_conf.append(ppt_data["conf"][_jidx])
                    # 信頼度が足りている場合、該当辺の該当関節値を設定
                    if dimensional == "xy":
                        past_per_joint_data.append(np.array([ppt_data["x"][_jidx],ppt_data["y"][_jidx]]))
                    elif dimensional == "xd":
                        past_per_joint_data.append(np.array([ppt_data["x"][_jidx],ppt_data["depth"][_jidx]]))
                    elif dimensional == "xyd":
                        past_per_joint_data.append(np.array([ppt_data["x"][_jidx],ppt_data["depth"][_jidx],ppt_data["y"][_jidx]]))
                    else:
                        past_per_joint_data.append(ppt_data[dimensional][_jidx])

                all_past_per_joint_data.append(past_per_joint_data)

                # 今回チェックしようとしている関節値
                if dimensional == "xy":
                    per_joint_value = np.array([pd["x"][_jidx],pd["y"][_jidx]])
                elif dimensional == "xd":
                    per_joint_value = np.array([pd["x"][_jidx],pd["depth"][_jidx]])
                elif dimensional == "xyd":
                    per_joint_value = np.array([pd["x"][_jidx],pd["depth"][_jidx],pd["y"][_jidx]])
                else:
                    per_joint_value = pd[dimensional][_jidx]

                if dimensional in ["xyd", "xd", "xy", "color"]:
                    if pd["conf"][_jidx] > 0:
                        # XY座標と色の場合は組合せでチェック
                        now_nearest_idxs.extend(get_nearest_idx_ary(past_per_joint_data, per_joint_value, past_per_joint_conf, th))
                    else:
                        # 足りてない場合、あり得ない値
                        now_nearest_idxs.append(-1)
                else:
                    if pd["conf"][_jidx] > 0:
                        # 信頼度が足りてる場合、直近のINDEXを取得
                        now_nearest_idxs.extend(get_nearest_idxs(past_per_joint_data, per_joint_value, past_per_joint_conf, th))
                    else:
                        # 足りてない場合、あり得ない値
                        now_nearest_idxs.append(-1)

            if len(now_nearest_idxs) > 0:
                most_common_idxs = Counter(now_nearest_idxs).most_common()
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
                
                most_common_cnt = 0
                for mci in most_common_idxs:
                    most_common_cnt += mci[1]

                if most_common_cnt > 0:
                    most_common_per = most_common_idxs[0][1] / most_common_cnt
                    if most_common_idxs[0][0] >= 0 and most_common_per >= most_th - (_didx * 0.1):
                        # 再頻出INDEXが有効で、再頻出INDEXの出現数が全体の既定割合を超えていれば終了
                        break
            
        all_most_common_per.append({"_eidx": _eidx, "most_common_per": most_common_per, "most_common_idxs":most_common_idxs, "pd": pd, "pd_conf_avg": np.mean(pd_confs)})

    # 4件全体の割合を計算
    for _eidx in range(number_people_max):
        sum_most_common_idxs = []
        for _pidx in range(4):
            for mci in all_most_common_per[_eidx*4+_pidx]["most_common_idxs"]:
                for n in range(mci[1]):
                    # 出現回数分登録
                    sum_most_common_idxs.append(mci[0])
        # 4件全体の出現頻出
        if len(sum_most_common_idxs) > 0:
            mci_sum_most_common_idxs = Counter(sum_most_common_idxs).most_common()
            for _pidx in range(4):
                # 全体出現平均を設定
                all_most_common_per[_eidx*4+_pidx]["avg_most_common_per"] = mci_sum_most_common_idxs[0][1] / len(sum_most_common_idxs)
        else:
            for _pidx in range(4):
                # 全体出現平均を設定
                all_most_common_per[_eidx*4+_pidx]["avg_most_common_per"] = 0

    # 全体平均の信頼度降順
    sorted_all_avg_most_common_per = sorted(all_most_common_per, key=lambda x: x["avg_most_common_per"], reverse=True)
    # 最終的な信頼度降順
    sorted_all_most_common_per = sorted(sorted_all_avg_most_common_per, key=lambda x: x["most_common_per"], reverse=True)

    return sorted_all_most_common_per

def get_nearest_idxs(target_list, num, conf_list=None, th=0):
    """
    概要: リストからある値に最も近い値のINDEXを返却する関数
    @param target_list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値のINDEXの配列（同じ値がヒットした場合、すべて返す）
    """

    target_conf_list = []

    if conf_list:
        for t, c in zip(target_list, conf_list):
            if c >= th:
                # 信頼度を満たしている場合のみリスト追加
                target_conf_list.append(t)
            else:
                # 要件を満たせない場合、あり得ない値
                target_conf_list.append(999999999999)
    else:
        target_conf_list = target_list

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(target_conf_list) - num).argmin()

    result_idxs = []

    for i, v in enumerate(target_conf_list):
        # INDEXが該当していて、かつ値が有効な場合、結果として追加
        if v == target_conf_list[idx] and v != 999999999999:
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

# ソートのための準備
# 人物データを、通常・全身反転・上半身反転・下半身反転の4パターンに分ける
def prepare_sort(_idx, number_people_max, data, pred_depth, pred_depth_support, pred_conf, pred_conf_support, frame_img, past_sorted_idxs):
    pattern_datas = [{} for x in range(number_people_max * 4)]

    for _eidx, _pidx in enumerate(past_sorted_idxs):
        for op_idx, op_idx_data in enumerate([OPENPOSE_NORMAL, OPENPOSE_REVERSE_ALL, OPENPOSE_REVERSE_UPPER, OPENPOSE_REVERSE_LOWER]):
            in_idx = (_pidx * 4) + op_idx

            # パターン別のデータ
            pattern_datas[in_idx] = {"eidx": _eidx, "pidx": _pidx, "sidx": _pidx, "in_idx": in_idx, "pattern": op_idx_data["pattern"], 
                "x": [0 for x in range(18)], "y": [0 for x in range(18)], "conf": [0 for x in range(18)], "fill": [0 for x in range(18)], 
                "depth": [0 for x in range(18)], "depth_support": [], "conf_support": [], "color": [0 for x in range(18)]}

            # 1人分の関節位置データ
            now_xyc = data["people"][_eidx]["pose_keypoints_2d"]

            for o in range(0,len(now_xyc),3):
                oidx = int(o/3)
                pattern_datas[in_idx]["x"][oidx] = now_xyc[op_idx_data[oidx]*3]
                pattern_datas[in_idx]["y"][oidx] = now_xyc[op_idx_data[oidx]*3+1]
                
                # 信頼度調整値(キーと値が合ってない反転系は信頼度を少し下げる)
                conf_tweak = 0.0 if oidx == op_idx_data[oidx] else -0.1
                pattern_datas[in_idx]["conf"][oidx] = now_xyc[op_idx_data[oidx]*3+2] + conf_tweak

                # 深度情報
                pattern_datas[in_idx]["depth"][oidx] = pred_depth[_eidx][op_idx_data[oidx]]

                # 色情報
                if 0 <= int(now_xyc[o+1]) < frame_img.shape[0] and 0 <= int(now_xyc[o]) < frame_img.shape[1]:
                    pattern_datas[in_idx]["color"][oidx] = frame_img[int(now_xyc[o+1]),int(now_xyc[o])]
                else:
                    pattern_datas[in_idx]["color"][oidx] = np.array([0,0,0])
                    
            # 深度補佐データ
            pattern_datas[in_idx]["depth_support"] = pred_depth_support[_eidx]
            pattern_datas[in_idx]["conf_support"] = pred_conf_support[_eidx]

            logger.debug(pattern_datas[in_idx])

        logger.debug(pattern_datas)

    return pattern_datas

# ソート順に合わせてデータを出力する
def output_sorted_data(_idx, _display_idx, number_people_max, sorted_idxs, now_pattern_datas, json_path, now_str, file_name, reverse_specific_dict, order_specific_dict):
    # 指定ありの場合、メッセージ追加
    if _idx in order_specific_dict:
        file_logger.warning("※※{0:05d}F目、順番指定 [{0}:{2}]".format( _idx, _display_idx, ','.join(map(str, order_specific_dict[_idx]))))

    display_sorted_idx = []
    display_nose_pos = {}
    for _eidx, _sidx in enumerate(sorted_idxs):
        npd = now_pattern_datas[_eidx]

        # データがある場合、そのデータ
        display_nose_pos[_eidx] = [npd["x"][1], npd["y"][1]]
        display_sorted_idx.append(_eidx)

        # インデックス対応分のディレクトリ作成
        idx_path = '{0}/{1}_{3}_idx{2:02d}/json/{4}'.format(os.path.dirname(json_path), os.path.basename(json_path), _eidx+1, now_str, file_name)
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        
        output_data = {"people": [{"pose_keypoints_2d": []}]}
        for (npd_x, npd_y, npd_conf, npd_fill) in zip(npd["x"], npd["y"], npd["conf"], npd["fill"]):
            if not npd_fill:
                # 過去補填以外のみ通常出力
                output_data["people"][0]["pose_keypoints_2d"].append(npd_x)
                output_data["people"][0]["pose_keypoints_2d"].append(npd_y)
                output_data["people"][0]["pose_keypoints_2d"].append(npd_conf)
            else:
                # 過去補填情報は無視
                output_data["people"][0]["pose_keypoints_2d"].append(0)
                output_data["people"][0]["pose_keypoints_2d"].append(0)
                output_data["people"][0]["pose_keypoints_2d"].append(0)

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

        # 深度データ
        conf_idx_path = '{0}/{1}_{3}_idx{2:02d}/conf.txt'.format(os.path.dirname(json_path), os.path.basename(json_path), _eidx+1, now_str)
        # 追記モードで開く
        conff = open(conf_idx_path, 'a')
        # 一行分を追記
        conff.write("{0}, {1},{2}\n".format(_display_idx, ','.join([ str(x) for x in npd["conf"] ]), ','.join([ str(x) for x in npd["conf_support"] ]) ))
        conff.close()

    file_logger.warning("＊＊{0:05d}F目の出力順番: [{0}:{2}], 位置: {3}".format(_idx, _display_idx, ','.join(map(str, sorted_idxs)), sorted(display_nose_pos.items()) ))

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
