# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from options.train_vmd_options import TrainVmdOptions
from loaders import aligned_data_loader
from models import pix2pixdata_model

import cv2
import numpy as np
from PIL import Image
from skimage import exposure, transform
import imageio
import os
import logging
import argparse
import datetime
import shutil
import re
import json
import sys
import csv
import sort_people
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ファイル出力ログ用
file_logger = logging.getLogger("message")

level = {0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG}

# 入力値
WIDTH = 512

def predict_video(now_str, video_path, depth_path, past_depth_path, interval, json_path, number_people_max, reverse_specific_dict, order_specific_dict, is_avi_output, end_frame_no, order_start_frame, verbose, opt):
    # Windows用に追加
    torch.multiprocessing.freeze_support()

    # 深度用サブディレクトリ
    subdir = '{0}/depth'.format(depth_path)
    if os.path.exists(subdir):
        # 既にディレクトリがある場合、一旦削除
        shutil.rmtree(subdir)
    os.makedirs(subdir)

    # 深度用サブディレクトリ(disparity)
    depth_pred_dir_path = '{0}/depth_disparity'.format(subdir)
    if os.path.exists(depth_pred_dir_path):
        # 既にディレクトリがある場合、一旦削除
        shutil.rmtree(depth_pred_dir_path)
    os.makedirs(depth_pred_dir_path)

    # ファイル用ログの出力設定
    log_file_path = '{0}/message.log'.format(depth_path)
    logger.debug(log_file_path)
    file_logger.addHandler(logging.FileHandler(log_file_path))
    file_logger.warning("深度推定出力開始 now: %s ---------------------------", now_str)

    logger.addHandler(logging.FileHandler('{0}/{1}.log'.format(depth_path, __name__)))

    # 映像情報取得
    org_width, org_height, scale, width, height = get_video_info(video_path)

    logger.debug("org_width: %s, org_height: %s, scale: %s, width: %s, height: %s", org_width, org_height, scale, width, height)

    for pidx in range(number_people_max):
        # 人数分サイズデータ出力
        size_idx_path = '{0}/{1}_idx{2:02d}/size.txt'.format(os.path.dirname(
            json_path), now_str, pidx+1)
        os.makedirs(os.path.dirname(size_idx_path), exist_ok=True)
        sizef = open(size_idx_path, 'w')
        # 一行分を追記
        sizef.write("{0}\n".format(org_width))
        sizef.write("{0}\n".format(org_height))
        sizef.close()

    # フレーム開始INDEX取得
    start_json_name, start_frame, json_size = read_openpose_start_json(json_path)

    logger.info("number_people_max: %s, json_size: %s, start_frame: %s", number_people_max, json_size, start_frame)

    # 深度アニメーションGIF用
    png_lib = []
    # 人数分の深度データ
    # pred_depth_ary = [[[0 for z in range(18)] for y in range(number_people_max)] for x in range(json_size)]
    pred_depth_ary = np.zeros((json_size,number_people_max,18))
    # 人数分の深度データ（追加分）
    pred_depth_support_ary = np.zeros((json_size,number_people_max,17))
    # 人数分の信頼度データ
    pred_conf_ary = np.zeros((json_size,number_people_max,18))
    # 人数分の信頼度データ（追加分）
    pred_conf_support_ary = np.zeros((json_size,number_people_max,17))
    # 人数分の深度画像データ
    pred_image_ary = [[] for x in range(json_size) ]
    # 過去ソートデータ(pastはsort_peopleで使ってるのでprev)
    prev_sorted_idxs = []

    # 深度用ファイル
    depthf_path = '{0}/depth.txt'.format(depth_path)
    # 信頼度用ファイル
    conff_path = '{0}/conf.txt'.format(depth_path)
    # ソート順用ファイル
    orderf_path = '{0}/order.txt'.format(depth_path)

    past_depthf_path = None
    past_conff_path = None
    past_orderf_path = None
    if past_depth_path is not None:
        past_depthf_path = '{0}/depth.txt'.format(past_depth_path)
        past_conff_path = '{0}/conf.txt'.format(past_depth_path)
        past_orderf_path = '{0}/order.txt'.format(past_depth_path)

    logger.info("past_depthf_path: %s", past_depthf_path)
    logger.info("past_conff_path: %s", past_conff_path)
    logger.info("past_orderf_path: %s", past_orderf_path)

    if past_depthf_path is not None and os.path.exists(past_depthf_path) and  past_conff_path is not None and os.path.exists(past_conff_path) and \
        (order_start_frame == 0 or(order_start_frame > 0 and past_orderf_path is not None and os.path.exists(past_orderf_path))):
        # 深度ファイルが両方ある場合、それを読み込む

        # ----------------------
        pdepthf = open(past_depthf_path, 'r')

        fkey = -1
        fnum = 0
        # カンマ区切りなので、csvとして読み込む
        reader = csv.reader(pdepthf)

        for row in reader:
            fidx = int(row[0])
            if fkey != fidx:
                # キー値が異なる場合、インデックス取り直し
                fnum = 0

            pred_depth_ary[fidx][fnum] = np.array([float(x) for x in row[1:19]])
            pred_depth_support_ary[fidx][fnum] = np.array([float(x) for x in row[19:]])

            # 人物インデックス加算
            fnum += 1
            # キー保持
            fkey = fidx
        
        pdepthf.close()
        
        # 自分の深度情報ディレクトリにコピー
        shutil.copyfile(past_depthf_path, depthf_path)

        # ----------------------
        pconff = open(past_conff_path, 'r')

        fkey = -1
        fnum = 0
        # カンマ区切りなので、csvとして読み込む
        reader = csv.reader(pconff)

        for row in reader:
            fidx = int(row[0])
            if fkey != fidx:
                # キー値が異なる場合、インデックス取り直し
                fnum = 0

            pred_conf_ary[fidx][fnum] = np.array([float(x) for x in row[1:19]])
            pred_conf_support_ary[fidx][fnum] = np.array([float(x) for x in row[19:]])

            # 人物インデックス加算
            fnum += 1
            # キー保持
            fkey = fidx
        
        pconff.close()
        
        # 自分の信頼度情報ディレクトリにコピー
        shutil.copyfile(past_conff_path, conff_path)

        if order_start_frame > 0:
            # ソート開始フレームが指定されている場合、そこまで読み込む

            # ----------------------
            porderf = open(past_orderf_path, 'r')

            n = 0
            # カンマ区切りなので、csvとして読み込む
            reader = csv.reader(porderf)

            for row in reader:
                if (n < order_start_frame):
                    prev_sorted_idxs.append([int(x) for x in row])
                else:
                    break
                n += 1

            with open(orderf_path, 'w', newline='') as f:
                csv.writer(f).writerows(prev_sorted_idxs)
                            
            for _eidx in range(number_people_max):
                # INDEX別情報をまるっとコピー
                past_idx_path = past_depth_path.replace('depth', 'idx{0:02d}'.format(_eidx+1))
                idx_path = '{0}/{1}_{3}_idx{2:02d}'.format(os.path.dirname(json_path), os.path.basename(json_path), _eidx+1, now_str)
                # 既に出来ているので一旦削除
                shutil.rmtree(idx_path)
                shutil.copytree(past_idx_path, idx_path)

                # 深度データと信頼度データを必要行まで上書き
                depth_idx_path = '{0}/{1}_{3}_idx{2:02d}/depth.txt'.format(os.path.dirname(json_path), os.path.basename(json_path), _eidx+1, now_str)
                
                with open(depth_idx_path, 'r') as f:
                    lines = f.readlines()
                    lines = lines[:order_start_frame]

                with open(depth_idx_path, 'w') as f:
                    f.write(''.join(lines))

                conf_idx_path = '{0}/{1}_{3}_idx{2:02d}/conf.txt'.format(os.path.dirname(json_path), os.path.basename(json_path), _eidx+1, now_str)

                with open(conf_idx_path, 'r') as f:
                    lines = f.readlines()
                    lines = lines[:order_start_frame]

                with open(depth_idx_path, 'w') as f:
                    f.write(''.join(lines))
                
                logger.warning("過去データコピー idx: %s", _eidx+1)

            porderf.close()
    else:                
        # 動画を1枚ずつ画像に変換する
        in_idx = 0
        cnt = 0
        cap = cv2.VideoCapture(video_path)
        img_list = []
        while(cap.isOpened()):
            # 動画から1枚キャプチャして読み込む
            flag, frame = cap.read()  # Capture frame-by-frame

            # 深度推定のindex
            _idx = cnt - start_frame

            # 開始フレームより前は飛ばす
            if start_frame > cnt:
                cnt += 1
                continue
                
            # 終わったフレームより後は飛ばす
            # 明示的に終わりが指定されている場合、その時も終了する
            if flag == False or cnt >= json_size + start_frame or (end_frame_no > 0 and cnt >= end_frame_no):
                break

            # キャプチャ画像を読み込む
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2Lab))
            # 偏差
            img = np.float32(img)/255.0
            # サイズを小さくする
            img = transform.resize(img, (height, width))
            # コントラストをあげる
            img = exposure.equalize_adapthist(img)

            img_list.append(img)

            logger.debug("cnt: %s, _idx: %s, flag: %s, len(img_list): %s", cnt, _idx, flag, len(img_list))

            if (_idx > 0 and _idx % interval == 0 and _idx < json_size) or (cnt >= json_size + start_frame - 1):
                start = time.time()

                eval_num_threads = 2
                video_data_loader = aligned_data_loader.DAVISCaptureDataLoader(img_list, opt.batchSize)
                video_dataset = video_data_loader.load_data()
                logger.debug('========================= Video dataset #images = %d =========' %
                    len(video_data_loader))

                model = pix2pixdata_model.Pix2PixDataModel(opt)

                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                best_epoch = 0
                global_step = 0

                logger.debug(
                    '=================================  BEGIN VALIDATION ====================================='
                )

                logger.debug('TESTING ON VIDEO')

                model.switch_to_eval()
                
                # 深度ファイルを追記形式で開く
                depthf = open(depthf_path, 'a')
                conff = open(conff_path, 'a')
                
                for i, data in enumerate(video_dataset):
                    stacked_img = data[0]

                    # 1件だけ解析する
                    pred, pred_d_ref = model.run_and_save_DAVIS_one(stacked_img)

                    if level[verbose] < logging.INFO:
                        # 一旦出力する
                        np.savetxt('{0}/pred_{1:012d}.txt'.format(depth_pred_dir_path, in_idx), pred, fmt='%.5f')
                        np.savetxt('{0}/predref_{1:012d}.txt'.format(depth_pred_dir_path, in_idx), pred_d_ref, fmt='%.5f')

                    # logger.debug("pred: %s", _idx)
                    # logger.debug(pred)
                    # logger.debug("len(pred): %s", len(pred))
                    # logger.debug("len(pred): %s", len(pred))

                    # 深度解析後の画像サイズ
                    pred_width = len(pred)
                    pred_height = len(pred)
                    logger.debug("%s: pred_width: %s, pred_height: %s", in_idx, pred_width, pred_height)

                    # 該当シーンのJSONデータを読み込む
                    file_name = re.sub(r'\d{12}', "{0:012d}".format(in_idx + start_frame), start_json_name)
                    _file = os.path.join(json_path, file_name)

                    try:
                        data = json.load(open(_file))
                    except Exception as e:
                        logger.warning("JSON読み込み失敗のため、空データ読み込み, %s %s", _file, e)
                        data = json.load(open("json/all_empty_keypoints.json"))

                    for dpidx in range(len(data["people"]), number_people_max):
                        # 人数分のデータが無い場合、空データを読み込む
                        data["people"].append(json.load(open("json/one_keypoints.json")))
                    
                    # 深度解析後の画像サイズ
                    pred_width = len(pred[0])
                    pred_height = len(pred)
                    logger.debug("pred_width: %s, pred_height: %s", pred_width, pred_height)
                        
                    for dpidx in range(number_people_max):
                        logger.debug("dpidx: %s, len(data[people]): %s", dpidx, len(data["people"]))
                        for o in range(0,len(data["people"][dpidx]["pose_keypoints_2d"]),3):
                            oidx = int(o/3)
                            # オリジナルの画像サイズから、縮尺を取得
                            scale_org_x = data["people"][dpidx]["pose_keypoints_2d"][o] / org_width
                            scale_org_y = data["people"][dpidx]["pose_keypoints_2d"][o+1] / org_height
                            # logger.debug("scale_org_x: %s, scale_org_y: %s", scale_org_x, scale_org_y)

                            # 縮尺を展開して、深度解析後の画像サイズに合わせる
                            pred_x = int(pred_width * scale_org_x)
                            pred_y = int(pred_height * scale_org_y)

                            if 0 <= pred_y < len(pred) and 0 <= pred_x < len(pred[pred_y]):
                                # depths = pred[pred_y-3:pred_y+4,pred_x-3:pred_x+4].flatten()
                                # for x_shift in range(-3,4):
                                #     for y_shift in range(-3, 4):
                                #         if 0 <= pred_x + x_shift < pred_width and 0 <= pred_y + y_shift < pred_height:
                                #             depths.append(pred[pred_y + y_shift][pred_x + x_shift])

                                # 周辺3ピクセルで平均値を取る
                                pred_list = pred[pred_y-1:pred_y+2,pred_x-1:pred_x+2].flatten()
                                depth = 0 if len(pred_list) == 0 else np.mean(pred_list)

                                logger.debug("pred_x: %s, pred_y: %s, depth: %s", pred_x, pred_y, depth)

                                pred_depth_ary[in_idx][dpidx][oidx] = depth
                                pred_conf_ary[in_idx][dpidx][oidx] = data["people"][dpidx]["pose_keypoints_2d"][o+2]
                                pred_image_ary[in_idx] = pred
                            else:
                                # たまにデータが壊れていて、「9.62965e-35」のように取れてしまった場合の対策
                                pred_depth_ary[in_idx][dpidx][oidx] = 0
                                pred_conf_ary[in_idx][dpidx][oidx] = 0
                                pred_image_ary[in_idx] = pred


                        depth_support = np.zeros(17)
                        conf_support = np.zeros(17)
                        weights = [0.1,0.8,0.4,0.1,0.05,0.4,0.1,0.05,0.8,0.5,0.2,0.8,0.5,0.2,0.05,0.05,0.05,0.05]

                        # # Openposeで繋がっているライン上の深度を取得する
                        # for _didx, (start_idx, end_idx, start_w, end_w) in enumerate([(0,1,weights[0],weights[1]),(1,2,weights[1],weights[2]),(2,3,weights[2],weights[3]),(3,4,weights[3],weights[4]), \
                        #         (1,5,weights[1],weights[5]),(5,6,weights[5],weights[6]),(6,7,weights[6],weights[7]),(1,8,weights[1],weights[8]),(8,9,weights[8],weights[9]), \
                        #         (9,10,weights[9],weights[10]),(1,11,weights[1],weights[11]),(11,12,weights[11],weights[12]),(12,13,weights[12],weights[13]),(0,14,weights[0],weights[14]), \
                        #         (14,16,weights[14],weights[16]),(0,15,weights[0],weights[15]),(15,17,weights[15],weights[17])]):
                        #     # オリジナルの画像サイズから、縮尺を取得
                        #     start_scale_org_x = data["people"][dpidx]["pose_keypoints_2d"][start_idx*3] / org_width
                        #     start_scale_org_y = data["people"][dpidx]["pose_keypoints_2d"][start_idx*3+1] / org_height
                        #     start_conf = data["people"][dpidx]["pose_keypoints_2d"][start_idx*3+2]
                        #     # logger.debug("scale_org_x: %s, scale_org_y: %s", scale_org_x, scale_org_y)

                        #     # 縮尺を展開して、深度解析後の画像サイズに合わせる
                        #     start_pred_x = int(pred_width * start_scale_org_x)
                        #     start_pred_y = int(pred_height * start_scale_org_y)

                        #     # オリジナルの画像サイズから、縮尺を取得
                        #     end_scale_org_x = data["people"][dpidx]["pose_keypoints_2d"][end_idx*3] / org_width
                        #     end_scale_org_y = data["people"][dpidx]["pose_keypoints_2d"][end_idx*3+1] / org_height
                        #     end_conf = data["people"][dpidx]["pose_keypoints_2d"][end_idx*3+2]
                        #     # logger.debug("scale_org_x: %s, scale_org_y: %s", scale_org_x, scale_org_y)

                        #     # 縮尺を展開して、深度解析後の画像サイズに合わせる
                        #     end_pred_x = int(pred_width * end_scale_org_x)
                        #     end_pred_y = int(pred_height * end_scale_org_y)

                        #     per_depth_support = []
                        #     per_weight_support = []
                            
                        #     # # 深度範囲
                        #     # pred_x_rng = abs(start_pred_x - end_pred_x)
                        #     # pred_y_rng = abs(start_pred_y - end_pred_y)

                        #     # # 短い方の距離を単位とする
                        #     # pred_per = min(pred_x_rng, pred_y_rng)

                        #     # # 軸
                        #     # pred_x_line = np.linspace( min(start_pred_x, end_pred_x), max(start_pred_x, end_pred_x), pred_per + 1, dtype=int )
                        #     # pred_y_line = np.linspace( min(start_pred_y, end_pred_y), max(start_pred_y, end_pred_y), pred_per + 1, dtype=int )

                        #     # # 重み
                        #     # pred_weigths = np.linspace( start_w, end_w, pred_per + 1 )

                        #     # for (x, y, w) in zip(pred_x_line, pred_y_line, pred_weigths):
                        #     #     # 直線状の深度と重みを計算
                        #     #     per_depth_support.append(pred[y][x])
                        #     #     per_weight_support.append(w)

                        #     # # 重み付き平均を計算
                        #     # depth_support[_didx] = np.average(per_depth_support, weights=per_weight_support)
                        #     # conf_support[_didx] = np.mean([start_conf, end_conf])

                        pred_depth_support_ary[in_idx][dpidx] = depth_support
                        pred_conf_support_ary[in_idx][dpidx] = conf_support

                        # ------------------

                        # 深度データ
                        depthf.write("{0}, {1},{2}\n".format(in_idx, ','.join([ str(x) for x in pred_depth_ary[in_idx][dpidx] ]), ','.join([ str(x) for x in pred_depth_support_ary[in_idx][dpidx] ])))
                        # 信頼度データ
                        conff.write("{0}, {1},{2}\n".format(in_idx, ','.join([ str(x) for x in pred_conf_ary[in_idx][dpidx] ]), ','.join([ str(x) for x in pred_conf_support_ary[in_idx][dpidx] ])))

                    in_idx += 1

                # 一定間隔フレームおきにキャプチャした画像を深度推定する
                logger.warning("深度推定 idx: %s(%s) 処理: %s[sec]", _idx, cnt, time.time() - start)

                img_list = []

                # 一旦閉じる
                depthf.close()
                conff.close()

            cnt += 1

        cap.release()
        cv2.destroyAllWindows()

    # 基準深度で再計算
    # zファイルの方は基準深度再計算なし
    pred_depth_ary, pred_depth_support_ary = recalc_depth(pred_depth_ary, pred_depth_support_ary)

    # 人物ソート
    sort_people.exec(pred_depth_ary, pred_depth_support_ary, pred_conf_ary, pred_conf_support_ary, pred_image_ary, video_path, now_str, subdir, json_path, json_size, number_people_max, reverse_specific_dict, order_specific_dict, start_json_name, start_frame, end_frame_no, org_width, org_height, png_lib, scale, prev_sorted_idxs, verbose)

    if is_avi_output:
        # MMD用背景AVI出力
        outputAVI(video_path, depth_path, json_path, number_people_max, now_str, start_frame, end_frame_no, start_json_name, org_width, org_height)

    if level[verbose] <= logging.INFO and len(png_lib) > 0:
        # 終わったらGIF出力
        logger.info("creating Gif {0}/movie_depth.gif, please Wait!".format(os.path.dirname(depth_path)))
        imageio.mimsave('{0}/movie_depth.gif'.format(os.path.dirname(depth_path)), png_lib, fps=30)


# 基準深度で再計算
def recalc_depth(pred_depth_ary, pred_depth_support_ary):
    pred_depth_ary = np.array(pred_depth_ary)
    pred_depth_support_ary = np.array(pred_depth_support_ary)

    # 基準となる深度
    base_depth = np.median(pred_depth_ary[0][pred_depth_ary[0] != 0])

    # # 深度0が含まれていると狂うので、ループしてチェックしつつ合算
    # pred_sum = 0
    # pred_cnt = 0
    # for pred_joint in depth_ary[0][0]:
    #     if pred_joint > 0:
    #         pred_sum += pred_joint
    #         pred_cnt += 1

    # # 1人目の0F目の場合、基準深度として平均値を保存
    # base_depth = pred_sum / pred_cnt if pred_cnt > 0 else 0

    logger.info("基準深度取得: base_depth: %s", base_depth)   

    # 基準深度で入れ直し
    return np.where(pred_depth_ary != 0, (pred_depth_ary - base_depth) * 100, pred_depth_ary), np.where(pred_depth_support_ary != 0, (pred_depth_support_ary - base_depth) * 100, pred_depth_support_ary)

def outputAVI(video_path, depth_path, json_path, number_people_max, now_str, start_frame, end_frame_no, start_json_name, org_width, org_height):
    fourcc_names = ["I420"]

    if os.name == "nt":
        # Windows
        fourcc_names = ["IYUV"]

    # MMD用AVI出力 -----------------------------------------------------
    for fourcc_name in fourcc_names:
        try:
            # コーデックは実行環境によるので、自環境のMMDで確認できたfourccを総当たり
            # FIXME IYUVはAVI2なので、1GBしか読み込めない。ULRGは出力がULY0になってMMDで動かない。とりあえずIYUVを1GB以内で出力する
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            # 出力先AVIを設定する（MMD用に小さめ)
            out_path = '{0}/output_{1}.avi'.format(depth_path, fourcc_name)

            if os.name == "nt":
                # Windows
                op_avi_path = re.sub(r'json$', "openpose.avi", json_path)
            else:
                op_avi_path = re.sub(r'json/?', "openpose.avi", json_path)
            
            # 動画ファイルパスをAlphaPose用に
            op_avi_path = re.sub(r'sep-json/?', "AlphaPose_{0}".format(os.path.basename(video_path)), json_path)
            
            logger.info("op_avi_path: %s", op_avi_path)
            # Openopse結果AVIを読み込む
            cnt = 0
            cap = cv2.VideoCapture(op_avi_path)

            avi_width = int(org_width*0.32)
            avi_height = int(org_height*0.32)

            out = cv2.VideoWriter(out_path, fourcc, 30.0, (avi_width, avi_height))
            
            while(cap.isOpened()):
                # 動画から1枚キャプチャして読み込む
                flag, frame = cap.read()  # Capture frame-by-frame

                # 動画が終わっていたら終了
                if flag == False:
                    break

                # # 開始フレームより前は飛ばす
                # if start_frame > cnt:
                #     cnt += 1
                #     continue

                for pidx, lcolor, rcolor in zip(range(number_people_max) \
                        , [(51,255,51), (255,51,51), (255,255,255), (51,255,255), (255,51,255), (255,255,51), (0,127,0), (127,0,0), (102,102,102), (0,127,127), (127,0,127), (127,127,0)] \
                        , [(51,51,255), (51,51,255),   (51,51,255),  (51,51,255),  (51,51,255),  (51,51,255), (0,0,127), (0,0,127),     (0,0,127),   (0,0,127),   (0,0,127),   (0,0,127)]):
                    # 人物別に色を設定, colorはBGR形式
                    # 【00番目】 左:緑, 右: 赤
                    # 【01番目】 左:青, 右: 赤
                    # 【02番目】 左:白, 右: 赤
                    # 【03番目】 左:黄, 右: 赤
                    # 【04番目】 左:桃, 右: 赤
                    # 【05番目】 左:濃緑, 右: 赤
                    # 【06番目】 左:濃青, 右: 赤
                    # 【07番目】 左:灰色, 右: 赤
                    # 【08番目】 左:濃黄, 右: 赤
                    # 【09番目】 左:濃桃, 右: 赤
                    idx_json_path = '{0}/{1}_idx{2:02d}/json/{3}'.format(os.path.dirname(json_path), now_str, pidx+1, re.sub(r'\d{12}', "{0:012d}".format(cnt + start_frame), start_json_name))
                    logger.warning("pidx: %s, color: %s, idx_json_path: %s", pidx, color, idx_json_path)

                    if os.path.isfile(idx_json_path):
                        data = json.load(open(idx_json_path))

                        for o in range(0,len(data["people"][0]["pose_keypoints_2d"]),3):
                            # 左右で色を分ける
                            color = rcolor if int(o/3) in [2,3,4,8,9,10,14,16] else lcolor

                            if data["people"][0]["pose_keypoints_2d"][o+2] > 0:
                                # 少しでも信頼度がある場合出力
                                # logger.debug("x: %s, y: %s", data["people"][0]["pose_keypoints_2d"][o], data["people"][0]["pose_keypoints_2d"][o+1])
                                # cv2.drawMarker( frame, (int(data["people"][0]["pose_keypoints_2d"][o]+5), int(data["people"][0]["pose_keypoints_2d"][o+1]+5)), color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=10)
                                # 座標のXY位置に点を置く。原点が左上なので、ちょっとずらす
                                cv2.circle( frame, (int(data["people"][0]["pose_keypoints_2d"][o]+1), int(data["people"][0]["pose_keypoints_2d"][o+1]+1)), 5, color, thickness=-1)
                
                # 縮小
                output_frame = cv2.resize(frame, (avi_width, avi_height))

                # 全人物が終わったら出力
                out.write(output_frame)

                # インクリメント
                cnt += 1

                if end_frame_no > 0 and cnt >= end_frame_no:
                    break

            logger.warning('MMD用AVI: {0}'.format(out_path))

            # 出力に成功したら終了
            # break
        except Exception as e:
            logger.warning("MMD用AVI出力失敗: %s, %s", fourcc_name, e)

        finally:
            # 終わったら開放
            cap.release()
            out.release()
            cv2.destroyAllWindows()


# Openposeの結果jsonの最初を読み込む
def read_openpose_start_json(json_path):
    # openpose output format:
    # [x1,y1,c1,x2,y2,c2,...]
    # ignore confidence score, take x and y [x1,y1,x2,y2,...]

    # load json files
    json_files = os.listdir(json_path)
    # check for other file types
    json_files = sorted([filename for filename in json_files if filename.endswith(".json")])

    # jsonのファイル数が読み取り対象フレーム数
    json_size = len(json_files)
    # 開始フレーム
    start_frame = 0
    # 開始フラグ
    is_started = False
    
    for file_name in json_files:
        logger.debug("reading {0}".format(file_name))
        _file = os.path.join(json_path, file_name)

        if not os.path.isfile(_file):
            if is_started:
                raise Exception("No file found!!, {0}".format(_file))
            else:
                continue

        try:
            data = json.load(open(_file))
        except Exception as e:
            logger.warning("JSON読み込み失敗のため、空データ読み込み, %s %s", _file, e)
            data = json.load(open("tensorflow/json/all_empty_keypoints.json"))

        # 12桁の数字文字列から、フレームINDEX取得
        frame_idx = int(re.findall("(\d{12})", file_name)[0])
        
        if (frame_idx <= 0 or is_started == False) and len(data["people"]) > 0:
            # 何らかの人物情報が入っている場合に開始
            # 開始したらフラグを立てる
            is_started = True
            # 開始フレームインデックス保持
            start_frame = frame_idx

            # ループ終了
            break

    logger.warning("開始フレーム番号: %s", start_frame)

    return json_files[0], start_frame, json_size


# 映像解析縮尺情報
def get_video_info(video_path):
    # 映像サイズを取得する
    cap = cv2.VideoCapture(video_path)
    # 幅
    org_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # 高さ
    org_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.debug("width: {0}, height: {1}".format(org_width, org_height))

    # 縮小倍率
    scale = WIDTH / org_width
    logger.debug("scale: {0}".format(scale))
    
    height = int(org_height * scale)
    logger.debug("width: {0}, height: {1}".format(WIDTH, height))

    return org_width, org_height, scale, WIDTH, height



def main():
    opt = TrainVmdOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

    logger.setLevel(level[opt.verbose])

    # 間隔は1以上の整数
    interval = opt.interval if opt.interval > 0 else 1

    # AVI出力有無
    is_avi_output = False if opt.avi_output == 'no' else True

    # 出力用日付
    if opt.now is None:
        now_str = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    else:
        now_str = opt.now

    print("json_path: %s" % opt.json_path)
    print("json_path.basename: %s" % os.path.dirname(opt.json_path))

    sep_json_path = '{0}/sep-json'.format(os.path.dirname(opt.json_path))

    # 日付+depthディレクトリ作成
    depth_path = '{0}/{1}_depth'.format(os.path.dirname(opt.json_path), now_str)
    os.makedirs(depth_path, exist_ok=True)

    # 過去深度ディレクトリ
    past_depth_path = opt.past_depth_path if opt.past_depth_path is not None and len(opt.past_depth_path) > 0 else None

    # 強制反転指定用辞書作成
    reverse_specific_dict = {}
    if opt.reverse_specific is not None and len(opt.reverse_specific) > 0:
        for frame in opt.reverse_specific.split(']'):
            # 終わりカッコで区切る
            if ':' in frame:
                # コロンでフレーム番号と人物を区切る
                frames = frame.lstrip("[").split(':')[0]
                # logger.debug("frame: %s", frame)
                # logger.debug("frames: %s", frames)
                # logger.debug("frame.split(':')[1]: %s", frame.split(':')[1])
                # logger.debug("frame.split(':')[1].split(','): %s", frame.split(':')[1].split(','))
                if '-' in frames:
                    frange = frames.split('-')
                    if len(frange) >= 2 and frange[0].isdecimal() and frange[1].isdecimal():
                        for f in range(int(frange[0]), int(frange[1])+1):
                            # 指定フレームの辞書作成
                            if f not in reverse_specific_dict:
                                reverse_specific_dict[f] = {}

                            # 人物INDEXとその反転内容を保持
                            reverse_specific_dict[f][int(frame.split(':')[1].split(',')[0])] = frame.split(':')[1].split(',')[1]
                else:        
                    if frames not in reverse_specific_dict:
                        # 該当フレームがまだない場合、作成
                        reverse_specific_dict[int(frames)] = {}

                    # 人物INDEXとその反転内容を保持
                    reverse_specific_dict[int(frames)][int(frame.split(':')[1].split(',')[0])] = frame.split(':')[1].split(',')[1]

        logger.warning("反転指定リスト: %s", reverse_specific_dict)

        paramf = open( depth_path + "/reverse_specific.txt", 'w')
        paramf.write(opt.reverse_specific)
        paramf.close()

    # 強制順番指定用辞書作成
    order_specific_dict = {}
    if opt.order_specific is not None and len(opt.order_specific) > 0:
        for frame in opt.order_specific.split(']'):
            # 終わりカッコで区切る
            if ':' in frame:
                # コロンでフレーム番号と人物を区切る
                frames = frame.lstrip("[").split(':')[0]
                logger.debug("frames: %s", frames)
                if '-' in frames:
                    frange = frames.split('-')
                    if len(frange) >= 2 and frange[0].isdecimal() and frange[1].isdecimal():
                        for f in range(int(frange[0]), int(frange[1])+1):
                            # 指定フレームの辞書作成
                            order_specific_dict[f] = []

                            for person_idx in frame.split(':')[1].split(','):
                                if int(person_idx) in order_specific_dict[int(frames)]:
                                    logger.error("×順番指定リストに同じINDEXが指定されています。処理を中断します。 %s", frame)
                                    return False
                                order_specific_dict[f].append(int(person_idx))
                else:        
                    if frames not in order_specific_dict:
                        # 該当フレームがまだない場合、作成
                        order_specific_dict[int(frames)] = []

                        for person_idx in frame.split(':')[1].split(','):
                            if int(person_idx) in order_specific_dict[int(frames)]:
                                logger.error("×順番指定リストに同じINDEXが指定されています。処理を中断します。 %s", frame)
                                return False
                            order_specific_dict[int(frames)].append(int(person_idx))

        logger.warning("順番指定リスト: %s", order_specific_dict)

        paramf = open( depth_path + "/order_specific.txt", 'w')
        paramf.write(opt.order_specific)
        paramf.close()

    # Predict the image
    predict_video(now_str, opt.video_path, depth_path, past_depth_path, interval, sep_json_path, opt.number_people_max, reverse_specific_dict, order_specific_dict, is_avi_output, opt.end_frame_no, opt.order_start_frame, opt.verbose, opt)

    logger.debug("Done!!")
    logger.debug("深度推定結果: {0}".format(depth_path +'/depth.txt'))

if __name__ == '__main__':
    main()
