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
from skimage import transform
import imageio
import os.path

BATCH_SIZE = 1

def run():
    # Windows用に追加
    torch.multiprocessing.freeze_support()

    opt = TrainVmdOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

    video_path = "E:/MMD/MikuMikuDance_v926x64/Work/201805_auto/04/yoiyoi/yoiyoi_3388-3530.mp4"

    width = 512
    height = 288

    # 動画を1枚ずつ画像に変換する
    cnt = 0
    cap = cv2.VideoCapture(video_path)
    img_list = []
    output_datas = []
    while(cap.isOpened()):
        # 動画から1枚キャプチャして読み込む
        flag, frame = cap.read()  # Capture frame-by-frame

        if flag == True:
            # キャプチャ画像を読み込む
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2Lab))
            img = np.float32(img)/255.0
            img = transform.resize(img, (height, width))

            img_list.append(img)

        if flag == False or len(img_list) >= 30:
            eval_num_threads = 2
            video_data_loader = aligned_data_loader.DAVISCaptureDataLoader(img_list, BATCH_SIZE)
            video_dataset = video_data_loader.load_data()
            print('========================= Video dataset #images = %d =========' %
                len(video_data_loader))

            model = pix2pixdata_model.Pix2PixDataModel(opt)

            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            best_epoch = 0
            global_step = 0

            print(
                '=================================  BEGIN VALIDATION ====================================='
            )

            print('TESTING ON VIDEO')

            model.switch_to_eval()
            save_path = 'test_data/viz_predictions/'
            print('save_path %s' % save_path)

            for i, data in enumerate(video_dataset):
                print(i)
                idx = (cnt - 30) + i
                stacked_img = data[0]
                targets = {}
                targets['img_1_path'] = {}
                targets['img_1_path'][0] = video_path.replace(".mp4", "_{0:07d}.jpg".format(idx))
                model.run_and_save_DAVIS(stacked_img, targets, save_path, output_datas)

        if flag == False:
            break
            
        if len(img_list) >= 30:
            img_list = []

        cnt += 1

    # 終わったらGIF出力
    # logger.info("creating Gif {0}/movie_depth.gif, please Wait!".format(os.path.dirname(save_path)))
    imageio.mimsave('{0}/movie_depth.gif'.format(os.path.dirname(save_path)), output_datas, fps=30)


if __name__ == '__main__':
    run()
