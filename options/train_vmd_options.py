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

from .base_options import BaseOptions


class TrainVmdOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int,
                                 default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument(
            '--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument(
            '--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_decay_epoch', type=int, default=8,
                                 help='# of epoch to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_policy', type=str, default='step',
                                 help='learning rate policy: lambda|step|plateau')

        self.parser.add_argument(
            '--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument(
            '--lr', type=float, default=0.0004, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument(
            '--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument(
            '--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                 help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data argumentation')

        self.isTrain = True

        self.parser.add_argument('--video_path', dest='video_path', help='input video', type=str)
        self.parser.add_argument('--json_path', dest='json_path', help='openpose json result path', type=str)
        self.parser.add_argument('--now', dest='now', help='now', default=None, type=str)
        self.parser.add_argument('--past_depth_path', dest='past_depth_path', help='past_depth_path', default=None, type=str)
        self.parser.add_argument('--interval', dest='interval', help='interval', type=int)
        self.parser.add_argument('--number_people_max', dest='number_people_max', help='number_people_max', type=int)
        self.parser.add_argument('--reverse_specific', dest='reverse_specific', help='reverse_specific', default="", type=str)
        self.parser.add_argument('--order_specific', dest='order_specific', help='order_specific', default="", type=str)
        self.parser.add_argument('--end_frame_no', dest='end_frame_no', help='end_frame_no', default=-1, type=int)
        self.parser.add_argument('--order_start_frame', dest='order_start_frame', help='order_start_frame', default=0, type=int)
        self.parser.add_argument('--avi_output', dest='avi_output', help='avi_output', default='yes', type=str)
        self.parser.add_argument('--verbose', dest='verbose', help='verbose', type=int)
