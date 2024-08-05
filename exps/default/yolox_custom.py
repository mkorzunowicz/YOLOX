#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.data_dir = "datasets/coco"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.num_classes = 81  # Number of classes in your dataset
        # self.device = "cpu"  # Ensure the device is set to CPU
        self.max_epoch = 150
        self.data_num_workers = 4
        self.eval_interval = 10
        self.exp_name = "yolox_s_poop"

# python tools/train.py -f exps/default/yolox_custom.py -d 1 -b 8 --fp16 -o -c pretrained/yolox_s.pth --cache ram
# python tools/train.py -f exps/default/yolox_custom.py -d 1 -b 16 --fp16 -o -c pretrained/yolox_s.pth --cache ram
# python tools/train.py -f exps/default/yolox_custom.py -d 1 -b 8 --fp16 -o -c YOLOX_outputs/yolox_s_poop/last_epoch_ckpt.pth --cache ram --resume
# python tools/train.py -f exps/default/yolox_custom.py -d 1 -b 16 --fp16 -o -c YOLOX_outputs/yolox_s_poop/last_epoch_ckpt.pth --cache ram --resume


# python tools/export_onnx.py --output-name yolox_tiny_poop.onnx -f .\exps\default\yolox_custom_tiny.py -c yolox_tiny_poop.pth -o 10 