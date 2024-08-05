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
        # self.num_classes = 81  # Number of classes in your dataset
        self.num_classes = 1  # Number of classes in your dataset
        # self.device = "cpu"  # Ensure the device is set to CPU
        self.max_epoch = 300
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.test_size = (416, 416)
        self.mosaic_prob = 0.5
        self.enable_mixup = True
        self.data_num_workers = 4
        self.eval_interval = 10
        self.exp_name = "yolox_nano_poop_cropped"
        self.save_history_ckpt = False
        


# python tools/train.py -f exps/default/yolox_custom_nano.py -d 1 -b 20 --fp16 -o -c pretrained/yolox_nano.pth --cache ram

# python tools/train.py -f exps/default/yolox_custom_nano.py -d 1 -b 8 --fp16 -o -c YOLOX_outputs/yolox_nano_poop_mixed/last_epoch_ckpt.pth --cache ram --resume


# .\export-onnx.bat yolox_nano_poop_cropped yolox_custom_nano

# python tools/export_onnx.py --output-name yolox_nano_poop_mixed.onnx -f .\exps\default\yolox_custom_nano.py -c yolox_nano_poop_mixed.pth -o 11 

# python tools/export_onnx.py --output-name yolox_nano_poop_all.onnx -f .\exps\default\yolox_custom_nano.py -c yolox_nano_poop_all.pth -o 10 