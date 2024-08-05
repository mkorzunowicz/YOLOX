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
        self.num_classes = 1  # Number of classes in your dataset
        # self.device = "cpu"  # Ensure the device is set to CPU
        self.max_epoch = 700
        self.data_num_workers = 4
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.depth = 0.33
        self.width = 0.375
        self.eval_interval = 8
        self.enable_mixup = True        
        self.exp_name = "yolox_tiny_poop_cropped"
        self.save_history_ckpt = False

# python tools/train.py -f exps/default/yolox_custom_tiny.py -d 1 -b 8 --fp16 -o -c pretrained/yolox_tiny.pth --cache ram

# python tools/train.py -f exps/default/yolox_custom_tiny.py -d 1 -b 16 --fp16 -o -c pretrained/yolox_tiny.pth --cache ram
# python tools/train.py -f exps/default/yolox_custom_tiny.py -d 1 -b 20 --fp16 -o -c YOLOX_outputs/yolox_tiny_poop/last_epoch_ckpt.pth --cache ram --resume

# .\export-onnx.bat yolox_tiny_poop_cropped yolox_custom_tiny

# opset 10 for OpenVINO conversion
# python tools/export_onnx.py --output-name yolox_tiny_poop.onnx -f .\exps\default\yolox_custom_tiny.py -c yolox_tiny_poop.pth -o 10 

# python3 onnx_inference.py -m <ONNX_MODEL_PATH> -i <IMAGE_PATH> -o <OUTPUT_DIR> -s 0.3 --input_shape 640,640

# python .\demo\ONNXRuntime\onnx_inference.py -m .\yolox_tiny_poop.onnx -i .\assets\IMG_0416.JPEG -o .\demo_output\ -s 0.2 --input_shape 416,416