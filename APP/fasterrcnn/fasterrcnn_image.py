# -*- coding: utf-8 -*-
import os
import torch
import torchvision
import pathlib
import argparse
import socket
import cv2
import numpy as np
import h5py
import sys
import random
import datetime
from PIL import Image
from . import coco_names
from ..util import utils


# def get_args():
#     parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')
#     parser.add_argument('--image_path', type=str,
#                         default='D:/1.jpg',
#                         help='image path')
#     parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
#     parser.add_argument('--dataset', default='coco', help='model')
#     parser.add_argument('--score', type=float, default=0.8, help='objectness score threshold')
#     args = parser.parse_args()
#
#     return args


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return (b, g, r)


# args = get_args()

num_classes = 91
names = coco_names.names
# if args.dataset == 'coco':
#     num_classes = 91
#     names = coco_names.names

print("Creating model")
model = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=num_classes, pretrained=True)
model = model.cuda()
model.eval()


class FasterrcnnImage(object):
    def __init__(self, input_image_name=None, input_images_path=None, output_images_path=None,
                 partition=None, server_ip=None):
        if (input_image_name is not None) and (input_images_path is not None) and (output_images_path is not None) \
                and (partition is not None) and (server_ip is not None):
            self.TEST_IMAGE_NAME = input_image_name
            self.TEST_IMAGE_PATHS = input_images_path
            self.OUTPUT_IMAGE_PATH = output_images_path

        else:
            self.TEST_IMAGE_NAME = ''
            self.TEST_IMAGE_PATHS = ''
            self.OUTPUT_IMAGE_PATH = './results/BlurBody/'

        if not os.path.exists(self.OUTPUT_IMAGE_PATH):
            os.mkdir(self.OUTPUT_IMAGE_PATH)

    def run_fasterrcnn_image(self):
        print('Running: ', self.TEST_IMAGE_PATHS)

        input = []
        src_img = cv2.imread(str(self.TEST_IMAGE_PATHS))
        img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        # img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()

        input.append(img_tensor)

        out = model(input, partition=0)

        print('Finish: ', self.TEST_IMAGE_PATHS)

        boxes = out[0]['boxes']
        labels = out[0]['labels']
        scores = out[0]['scores']

        for idx in range(boxes.shape[0]):
            if scores[idx] >= 0.8:
                x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
                name = names.get(str(labels[idx].item()))
                # cv2.rectangle(img,(x1,y1),(x2,y2),colors[labels[idx].item()],thickness=2)
                cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
                cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

        # cv2.imshow('result', src_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(self.OUTPUT_IMAGE_PATH, self.TEST_IMAGE_NAME), src_img)
        print("Save result image-" + self.TEST_IMAGE_PATHS)

        torch.cuda.empty_cache()
