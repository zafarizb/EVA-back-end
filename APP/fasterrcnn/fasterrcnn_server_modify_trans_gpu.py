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
import utils
from collections import OrderedDict
from matplotlib import pyplot as plt
from PIL import Image
import datetime
sys.path.append('./')
import coco_names


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')
    # parser.add_argument('--model_path', type=str, default='./result/model_19.pth', help='model path')
    # parser.add_argument('--image_path', type=str, default='/home/k8s/zhuangbw/models-master/research/object_detection/test_images/BlurBody/img/0001.jpg', help='image path')
    parser.add_argument('--image_path', type=str,
                        default='D:/1.jpg',
                        help='image path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='model')
    parser.add_argument('--score', type=float, default=0.8, help='objectness score threshold')
    args = parser.parse_args()

    return args


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    return (b, g, r)


args = get_args()
num_classes = 91
names = coco_names.names
if args.dataset == 'coco':
    num_classes = 91
    names = coco_names.names

print("Creating model")
model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=True)
model = model.cuda()
model.eval()


class FasterrcnnServer(object):
    def __init__(self, input_images_path=None, path_groundtruth_rect=None, ground_truth_class=None,
                 output_images_path=None, partition=None, server_ip=None, is_trainer=None):
        if (input_images_path is not None) and (path_groundtruth_rect is not None) \
                and (ground_truth_class is not None) and (output_images_path is not None) \
                and (partition is not None) and (server_ip is not None):
            self.PATH_TO_TEST_IMAGES_DIR = input_images_path
            self.TEST_IMAGE_PATHS = sorted(list(self.PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
            self.PATH_TO_GROUNDTRUTH_RECT = path_groundtruth_rect
            self.GROUND_TRUTH_CLASS = ground_truth_class
            self.OUTPUT_IMAGE_PATH = output_images_path
            self.partition = partition
            self.server_ip = server_ip
            self.is_trainer = is_trainer
        else:
            self.PATH_TO_TEST_IMAGES_DIR = pathlib.Path('/home/k8s/zhuangbw/Visual Tracker Benchmark/origin/BlurBody/img')
            self.TEST_IMAGE_PATHS = sorted(list(self.PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
            self.PATH_TO_GROUNDTRUTH_RECT = '/home/k8s/zhuangbw/Visual Tracker Benchmark/origin/BlurBody/groundtruth_rect.txt'
            self.GROUND_TRUTH_CLASS = 'person'
            self.OUTPUT_IMAGE_PATH = './results/BlurBody/'
            self.partition = 4
            self.server_ip = 'localhost'
            self.is_trainer = True

        if not os.path.exists(self.OUTPUT_IMAGE_PATH):
            os.mkdir(self.OUTPUT_IMAGE_PATH)

        if self.is_trainer:
            self.recv_edge_out_name = "recv_edge_out_trainer"
        else:
            self.recv_edge_out_name = "recv_edge_out_tester"

        self.edge_out_size = 0
        self.server_time = []
        self.det_results = {}

    def run_fasterrcnn_server(self, edge_outs):
        self.TEST_IMAGE_PATHS = sorted(list(self.PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
        for j in range(len(self.TEST_IMAGE_PATHS)):
            input = []
            src_img = cv2.imread(str(self.TEST_IMAGE_PATHS[j]))
            img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            # img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
            img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()

            input.append(img_tensor)
            if self.partition == 1:  # partition at resnet Layer1
                start = datetime.datetime.now()
                for k, _ in edge_outs[j].items():
                    edge_outs[j][k] = edge_outs[j][k].cuda()
                out = model(input, partition=1, node='server', edge_out=edge_outs[j])
                end = datetime.datetime.now()

                print('Finished fasterrcnn')

                timestr = str(end - start)[5:]
                self.server_time.append(float(timestr))

            elif self.partition == 2:  # partition at resnet Layer2
                start = datetime.datetime.now()
                for k, _ in edge_outs[j].items():
                    edge_outs[j][k] = edge_outs[j][k].cuda()
                out = model(input, partition=2, node='server', edge_out=edge_outs[j])
                end = datetime.datetime.now()

                print('Finished fasterrcnn')

                timestr = str(end - start)[5:]
                self.server_time.append(float(timestr))

            elif self.partition == 3:  # partition at resnet Layer3
                start = datetime.datetime.now()
                for k, _ in edge_outs[j].items():
                    edge_outs[j][k] = edge_outs[j][k].cuda()
                out = model(input, partition=3, node='server', edge_out=edge_outs[j])
                end = datetime.datetime.now()

                print('Finished fasterrcnn')

                timestr = str(end - start)[5:]
                self.server_time.append(float(timestr))

            elif self.partition == 4:  # partition at resnet Layer4
                start = datetime.datetime.now()
                for k, _ in edge_outs[j].items():
                    edge_outs[j][k] = edge_outs[j][k].cuda()
                out = model(input, partition=4, node='server', edge_out=edge_outs[j])
                end = datetime.datetime.now()

                print('Finished fasterrcnn')

                timestr = str(end - start)[5:]
                self.server_time.append(float(timestr))

            boxes = out[0]['boxes']
            labels = out[0]['labels']
            scores = out[0]['scores']

            self.det_results[self.TEST_IMAGE_PATHS[j].name] = {"det_boxes": boxes.detach().cpu().numpy(),
                                                          "det_labels": np.array(labels.cpu()),
                                                          "det_scores": scores.detach().cpu().numpy()}

            for idx in range(boxes.shape[0]):
                if scores[idx] >= args.score:
                    # det_results[test_images_path[j].name]["det_boxes"].append(boxes[idx].detach().numpy())
                    # det_results[test_images_path[j].name]["det_labels"].append(labels[idx].detach().numpy())
                    # det_results[test_images_path[j].name]["det_scores"].append(scores[idx].detach().numpy())
                    x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
                    name = names.get(str(labels[idx].item()))
                    # cv2.rectangle(img,(x1,y1),(x2,y2),colors[labels[idx].item()],thickness=2)
                    cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
                    cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

            # cv2.imshow('result', src_img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            cv2.imwrite(os.path.join(self.OUTPUT_IMAGE_PATH, self.TEST_IMAGE_PATHS[j].name), src_img)
            print("Save result image-" + self.TEST_IMAGE_PATHS[j].name)

    def eval(self):
        iou_for_videos = []
        scores_for_videos = []

        im = Image.open(self.TEST_IMAGE_PATHS[0])
        IMAGE_HEIGHT = im.height
        IMAGE_WIDTH = im.width

        gt_file = open(self.PATH_TO_GROUNDTRUTH_RECT, 'r')
        text = gt_file.read()
        gt_file.close()
        text = text.replace(",", " ")
        gt_file = open(self.PATH_TO_GROUNDTRUTH_RECT, 'w')
        gt_file.write(text)
        gt_file.close()

        gt_rect = np.loadtxt(self.PATH_TO_GROUNDTRUTH_RECT)
        normalize_gt_rect = []
        for gt in gt_rect:
            normalize_gt = [gt[0] / IMAGE_WIDTH, gt[1] / IMAGE_HEIGHT, (gt[0] + gt[2]) / IMAGE_WIDTH,
                            (gt[1] + gt[3]) / IMAGE_HEIGHT]  # 宽，高，宽，高  左上角与右下角
            normalize_gt_rect.append(normalize_gt)

        normalize_gt_rect = np.array(normalize_gt_rect)

        for fileName, det_result in self.det_results.items():
            frame_index = int(fileName[0:-4])
            groundtruth_boxes = np.array(normalize_gt_rect[frame_index - 1:frame_index])

            the_iou = 0
            the_score = 0
            for i in range(len(det_result["det_labels"])):
                if det_result['det_scores'][i] >= args.score:
                    if names.get(str(det_result["det_labels"][i].item())) == self.GROUND_TRUTH_CLASS:
                        det_boxes = det_result["det_boxes"][i:i+1]
                        det_boxes[0][0] /= IMAGE_WIDTH
                        det_boxes[0][1] /= IMAGE_HEIGHT
                        det_boxes[0][2] /= IMAGE_WIDTH
                        det_boxes[0][3] /= IMAGE_HEIGHT
                        iou = utils.iou(det_boxes, groundtruth_boxes)

                        if iou[0] > the_iou:
                            the_iou = iou[0]
                            the_score = det_result['det_scores'][i]
            iou_for_videos.append(the_iou)
            scores_for_videos.append(the_score)

        return iou_for_videos, scores_for_videos

    def main(self):
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the port
        # server_address = ('0.0.0.0', 10009)
        print(sys.stderr, 'starting up on %s port %s' % self.server_ip)
        sock.bind(self.server_ip)

        # Listen for incoming connections
        sock.listen(1)
        connect = True

        while True:
            # Wait for a connection
            byte_size = 0

            print(sys.stderr, 'waiting for a connection')
            connection, client_address = sock.accept()

            try:
                print(sys.stderr, 'connection from', client_address)
                fn = self.recv_edge_out_name
                fp = open('./' + fn, 'wb')

                # Receive the data in small chunks and retransmit it
                while True:
                    data = connection.recv(1024)
                    if data:
                        fp.write(data)
                    else:
                        break
                    byte_size = byte_size + len(data)
                fp.close()

            finally:
                # Clean up the connection
                connection.close()
                # TODO: Transform the data type
                edge_outs = []
                with h5py.File('./' + self.recv_edge_out_name, 'r') as f:
                    h5py_group_names = f['group_names']
                    for group_name in h5py_group_names:
                        edge_out = OrderedDict()
                        for k in f[group_name].keys():
                            edge_out[k] = torch.from_numpy(f[group_name.decode() + "/" + k][:])
                        edge_outs.append(edge_out)

                print('Size of data received: ', byte_size)
                self.edge_out_size = byte_size
                self.run_fasterrcnn_server(edge_outs)
                break


if __name__ == '__main__':
    fasterrcnn_server_instance = FasterrcnnServer()
    fasterrcnn_server_instance.main()

    for i in range(len(fasterrcnn_server_instance.server_time)):
        print("server-Time-Frame-" + str(i+1) + ": " + str(fasterrcnn_server_instance.server_time[i]))

    total_time = 0
    for i in range(len(fasterrcnn_server_instance.server_time)):
        total_time += fasterrcnn_server_instance.server_time[i]

    server_avg_time = total_time / len(fasterrcnn_server_instance.server_time)
    print("server_avg_time:" + str(server_avg_time))

    print("edge_out_size:" + str(fasterrcnn_server_instance.edge_out_size))
    edge_out_avg_size = fasterrcnn_server_instance.edge_out_size / len(fasterrcnn_server_instance.TEST_IMAGE_PATHS)
    print("edge_out_avg_size:" + str(edge_out_avg_size))

    iou_for_videos, scores_for_videos = fasterrcnn_server_instance.eval()

    x = np.arange(len(fasterrcnn_server_instance.TEST_IMAGE_PATHS))
    plt.plot(x, iou_for_videos, label='iou')
    plt.plot(x, scores_for_videos, label='scores')
    plt.title('fasterrcnn for BlurBody')
    plt.xlabel('Frame')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # plt.ylabel('y')
    plt.legend()
    plt.savefig(fasterrcnn_server_instance.OUTPUT_IMAGE_PATH + 'evaluation_fasterrcnn.jpg')

    print(iou_for_videos)
    print(scores_for_videos)
    print('Mean iou:' + str(np.array(iou_for_videos).mean()))
    print('Mean scores:' + str(np.array(scores_for_videos).mean()))
