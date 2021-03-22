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
sys.path.append('./')
import coco_names
import utils


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


class FasterrcnnClient(object):
    def __init__(self, input_images_path=None, path_groundtruth_rect=None, ground_truth_class=None,
                 output_images_path=None, partition=None, server_ip=None, is_trainer=None):

        if (input_images_path is not None) and (path_groundtruth_rect is not None) \
                and (ground_truth_class is not None) and (output_images_path is not None) \
                and (partition is not None) and (server_ip is not None) and (is_trainer is not None):
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
            self.send_edge_out_name = "send_edge_out_trainer"
        else:
            self.send_edge_out_name = "send_edge_out_tester"

        self.image_width = 0
        self.image_height = 0

        self.client_time = []
        self.gpu_time = []
        self.transfer_time = 0

        self.local_client_time = []
        self.local_gpu_time = []

        self.det_results = {}

    def run_fasterrcnn_client(self):
        h5py_group_names = []
        if os.path.exists('./' + self.send_edge_out_name):
            os.remove('./' + self.send_edge_out_name)

        test_image = Image.open(self.TEST_IMAGE_PATHS[0], mode='r')
        self.image_width = max(test_image.size)
        self.image_height = min(test_image.size)
        for j in range(len(self.TEST_IMAGE_PATHS)):
            start = datetime.datetime.now()
            input = []
            print('Running: ', self.TEST_IMAGE_PATHS[j])
            h5py_group_names.append(self.TEST_IMAGE_PATHS[j].name[0:-4].encode())

            src_img = cv2.imread(str(self.TEST_IMAGE_PATHS[j]))
            img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            # img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
            img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()

            input.append(img_tensor)

            if self.partition == 0:  # no partition
                gpu_start = datetime.datetime.now()
                out = model(input, partition=0)
                gpu_end = datetime.datetime.now()

                print('Finish: ', self.TEST_IMAGE_PATHS[j])

                timestr = str(gpu_end - gpu_start)[5:]
                self.gpu_time.append(float(timestr))

                boxes = out[0]['boxes']
                labels = out[0]['labels']
                scores = out[0]['scores']

                self.det_results[self.TEST_IMAGE_PATHS[j].name] = {"det_boxes": boxes.detach().cpu().numpy(),
                                                                   "det_labels": np.array(labels.cpu()),
                                                                   "det_scores": scores.detach().cpu().numpy()}

                for idx in range(boxes.shape[0]):
                    if scores[idx] >= args.score:
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

                end = datetime.datetime.now()
                timestr = str(end - start)[5:]
                self.client_time.append(float(timestr))

            elif self.partition == 1:  # partition at resnet Layer1
                gpu_start = datetime.datetime.now()
                edge_out = model(input, partition=1, node='client')
                gpu_end = datetime.datetime.now()

                timestr = str(gpu_end - gpu_start)[5:]
                self.gpu_time.append(float(timestr))

                with h5py.File('./' + self.send_edge_out_name, 'a') as f:
                    for index, tensor in edge_out.items():
                        # ts2np = tensor.detach().numpy()
                        ts2np = tensor.cpu().detach().numpy()
                        f.create_dataset(self.TEST_IMAGE_PATHS[j].name[0:-4] + "/" + str(index), data=ts2np)

                end = datetime.datetime.now()
                timestr = str(end - start)[5:]
                self.client_time.append(float(timestr))

            elif self.partition == 2:  # partition at resnet Layer2
                gpu_start = datetime.datetime.now()
                edge_out = model(input, partition=2, node='client')
                gpu_end = datetime.datetime.now()

                timestr = str(gpu_end - gpu_start)[5:]
                self.gpu_time.append(float(timestr))

                with h5py.File('./' + self.send_edge_out_name, 'a') as f:
                    for index, tensor in edge_out.items():
                        # ts2np = tensor.detach().numpy()
                        ts2np = tensor.cpu().detach().numpy()
                        f.create_dataset(self.TEST_IMAGE_PATHS[j].name[0:-4] + "/" + str(index), data=ts2np)

                end = datetime.datetime.now()
                timestr = str(end - start)[5:]
                self.client_time.append(float(timestr))

            elif self.partition == 3:  # partition at resnet Layer3
                gpu_start = datetime.datetime.now()
                edge_out = model(input, partition=3, node='client')
                gpu_end = datetime.datetime.now()

                timestr = str(gpu_end - gpu_start)[5:]
                self.gpu_time.append(float(timestr))

                with h5py.File('./' + self.send_edge_out_name, 'a') as f:
                    for index, tensor in edge_out.items():
                        # ts2np = tensor.detach().numpy()
                        ts2np = tensor.cpu().detach().numpy()
                        f.create_dataset(self.TEST_IMAGE_PATHS[j].name[0:-4] + "/" + str(index), data=ts2np)

                end = datetime.datetime.now()
                timestr = str(end - start)[5:]
                self.client_time.append(float(timestr))

            elif self.partition == 4:  # partition at resnet Layer4
                gpu_start = datetime.datetime.now()
                edge_out = model(input, partition=4, node='client')
                gpu_end = datetime.datetime.now()

                timestr = str(gpu_end - gpu_start)[5:]
                self.gpu_time.append(float(timestr))

                with h5py.File('./' + self.send_edge_out_name, 'a') as f:
                    for index, tensor in edge_out.items():
                        # ts2np = tensor.detach().numpy()
                        ts2np = tensor.cpu().detach().numpy()
                        f.create_dataset(self.TEST_IMAGE_PATHS[j].name[0:-4] + "/" + str(index), data=ts2np)

                end = datetime.datetime.now()
                timestr = str(end - start)[5:]
                self.client_time.append(float(timestr))

        torch.cuda.empty_cache()

        if self.partition != 0:
            with h5py.File('./' + self.send_edge_out_name, 'a') as f:
                f.create_dataset("group_names", data=np.array(h5py_group_names))

            # 边缘服务器本地运行全流程，记录时间与资源开销
            for j in range(len(self.TEST_IMAGE_PATHS)):
                start = datetime.datetime.now()
                input = []
                print('Local Running: ', self.TEST_IMAGE_PATHS[j])

                src_img = cv2.imread(str(self.TEST_IMAGE_PATHS[j]))
                img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
                # img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
                img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()

                input.append(img_tensor)
                gpu_start = datetime.datetime.now()
                out = model(input, partition=0)
                gpu_end = datetime.datetime.now()

                print('Local Finish: ', self.TEST_IMAGE_PATHS[j])

                timestr = str(gpu_end - gpu_start)[5:]
                self.local_gpu_time.append(float(timestr))

                boxes = out[0]['boxes']
                labels = out[0]['labels']
                scores = out[0]['scores']

                for idx in range(boxes.shape[0]):
                    if scores[idx] >= args.score:
                        x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
                        name = names.get(str(labels[idx].item()))
                        # cv2.rectangle(img,(x1,y1),(x2,y2),colors[labels[idx].item()],thickness=2)
                        cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
                        cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

                cv2.imwrite(os.path.join(self.OUTPUT_IMAGE_PATH, self.TEST_IMAGE_PATHS[j].name), src_img)
                print("Save result image-" + self.TEST_IMAGE_PATHS[j].name)

                end = datetime.datetime.now()
                timestr = str(end - start)[5:]
                self.local_client_time.append(float(timestr))
            # 将模型划分的中间结果传送给云服务器
            self.send()

            torch.cuda.empty_cache()

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

    def send(self):
        print('Data being sent to server')
        start = datetime.datetime.now()
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the port where the server is listening
        print(sys.stderr, 'connecting to %s port %s' % self.server_ip)
        sock.connect(self.server_ip)

        try:
            filepath = './' + self.send_edge_out_name
            fp = open(filepath, 'rb')
            data = fp.read()
            sock.sendall(data)
        finally:

            print(sys.stderr, 'closing socket')
            sock.close()
            end = datetime.datetime.now()
            timestr = str(end - start)[5:]
            self.transfer_time = float(timestr)


if __name__ == '__main__':
    fasterrcnn_client_instance = FasterrcnnClient()
    fasterrcnn_client_instance.run_fasterrcnn_client()

    for i in range(len(fasterrcnn_client_instance.client_time)):
        print("client-Time-Frame-" + str(i+1) + ": " + str(fasterrcnn_client_instance.client_time[i]))

    total_time = 0
    for i in range(len(fasterrcnn_client_instance.client_time)):
        total_time += fasterrcnn_client_instance.client_time[i]

    client_avg_time = total_time / len(fasterrcnn_client_instance.client_time)
    print("client_avg_time:" + str(client_avg_time))

    total_time = 0
    for i in range(len(fasterrcnn_client_instance.gpu_time)):
        total_time += fasterrcnn_client_instance.gpu_time[i]

    client_gpu_avg_time = total_time / len(fasterrcnn_client_instance.gpu_time)
    print("client_gpu_avg_time:" + str(client_gpu_avg_time))

    print("transfer_time:" + str(fasterrcnn_client_instance.transfer_time))
    transfer_avg_time = fasterrcnn_client_instance.transfer_time / len(fasterrcnn_client_instance.TEST_IMAGE_PATHS)
    print("transfer_avg_time:" + str(transfer_avg_time))
