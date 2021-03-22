# coding:utf-8

import os
import sys
import datetime
import socket
import pathlib
import h5py
import numpy as np
from torchvision import transforms
from ssd_util import *
import utils

from matplotlib import pyplot as plt, font_manager as fm
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Load model checkpoint
checkpoint = 'ssd/checkpoint/checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

min_score = 0.5
max_overlap = 0.5
top_k = 200
suppress = None


class SsdClient(object):
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
            self.partition = 1
            self.server_ip = 'localhost'
            self.is_trainer = True

        if self.is_trainer:
            self.send_edge_out_name = "send_edge_out_trainer"
        else:
            self.send_edge_out_name = "send_edge_out_tester"

        if not os.path.exists(self.OUTPUT_IMAGE_PATH):
            os.mkdir(self.OUTPUT_IMAGE_PATH)

        self.image_width = 0
        self.image_height = 0

        self.client_time = []
        self.gpu_time = []
        self.transfer_time = 0

        self.local_client_time = []
        self.local_gpu_time = []

        self.det_results = {}

    def run_ssd_client(self):
        h5py_group_names = []
        if os.path.exists('./' + self.send_edge_out_name):
            os.remove('./' + self.send_edge_out_name)
        test_image = Image.open(self.TEST_IMAGE_PATHS[0], mode='r')
        self.image_width = max(test_image.size)
        self.image_height = min(test_image.size)
        for j in range(len(self.TEST_IMAGE_PATHS)):
            start = datetime.datetime.now()
            print('Running: ', self.TEST_IMAGE_PATHS[j])
            h5py_group_names.append(self.TEST_IMAGE_PATHS[j].name[0:-4].encode())
            original_image = Image.open(self.TEST_IMAGE_PATHS[j], mode='r')
            original_image = original_image.convert('RGB')

            # Transform
            image = normalize(to_tensor(resize(original_image)))

            # Move to default device
            image = image.to(device)

            if self.partition == 0:  # no partition:
                gpu_start = datetime.datetime.now()
                # Forward prop.
                predicted_locs, predicted_scores = model(image.unsqueeze(0))

                # Detect objects in SSD output
                det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores,
                                                                         min_score=min_score,
                                                                         max_overlap=max_overlap, top_k=top_k)
                gpu_end = datetime.datetime.now()
                timestr = str(gpu_end - gpu_start)[5:]
                self.gpu_time.append(float(timestr))


                # Move detections to the CPU
                det_boxes = det_boxes[0].to('cpu')

                # Transform to original image dimensions
                original_dims = torch.FloatTensor(
                    [original_image.width, original_image.height, original_image.width,
                     original_image.height]).unsqueeze(0)
                det_boxes = det_boxes * original_dims

                # Decode class integer labels
                det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

                det_scores = [items.detach().cpu().numpy() for items in det_scores]

                self.det_results[self.TEST_IMAGE_PATHS[j].name] = {"det_boxes": det_boxes.detach().numpy(),
                                                                   "det_labels": np.array(det_labels),
                                                                   "det_scores": np.array(det_scores)}

                # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
                if det_labels == ['background']:
                    # Just return original image
                    original_image.save(os.path.join(self.OUTPUT_IMAGE_PATH, self.TEST_IMAGE_PATHS[j].name))
                else:
                    # Annotate
                    annotated_image = original_image
                    draw = ImageDraw.Draw(annotated_image)
                    # font = ImageFont.truetype("./calibril.ttf", 15)
                    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='calibril')), 15)

                    # Suppress specific classes, if needed
                    for i in range(det_boxes.size(0)):
                        if suppress is not None:
                            if det_labels[i] in suppress:
                                continue

                        # Boxes
                        box_location = det_boxes[i].tolist()
                        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
                        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
                            det_labels[i]])
                        # Text
                        text_size = font.getsize(det_labels[i].upper())
                        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
                        textbox_location = [box_location[0], box_location[1] - text_size[1],
                                            box_location[0] + text_size[0] + 4.,
                                            box_location[1]]
                        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
                        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                                  font=font)
                    del draw

                    annotated_image.save(os.path.join(self.OUTPUT_IMAGE_PATH, self.TEST_IMAGE_PATHS[j].name))

                print('Finished: ', self.TEST_IMAGE_PATHS[j])
                end = datetime.datetime.now()

                timestr = str(end - start)[5:]
                self.client_time.append(float(timestr))

            elif self.partition in [1, 2, 3]:    # partition at vgg layer1, 2, 3
                gpu_start = datetime.datetime.now()

                edge_out = model(image.unsqueeze(0), self.partition)

                gpu_end = datetime.datetime.now()
                timestr = str(gpu_end - gpu_start)[5:]
                self.gpu_time.append(float(timestr))

                # edge_out_np = edge_out.detach().numpy()  # torch.tensor -> numpy array
                edge_out_np = edge_out.detach().cpu().numpy()  # torch.tensor -> numpy array
                with h5py.File('./' + self.send_edge_out_name, 'a') as f:
                    f.create_dataset(self.TEST_IMAGE_PATHS[j].name[0:-4] + "/edge_out", data=edge_out_np)
                    f.create_dataset(self.TEST_IMAGE_PATHS[j].name[0:-4] + "/edge_conv4_3_feats", data=np.array([]))

                end = datetime.datetime.now()
                timestr = str(end - start)[5:]
                self.client_time.append(float(timestr))

            else:
                gpu_start = datetime.datetime.now()

                edge_out, edge_conv4_3_feats = model(image.unsqueeze(0), self.partition)

                gpu_end = datetime.datetime.now()
                timestr = str(gpu_end - gpu_start)[5:]
                self.gpu_time.append(float(timestr))

                # edge_out_np = edge_out.detach().numpy()  # torch.tensor -> numpy array
                # edge_conv4_3_feats_np = edge_conv4_3_feats.detach().numpy()
                edge_out_np = edge_out.detach().cpu().numpy()  # torch.tensor -> numpy array
                edge_conv4_3_feats_np = edge_conv4_3_feats.detach().cpu().numpy()
                with h5py.File('./' + self.send_edge_out_name, 'a') as f:
                    f.create_dataset(self.TEST_IMAGE_PATHS[j].name[0:-4] + "/edge_out", data=edge_out_np)
                    f.create_dataset(self.TEST_IMAGE_PATHS[j].name[0:-4] + "/edge_conv4_3_feats", data=edge_conv4_3_feats_np)

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
                print('Local Running: ', self.TEST_IMAGE_PATHS[j])
                original_image = Image.open(self.TEST_IMAGE_PATHS[j], mode='r')
                original_image = original_image.convert('RGB')

                # Transform
                image = normalize(to_tensor(resize(original_image)))
                # Move to default device
                image = image.to(device)
                gpu_start = datetime.datetime.now()
                # Forward prop.
                predicted_locs, predicted_scores = model(image.unsqueeze(0))

                # Detect objects in SSD output
                det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores,
                                                                         min_score=min_score,
                                                                         max_overlap=max_overlap, top_k=top_k)
                gpu_end = datetime.datetime.now()
                timestr = str(gpu_end - gpu_start)[5:]
                self.local_gpu_time.append(float(timestr))

                # Move detections to the CPU
                det_boxes = det_boxes[0].to('cpu')

                # Transform to original image dimensions
                original_dims = torch.FloatTensor(
                    [original_image.width, original_image.height, original_image.width,
                     original_image.height]).unsqueeze(0)
                det_boxes = det_boxes * original_dims

                # Decode class integer labels
                det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

                # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
                if det_labels == ['background']:
                    # Just return original image
                    original_image.save(os.path.join(self.OUTPUT_IMAGE_PATH, self.TEST_IMAGE_PATHS[j].name))
                else:
                    # Annotate
                    annotated_image = original_image
                    draw = ImageDraw.Draw(annotated_image)
                    # font = ImageFont.truetype("./calibril.ttf", 15)
                    font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='calibril')), 15)

                    # Suppress specific classes, if needed
                    for i in range(det_boxes.size(0)):
                        if suppress is not None:
                            if det_labels[i] in suppress:
                                continue

                        # Boxes
                        box_location = det_boxes[i].tolist()
                        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
                        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
                            det_labels[i]])
                        # Text
                        text_size = font.getsize(det_labels[i].upper())
                        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
                        textbox_location = [box_location[0], box_location[1] - text_size[1],
                                            box_location[0] + text_size[0] + 4.,
                                            box_location[1]]
                        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
                        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                                  font=font)
                    del draw

                    annotated_image.save(os.path.join(self.OUTPUT_IMAGE_PATH, self.TEST_IMAGE_PATHS[j].name))

                print('Local Finished: ', self.TEST_IMAGE_PATHS[j])
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
                if det_result["det_labels"][i] == self.GROUND_TRUTH_CLASS:
                    det_boxes = det_result["det_boxes"][i:i+1]
                    det_boxes[0][0] /= IMAGE_WIDTH
                    det_boxes[0][1] /= IMAGE_HEIGHT
                    det_boxes[0][2] /= IMAGE_WIDTH
                    det_boxes[0][3] /= IMAGE_HEIGHT
                    iou = utils.iou(det_boxes, groundtruth_boxes)

                    if iou[0] > the_iou:
                        the_iou = iou[0]
                        the_score = det_result['det_scores'][0][i]
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
    ssd_client_instance = SsdClient()
    ssd_client_instance.run_ssd_client()

    for i in range(len(ssd_client_instance.client_time)):
        print("client-Time-Frame-" + str(i+1) + ": " + str(ssd_client_instance.client_time[i]))

    total_time = 0
    for i in range(len(ssd_client_instance.client_time)):
        total_time += ssd_client_instance.client_time[i]

    client_avg_time = total_time / len(ssd_client_instance.client_time)
    print("client_avg_time:" + str(client_avg_time))

    total_time = 0
    for i in range(len(ssd_client_instance.gpu_time)):
        total_time += ssd_client_instance.gpu_time[i]

    client_gpu_avg_time = total_time / len(ssd_client_instance.gpu_time)
    print("client_gpu_avg_time:" + str(client_gpu_avg_time))

    print("transfer_time:" + str(ssd_client_instance.transfer_time))
    transfer_avg_time = ssd_client_instance.transfer_time / len(ssd_client_instance.TEST_IMAGE_PATHS)
    print("transfer_avg_time:" + str(transfer_avg_time))
