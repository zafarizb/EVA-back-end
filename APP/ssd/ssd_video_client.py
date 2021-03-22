# coding:utf-8

import os
import sys
import pathlib
import socket
import h5py
import numpy as np
import torch
from torchvision import transforms
from APP.ssd import ssd_util

from matplotlib import font_manager as fm
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Load model checkpoint
checkpoint = 'APP/ssd/checkpoint/checkpoint_ssd300.pth.tar'
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


class SsdVideoClient(object):
    def __init__(self, input_images_path=None, output_images_path=None,
                 partition=None, client_ip=None, server_ip=None):
        if (input_images_path is not None) and (output_images_path is not None) \
                and (partition is not None) and (client_ip is not None) and (server_ip is not None):
            self.PATH_TO_TEST_IMAGES_DIR = input_images_path
            self.TEST_IMAGE_PATHS = sorted(list(self.PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
            self.OUTPUT_IMAGE_PATH = output_images_path
            self.partition = partition
            self.client_ip = client_ip
            self.server_ip = server_ip
        else:
            self.PATH_TO_TEST_IMAGES_DIR = pathlib.Path('')
            self.TEST_IMAGE_PATHS = sorted(list(self.PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
            self.OUTPUT_IMAGE_PATH = './results/BlurBody/'
            self.partition = 1
            self.client_ip = 'localhost'
            self.server_ip = 'localhost'

        self.send_edge_out_name = "send_edge_out"

        if not os.path.exists(self.OUTPUT_IMAGE_PATH):
            os.mkdir(self.OUTPUT_IMAGE_PATH)

    def run_ssd_video_client(self):
        h5py_group_names = []
        for j in range(len(self.TEST_IMAGE_PATHS)):
            print('Running: ', self.TEST_IMAGE_PATHS[j])
            h5py_group_names.append(self.TEST_IMAGE_PATHS[j].name[0:-4].encode())
            original_image = Image.open(self.TEST_IMAGE_PATHS[j], mode='r')
            original_image = original_image.convert('RGB')

            # Transform
            image = normalize(to_tensor(resize(original_image)))

            # Move to default device
            image = image.to(device)

            if self.partition == 0:  # no partition:
                # Forward prop.
                predicted_locs, predicted_scores = model(image.unsqueeze(0))

                # Detect objects in SSD output
                det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores,
                                                                         min_score=min_score,
                                                                         max_overlap=max_overlap, top_k=top_k)
                # Move detections to the CPU
                det_boxes = det_boxes[0].to('cpu')

                # Transform to original image dimensions
                original_dims = torch.FloatTensor(
                    [original_image.width, original_image.height, original_image.width,
                     original_image.height]).unsqueeze(0)
                det_boxes = det_boxes * original_dims

                # Decode class integer labels
                det_labels = [ssd_util.rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

                det_scores = [items.detach().cpu().numpy() for items in det_scores]

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
                        draw.rectangle(xy=box_location, outline=ssd_util.label_color_map[det_labels[i]])
                        draw.rectangle(xy=[l + 1. for l in box_location], outline=ssd_util.label_color_map[
                            det_labels[i]])
                        # Text
                        text_size = font.getsize(det_labels[i].upper())
                        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
                        textbox_location = [box_location[0], box_location[1] - text_size[1],
                                            box_location[0] + text_size[0] + 4.,
                                            box_location[1]]
                        draw.rectangle(xy=textbox_location, fill=ssd_util.label_color_map[det_labels[i]])
                        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                                  font=font)
                    del draw

                    annotated_image.save(os.path.join(self.OUTPUT_IMAGE_PATH, self.TEST_IMAGE_PATHS[j].name))

                print('Finished: ', self.TEST_IMAGE_PATHS)

            elif self.partition in [1, 2, 3]:    # partition at vgg layer1, 2, 3
                edge_out = model(image.unsqueeze(0), self.partition)

                # edge_out_np = edge_out.detach().numpy()  # torch.tensor -> numpy array
                edge_out_np = edge_out.detach().cpu().numpy()  # torch.tensor -> numpy array
                with h5py.File('./' + self.send_edge_out_name, 'a') as f:
                    f.create_dataset(self.TEST_IMAGE_PATHS[j].name[0:-4] + "/edge_out", data=edge_out_np)
                    f.create_dataset(self.TEST_IMAGE_PATHS[j].name[0:-4] + "/edge_conv4_3_feats", data=np.array([]))

            else:
                edge_out, edge_conv4_3_feats = model(image.unsqueeze(0), self.partition)

                # edge_out_np = edge_out.detach().numpy()  # torch.tensor -> numpy array
                # edge_conv4_3_feats_np = edge_conv4_3_feats.detach().numpy()
                edge_out_np = edge_out.detach().cpu().numpy()  # torch.tensor -> numpy array
                edge_conv4_3_feats_np = edge_conv4_3_feats.detach().cpu().numpy()
                with h5py.File('./' + self.send_edge_out_name, 'a') as f:
                    f.create_dataset(self.TEST_IMAGE_PATHS[j].name[0:-4] + "/edge_out", data=edge_out_np)
                    f.create_dataset(self.TEST_IMAGE_PATHS[j].name[0:-4] + "/edge_conv4_3_feats",
                                     data=edge_conv4_3_feats_np)

        torch.cuda.empty_cache()

        if self.partition != 0:
            with h5py.File('./' + self.send_edge_out_name, 'a') as f:
                f.create_dataset("group_names", data=np.array(h5py_group_names))

            # 将模型划分的中间结果传送给云服务器
            self.send()
            torch.cuda.empty_cache()

    def send(self):
        print('Data being sent to server')
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
