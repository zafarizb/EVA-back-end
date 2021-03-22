# coding:utf-8

import os
import sys
import pathlib
import socket
import h5py
import numpy as np
import torch
from torchvision import transforms
import ssd_util

from matplotlib import font_manager as fm
from PIL import Image, ImageDraw, ImageFont

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


class SsdVideoServer(object):
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

        self.recv_edge_out_name = "recv_edge_out"

        if not os.path.exists(self.OUTPUT_IMAGE_PATH):
            os.mkdir(self.OUTPUT_IMAGE_PATH)

    def run_ssd_video_server(self, edge_out, edge_conv4_3_feats):
        self.TEST_IMAGE_PATHS = sorted(list(self.PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
        for j in range(len(self.TEST_IMAGE_PATHS)):
            original_image = Image.open(self.TEST_IMAGE_PATHS[j], mode='r')
            original_image = original_image.convert('RGB')

            # Transform
            image = normalize(to_tensor(resize(original_image)))

            # Move to default device
            image = image.to(device)

            if self.partition in [1, 2, 3]:  # partition at vgg layer1, 2, 3
                predicted_locs, predicted_scores = model(image.unsqueeze(0), self.partition, edge_out[j])

            else:
                predicted_locs, predicted_scores = model(image.unsqueeze(0), self.partition, edge_out[j], edge_conv4_3_feats[j])

            # Detect objects in SSD output
            det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
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
                    draw.rectangle(xy=box_location, outline=ssd_util.label_color_map[det_labels[i]])
                    draw.rectangle(xy=[l + 1. for l in box_location], outline=ssd_util.label_color_map[
                        det_labels[i]])
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

            print('Finished: ', self.TEST_IMAGE_PATHS[j])

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

                edge_out = []
                edge_conv4_3_feats = []
                with h5py.File('./' + self.recv_edge_out_name, 'r') as f:
                    h5py_group_names = f['group_names']
                    for group_name in h5py_group_names:
                        edge_out.append(torch.from_numpy(f[group_name.decode() + "/edge_out"][:]).to(device))
                        edge_conv4_3_feats.append(torch.from_numpy(f[group_name.decode() + "/edge_conv4_3_feats"][:]).to(device))

                self.run_ssd_video_server(edge_out, edge_conv4_3_feats)
                break


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    pl = sys.argv[3]
    client_ip = ('127.0.0.1', 10009)
    server_ip = ('127.0.0.1', 20009)
    print(input_path, output_path, pl, client_ip, server_ip)
    # ssd_server_instance = SsdVideoServer()
    # ssd_server_instance.main()

    # D:\Anaconda3\Anaconda3\envs\py367\python.exe
