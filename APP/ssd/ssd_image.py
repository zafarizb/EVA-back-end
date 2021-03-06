# coding:utf-8

import os
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


class SsdImage(object):
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

    def run_ssd_image(self):
        print('Running: ', self.TEST_IMAGE_PATHS)
        original_image = Image.open(self.TEST_IMAGE_PATHS, mode='r')
        original_image = original_image.convert('RGB')

        # Transform
        image = normalize(to_tensor(resize(original_image)))

        # Move to default device
        image = image.to(device)

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
            original_image.save(os.path.join(self.OUTPUT_IMAGE_PATH, self.TEST_IMAGE_NAME))
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

            annotated_image.save(os.path.join(self.OUTPUT_IMAGE_PATH, self.TEST_IMAGE_NAME))

        print('Finished: ', self.TEST_IMAGE_PATHS)

        torch.cuda.empty_cache()
