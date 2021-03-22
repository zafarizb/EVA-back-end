import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision
import cv2
from PIL import Image
import numpy as np
import json
import math
import pickle

from torch.autograd import Variable


def area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def intersection(boxes1, boxes2):
    [y_min1, x_min1, y_max1, x_max1] = np.split(boxes1, 4, axis=1)
    [y_min2, x_min2, y_max2, x_max2] = np.split(boxes2, 4, axis=1)

    all_pairs_min_ymax = np.minimum(y_max1, np.transpose(y_max2))
    all_pairs_max_ymin = np.maximum(y_min1, np.transpose(y_min2))
    intersect_heights = np.maximum(
        np.zeros(all_pairs_max_ymin.shape),
        all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(x_max1, np.transpose(x_max2))
    all_pairs_max_xmin = np.maximum(x_min1, np.transpose(x_min2))
    intersect_widths = np.maximum(
        np.zeros(all_pairs_max_xmin.shape),
        all_pairs_min_xmax - all_pairs_max_xmin)

    return intersect_heights * intersect_widths


def iou(boxes1, boxes2):
    intersect = intersection(boxes1, boxes2)
    area1 = area(boxes1)
    area2 = area(boxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(
        area2, axis=0) - intersect
    return intersect / union


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def get_fps_and_number_of_frames(path_to_video):
    print('finding Number of frames')
    cap = cv2.VideoCapture(path_to_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, num_frames


def read_in_frame_from_video(path_to_video, frameNumber, write=False):
    cap = cv2.VideoCapture(path_to_video)

    if not cap.isOpened():
        print("could not open :", path_to_video)
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frameNumber - 1)
    res, frame = cap.read()

    if write:
        cv2.imwrite('frames/' + "frame%d.jpg" % frameNumber, frame)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]  # Numbers as per example on Pytorch website
    )
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        normalize
    ])

    img = Image.fromarray(frame)
    img = preprocess(img)
    img = img.unsqueeze(0)

    cap.release()
    cv2.destroyAllWindows()
    return img


def read_first_frame(FLAGS):
    print('Reading Frame')
    vidcap = cv2.VideoCapture(FLAGS.video_file)
    success, image = vidcap.read()
    count = 0
    success = True
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    cv2.imwrite('frames/' + "frame%d.jpg" % count, image)  # save frame as JPEG file
    count += 1


def read_in_all_frames(fileName):
    print('Reading Frames from: ', fileName)
    vidcap = cv2.VideoCapture(fileName)
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: ', success, ' :', count)
        cv2.imwrite("frames/frame%d.jpg" % count, image)  # save frame as JPEG file
        count += 1
    return count


def read_in_frame_per_second(fileName):
    cap = cv2.VideoCapture(fileName)
    frameRate = cap.get(5)  # frame rate
    count = 0
    while (cap.isOpened()):
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            filename = ("frames/frame%d.jpg" % count)
            cv2.imwrite(filename, frame)
        count += 1
    cap.release()
    return count


def load_in_image(path_to_image):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        normalize
    ])
    img = Image.open(path_to_image)
    img = preprocess(img)
    img = img.unsqueeze(0)

    return img


def read_in_frame_number_from_file(frameNumber):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        normalize
    ])
    img = Image.open('frames/frame%d.jpg' % frameNumber)
    img = preprocess(img)
    img = img.unsqueeze(0)

    return img


def encode(array, num_bins, min_num=-8, max_num=8):
    print('Encoding..')
    arr = array
    arr = np.clip(arr, min_num, max_num)
    arr = (arr / (max_num - min_num))  # -0.5 -> 0.5
    arr = (arr * num_bins)
    arr = np.round(arr)

    return arr


def compute_delta(previous_array, current_array, delta_value):
    print('Computing deltas')
    delta_array = previous_array - current_array
    delta_array[abs(delta_array) < delta_value] = 0
    return delta_array


def decode_delta(previous_array, delta_array):
    print('decoding deltas')
    return previous_array - delta_array


def decode(array, num_bins, min_num=-8, max_num=8):
    print('Decoding')
    arr = (array / num_bins) * (max_num - min_num)
    arr = np.expand_dims(arr, axis=0)

    return arr


def get_max_and_min(array):
    sort = array.data.numpy().argsort().squeeze(0)
    flat_arr = array.data.numpy().flatten()
    maxes = array.data.numpy().argmax()
    mins = array.data.numpy().argmin()
    print('Maxes: ', maxes)
    print('Mins: ', mins)
    print('Max: ', flat_arr[maxes])
    print('Min: ', flat_arr[mins])


def load_huff_dictionary(path):
    hist = None
    print('loading dictionary from: ', path)
    try:
        with open(path + '.pickle', 'rb') as handle:
            hist = pickle.load(handle)
    except Exception as e:
        print(e)
        print("Problem loading huffman dictionary.\nEnsure path to pickle file is correct."
              "\nOr try running train_huff_tree.py to create file if does no exist")
    return hist


