import argparse
import os

import cv2
import numpy as np

import face_model

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)


def get_feature_from_img(img_path):
    img = cv2.imread(img_path)
    img = model.get_input(img)
    f1 = model.get_feature(img)
    return f1


f1 = get_feature_from_img(os.getenv('IMGPATH','/home/zzx/facedata/photo_2019-01-05_12-43-40.jpg'))
print(f1)



# sim = np.dot(f3, f1.T)
# print(sim)