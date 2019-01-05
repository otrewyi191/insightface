import argparse

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


img = cv2.imread('/home/zzx/facedata/photo_2019-01-05_12-43-40.jpg')
img = model.get_input(img)

# img = np.transpose(img,[1,2,0])
# cv2.imshow('Color image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

f1 = model.get_feature(img)
print(f1[0:10])
# gender, age = model.get_ga(img)
# print(gender)
# print(age)
# sys.exit(0)
# img = cv2.imread('/home/zzx/facedata/photo_2019-01-05_12-43-29.jpg')
# img = model.get_input(img)
# f2 = model.get_feature(img)
# dist = np.sum(np.square(f1-f2))

img = cv2.imread('/home/zzx/github.com/otrewyi191/MSCELEB1M-GenImage/low_images/m.04xzm/354_o0MpowuG-FaceId-0.jpg')
# img = cv2.imread('/home/zzx/facedata/photo_2019-01-05_13-48-45.jpg')
img = model.get_input(img)
f3 = model.get_feature(img)

# print(dist)
sim = np.dot(f3, f1.T)
print(sim)
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
