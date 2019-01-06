import argparse
import os

import cv2
import grpc
import numpy as np

_HOST = '127.0.0.1'
_PORT = '8080'

import face_model

from grpcserver.arcface import facenet_pb2_grpc, facenet_pb2

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '127.0.0.1'
_PORT = '8080'

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


def get_numpy_from_img(img_path):
    img = cv2.imread(img_path)
    img = model.get_input(img)
    return img




def get_input_from_img(img_path):
    img = cv2.imread(img_path)
    img = model.get_input(img)

def run():
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)
    client = facenet_pb2_grpc.GetEmbeddingStub(channel=conn)
    image = get_numpy_from_img(os.getenv('IMGPATH', '/home/zzx/facedata/photo_2019-01-05_12-43-40.jpg'))

    shape = image.shape

    image_message = facenet_pb2.ImageMessage(image=image.tobytes(), dim=list(shape))


    response = client.Get(image_message)

    embedding = response.embedding
    embedding = np.frombuffer(embedding,dtype=np.float32)
    embedding = embedding.reshape(list(response.dim))


    print("embedding: ")
    print(embedding)
    print(embedding.shape)


if __name__ == '__main__':
    run()