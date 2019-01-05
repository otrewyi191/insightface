import argparse
import os
import time

import cv2
import grpc
import numpy as np
from concurrent import futures

import face_model

from grpcserver import facenet_pb2_grpc, facenet_pb2
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '0.0.0.0'
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

class FormatData(facenet_pb2_grpc.GetEmbeddingServicer):
    def Get(self, request, context):

        # parse image to numpy array
        embedding = request.image
        nparr = np.frombuffer(embedding, dtype=np.uint8)
        shape = list(request.dim)

        reshape = nparr.reshape(shape)
        print(reshape)
        print(reshape.shape)

        # predict
        feature = model.get_feature(reshape)
        print(feature)

        embedding_message = facenet_pb2.EmbeddingMessage(embedding=feature.tobytes(),dim=(feature.shape))
        return embedding_message


def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    facenet_pb2_grpc.add_GetEmbeddingServicer_to_server(FormatData(), grpcServer)
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)
    grpcServer.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)


if __name__ == '__main__':
    serve()
