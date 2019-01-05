import argparse
import base64
import os
import time

import cv2
import grpc
import numpy as np
from concurrent import futures

import face_model

from mtcnngrpcserver import mtcnn_pb2_grpc, mtcnn_pb2
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '0.0.0.0'
_PORT = '8081'

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


def show_numpy_img(npyimg):
    # img = np.transpose(npyimg, (1,2,0))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Color image', npyimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class FormatData(mtcnn_pb2_grpc.GetFaceServicer):
    def Get(self, request, context):

        # parse image to numpy array
        b64image = request.b64image
        decode = base64.b64decode(b64image)

        image = cv2.imdecode(np.fromstring(decode, dtype=np.uint8), -1)
        # show_numpy_img(image)
        faces = model.get_face(image)

        face_message = mtcnn_pb2.FaceMessage()
        for face in faces:
            face_proto = face_message.faces.add()
            bbox,image = face
            face_proto.bbox.extend(list(bbox))
            face_proto.aliged_image=image.tobytes()
            face_proto.image_dim.extend(list(image.shape))


        return face_message


def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    mtcnn_pb2_grpc.add_GetFaceServicer_to_server(FormatData(), grpcServer)
    grpcServer.add_insecure_port(_HOST + ':' + _PORT)
    grpcServer.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)


if __name__ == '__main__':
    serve()
