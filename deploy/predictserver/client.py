# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

import grpc
import mtcnngrpcserver.mtcnn_pb2, mtcnngrpcserver.mtcnn_pb2_grpc
import grpcserver.facenet_pb2, grpcserver.facenet_pb2_grpc
import numpy as np
import base64

MTCNN_HOST = '127.0.0.1:8080'


def get_b64image():
    imgpath = '/home/zzx/下载/20180515105213848.jpg'
    with open(imgpath, 'r') as f:
        data = f.read()
        encode = base64.b64encode(data)
        return encode


def run():
    conn = grpc.insecure_channel(MTCNN_HOST)
    client = mtcnngrpcserver.mtcnn_pb2_grpc.GetFaceStub(channel=conn)

    image = get_b64image()

    image_message = mtcnngrpcserver.mtcnn_pb2.ImageMessage(b64image=image)

    response = client.Get(image_message)

    for face in response.faces:
        bbox = list(face.bbox)
        dim = list(face.image_dim)
        aliged_image = face.aliged_image

        frombuffer = np.frombuffer(aliged_image, dtype=np.uint8)
        reshape = frombuffer.reshape(dim)
        print bbox, dim, reshape


if __name__ == '__main__':
    run()
