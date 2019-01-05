# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2

import grpc
import mtcnn_pb2,mtcnn_pb2_grpc
import numpy as np
import base64


_HOST = '127.0.0.1'
_PORT = '8080'

def get_b64image():

    imgpath='/home/zzx/facedata/photo_2019-01-05_13-03-07.jpg'
    with open(imgpath,'r') as f:
        data = f.read()
        encode = base64.b64encode(data)
        return encode


def run():
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)
    client = mtcnn_pb2_grpc.GetFaceStub(channel=conn)

    image = get_b64image()

    image_message = mtcnn_pb2.ImageMessage(b64image=image )


    response = client.Get(image_message)
    print("received: ")
    print(response)


if __name__ == '__main__':
    run()
