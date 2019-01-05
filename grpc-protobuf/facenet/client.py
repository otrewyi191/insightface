# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import grpc
import facenet_pb2,facenet_pb2_grpc
import numpy as np


_HOST = '127.0.0.1'
_PORT = '8080'

def get_str():

    # Create a dummy matrix
    img = np.ones((50, 50, 3), dtype=np.uint8) * 255
    # Save the shape of original matrix.
    img_shape = img.shape

    message_image = img.tobytes()
    # message_image = np.ndarray.tostring(img)
    return message_image


def run():
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)
    client = facenet_pb2_grpc.GetEmbeddingStub(channel=conn)
    image_message = facenet_pb2.ImageMessage(image=get_str())
    response = client.Get(image_message)
    print("received: ")
    print(response)


if __name__ == '__main__':
    run()