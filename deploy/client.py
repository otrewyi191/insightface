# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import grpc
import grpcserver.arcface.facenet_pb2
import grpcserver
import base64


MTCNN_HOST = '127.0.0.1:8081'
ARCFACE_HOST = '127.0.0.1:8080'


def get_b64image():
    imgpath = '/home/zzx/下载/20180515105213848.jpg'
    with open(imgpath, 'r') as f:
        data = f.read()
        encode = base64.b64encode(data)
        return encode


def run():
    conn = grpc.insecure_channel(MTCNN_HOST)
    from grpcserver.mtcnn import mtcnn_pb2_grpc
    client = mtcnn_pb2_grpc.GetFaceStub(channel=conn)

    image = get_b64image()

    from grpcserver.mtcnn import mtcnn_pb2
    image_message = mtcnn_pb2.ImageMessage(b64image=image)

    response = client.Get(image_message)

    for face in response.faces:
        bbox = list(face.bbox)
        dim = list(face.image_dim)
        aliged_image = face.aliged_image

        #
        conn = grpc.insecure_channel(ARCFACE_HOST)
        client = grpcserver.arcface.facenet_pb2_grpc.GetEmbeddingStub(channel=conn)

        from grpcserver.arcface import facenet_pb2
        image_message = facenet_pb2.ImageMessage(image=aliged_image, dim=dim)

        response = client.Get(image_message)


        print bbox, dim,response


if __name__ == '__main__':
    run()
