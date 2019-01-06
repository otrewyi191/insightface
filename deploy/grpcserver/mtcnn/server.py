#! /usr/bin/env python
# -*- coding: utf-8 -*-
import grpc
import time
from concurrent import futures
import mtcnn_pb2
import mtcnn_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '127.0.0.1'
_PORT = '8080'


class FormatData(mtcnn_pb2_grpc.GetFaceServicer):
    def Get(self, request, context):
        # parse image to numpy array
        embedding = request.b64image
        print(embedding)

        # predict
        return mtcnn_pb2.FaceMessage()


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
