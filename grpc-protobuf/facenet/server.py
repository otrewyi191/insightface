#! /usr/bin/env python
# -*- coding: utf-8 -*-
import grpc
import time
from concurrent import futures
import facenet_pb2,facenet_pb2_grpc
import numpy as np

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '127.0.0.1'
_PORT = '8080'


class FormatData(facenet_pb2_grpc.GetEmbeddingServicer):
    def Get(self, request, context):
        embedding = request.image
        nparr = np.frombuffer(embedding, dtype=np.uint8)
        shape = list(request.dim)

        reshape = nparr.reshape(shape)
        print(reshape)
        print(reshape.shape)
        return facenet_pb2.EmbeddingMessage()


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
