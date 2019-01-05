#!/usr/bin/env bash
docker run --rm -ti -v /home/zzx/github.com/otrewyi191/insightface/models/model-r100-ii/:/model \
	-v '/home/zzx/facedata/photo_2019-01-05_13-48-45-2.jpg':/img.jpg \
	arcface bash
