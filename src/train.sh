#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/home/zzx/github.com/otrewyi191/insightface/datasets

NETWORK=r100
JOB=softmax1e3
MODELDIR="../model-$NETWORK-$JOB"
mkdir -p "$MODELDIR"
PREFIX="$MODELDIR/model"
LOGFILE="$MODELDIR/log"

PRETRAIN_MODEL="/home/zzx/github.com/otrewyi191/insightface/models/model-r100-ii"

python -u train_softmax.py --pretrained $PRETRAIN_MODEL --data-dir $DATA_DIR --network "$NETWORK" --loss-type 0 --prefix "$PREFIX" --per-batch-size 128 > "$LOGFILE" 2>&1 &

