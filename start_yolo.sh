#!/bin/bash

# Set display permissions
xhost +local:docker

# Wait for X server to be ready
while [ ! -e /tmp/.X11-unix/X0 ]; do
    sleep 1
done

# Wait for Docker to be ready
while ! docker info >/dev/null 2>&1; do
    sleep 1
done

# Run Docker with proper X11 forwarding
docker run --gpus all \
    --rm \
    -v /dev:/dev \
    --device /dev/video0 \
    --device /dev/ttyACM0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $XAUTHORITY:/root/.Xauthority \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=/root/.Xauthority \
    -e QT_X11_NO_MITSHM=1 \
    --ipc=host \
    --runtime=nvidia \
    -v /home/geopavenet:/home/geopavenet \
    -v /media/geopavenet:/media/geopavenet \
    custom-ultralytics:latest \
    python3 /home/geopavenet/yolo/yolodetect12.py
