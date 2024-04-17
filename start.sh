#!/bin/bash
docker run --rm -it -p 5000:5000 -p 8890:8888 -v $PWD/models:/app/models -v $PWD/src:/app/src -v $PWD/datasets:/app/datasets garbage
#  -u $(id -u):$(id -g)
