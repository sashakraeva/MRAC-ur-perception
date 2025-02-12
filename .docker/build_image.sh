#!/usr/bin/env bash

echo -e "Building iaac_ur_perception:lastest image"

DOCKER_BUILDKIT=1 \
docker build --pull --rm -f ./.docker/Dockerfile \
--build-arg BUILDKIT_INLINE_CACHE=1 \
--target bash \
--tag iaac_ur_perception:latest .