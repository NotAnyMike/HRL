#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line


sudo docker run -it --runtime=nvidia --rm --network host --ipc=host \
  --mount src=$(pwd)/outside_experiments,target=/outside_experiments,type=bind hrl_cudnn\
  bash -c "cd /HRL/ && $cmd_line"
