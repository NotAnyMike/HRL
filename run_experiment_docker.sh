#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line


nvidia-docker run -it --rm \
  --mount src=$(pwd)/experiments,target=/HRL/outside_experiments,type=bind \
  --mount src=$(pwd)/tracks,target=/HRL/tracks,type=bind \
  hrl:entry-16.04 \
  bash -c "cd /stable-baselines && git reset --hard && git pull && 
	cd /gym && git reset --hard && git pull && 
	cd /HRL && git reset --hard && git pull &&
	$cmd_line"
