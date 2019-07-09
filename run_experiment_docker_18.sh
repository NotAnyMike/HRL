#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line


#sudo docker run -it --runtime=nvidia --rm --network host --ipc=host \
docker run --runtime=nvidia --rm -ti \
  --mount src=$(pwd)/experiments,target=/HRL/outside_experiments,type=bind \
  --mount src=$(pwd)/tracks,target=/HRL/tracks,type=bind \
  notanymike/hrl_entry18_notracks_v2 \
  bash -c "cd /stable-baselines && git reset --hard && git pull && 
	cd /gym && git reset --hard && git pull && 
	cd /HRL && git reset --hard && git pull &&
	$cmd_line"
