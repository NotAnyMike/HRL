#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line


#sudo docker run -it --runtime=nvidia --rm --network host --ipc=host \
docker run --runtime=nvidia --rm -ti \
  --mount src=$(pwd)/experiments,target=/HRL/outside_experiments,type=bind notanymike/hrl_entry18_notracks \
  bash -c "cd /stable-baselines && git reset --hard && git pull && 
	cd /gym && git reset --hard && git pull && 
	cd /HRL && git reset --hard && git pull &&
	$cmd_line"
