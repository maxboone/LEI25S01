#!/bin/bash
docker run --name adm242504_jupyter \
	--rm \
	-p 8888:8888 \
	-v $(dirname "$0")/..:/home/jovyan/work \
	-d \
	quay.io/jupyter/datascience-notebook \
	jupyter lab --NotebookApp.token=''

echo "JupyterHub should be up at localhost:8888 soon, if not, check the logs:"
echo "    docker logs adm242504_jupyter -f"
