docker build -f Dockerfile -t tensorflow/tensorflow:2.1.0rc0-py3 .

docker run -u $(id -u):$(id -g) -v $(pwd):/regeage -v
/Volumes/ELEMENTS/:/regeage/data -it tensorflow/tensorflow:2.1.0rc0-py3 python /regeage/code/datapreprocess.py
docker run -u $(id -u):$(id -g) -v $(pwd):/regeage -v
/Volumes/ELEMENTS/:/regeage/data -it tensorflow/tensorflow:2.1.0rc0-py3 python /regeage/code/train.py

