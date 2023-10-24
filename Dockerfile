FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV ROOT_DIR=/opt/hd_wsi

RUN printf "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse\n\
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse\n\
    deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse\n\
    deb http://security.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse" > /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y build-essential libgdal-dev ffmpeg libsm6 libxext6 && \
    apt-get install -y openslide-tools libgl1-mesa-dev libglib2.0-0 gobject-introspection && \
    apt-get clean

RUN mkdir $ROOT_DIR
COPY ./ $ROOT_DIR

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install -U pip && \
    cd $ROOT_DIR && pip install -r requirements.txt && rm requirements.txt

ENTRYPOINT ["python", "/opt/hd_wsi/main.py"]

