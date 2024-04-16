FROM python:3.11-bookworm

ENV ROOT_DIR=/opt/hd_wsi


RUN pip install poetry # curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local/bin python3 -

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN printf "deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm main contrib non-free non-free-firmware\n\
        deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-updates main contrib non-free non-free-firmware\n\
        deb https://mirrors.tuna.tsinghua.edu.cn/debian/ bookworm-backports main contrib non-free non-free-firmware\n\
        deb https://security.debian.org/debian-security bookworm-security main contrib non-free non-free-firmware" > /etc/apt/sources.list


RUN apt-get update && \
    apt-get install -y build-essential libgdal-dev ffmpeg libsm6 libxext6 git && \
    apt-get install -y openslide-tools libgl1-mesa-dev libglib2.0-0 gobject-introspection && \
    apt-get clean

RUN mkdir $ROOT_DIR
COPY ./ $ROOT_DIR
WORKDIR $ROOT_DIR

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install -U pip && \
    poetry export -f requirements.txt -o requirements.txt --without-hashes --without-urls && \
    pip install -r requirements.txt && \
    rm -rf $POETRY_CACHE_DIR && rm -r $(pip cache dir)

ENTRYPOINT ["python", "/opt/hd_wsi/main.py"]

