FROM taikiinoue45/mvtec:mvtec
ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
WORKDIR /root

COPY requirements_apt.txt requirements_apt.txt
COPY requirements_pip.txt requirements_pip.txt
ENV PYTHONPATH /app/RIAD:$PYTHONPATH
RUN set -xe \
        && apt update -y \
        && apt install -y --no-install-recommends $(cat requirements_apt.txt) \
        && pip install --no-cache-dir -r requirements_pip.txt \
        && rm -rf /var/lib/apt/lists/* \
        && rm requirements_apt.txt requirements_pip.txt
