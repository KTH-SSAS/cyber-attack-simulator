FROM rayproject/ray:latest-py38
USER root
RUN set -ex; \
    apt-get update; \
    apt-get install -y less vim-nox locate screen \
        libglib2.0-0  # libsm6 libxrender1 libxext6 ## OpenCV
USER ray
WORKDIR /home/ray/openai_attack_simulation
COPY --chown=ray:ray . .
RUN pip install --upgrade pip poetry; \
    poetry update; \
    poetry install
ENV SHELL=bash
EXPOSE 8265
ENTRYPOINT ["poetry", "shell"]
