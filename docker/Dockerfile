FROM python:3.8
RUN set -ex; \
    apt-get update; \
    apt-get install -y less vim-nox locate screen; \
    pip install poetry
WORKDIR /root/openai_attack_simulation
COPY . .
RUN poetry install
ENV SHELL=bash
ENTRYPOINT ["poetry", "shell"]
