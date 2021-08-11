FROM python:3.8
RUN set -ex; \
    apt-get update; \
    apt-get install -y less vim-nox locate screen \
        npm libglib2.0-0; \
    npm install -g @bazel/bazelisk; \
    pip install --upgrade pip; \
    pip install poetry
WORKDIR /root/openai_attack_simulation
COPY . .
RUN poetry update; \
    poetry install
ENV SHELL=bash
ENTRYPOINT ["poetry", "shell"]
