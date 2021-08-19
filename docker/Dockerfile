FROM python:3.8
RUN set -ex; \
    apt-get update; \
    apt-get install -y less vim-nox locate screen rsync libglib2.0-0; \
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -; \
    apt-get install -y nodejs; \
    npm install -g @bazel/bazelisk; \
    pip install --upgrade pip; \
    pip install poetry
WORKDIR /root/openai_attack_simulation
COPY . .
RUN poetry update; \
    poetry install
ENV SHELL=bash
ENTRYPOINT ["poetry", "shell"]
