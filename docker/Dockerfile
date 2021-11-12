FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
RUN set -ex; \
    export DEBIAN_FRONTEND=noninteractive; \
    ln -fs /usr/share/zoneinfo/Europe/Stockholm /etc/localtime; \
    apt-get update; \
    apt-get install -y python3.9 python3.9-distutils python3.9-dev build-essential \
        curl less vim-nox locate screen rsync; \
    curl https://bootstrap.pypa.io/get-pip.py | python3.9; \
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -; \
    apt-get install -y nodejs; \
    npm install -g @bazel/bazelisk; \
    pip install poetry\>=1.2.\*; \
    : install kubectl; \
    g_key="/usr/share/keyrings/google.gpg"; \
    echo "deb [signed-by=$g_key] https://apt.kubernetes.io/ kubernetes-xenial main" \
      > /etc/apt/sources.list.d/kubernetes.list; \
    curl http://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring "$g_key" add -; \
    apt-get update; \
    apt-get install -y kubectl

WORKDIR /root/openai_attack_simulation
COPY . .
RUN set -ex; \
    if [ $(uname -m) = aarch64 ]; then \
        : use a custom wheel, since Ray does NOT provide official wheels for linux/arm64; \
        poetry source add ray-wheels https://mehes-kth.github.io/ray-wheels/simple; \
        : skip GPU support, since PyTorch does NOT provide GPU builds for ARM; \
        poetry source remove torch; \
        sed -i s/+cu113// pyproject.toml; \
        rm poetry.lock; \
    fi; \
    poetry install; \
    ln -s /root/.cache/pypoetry/virtualenvs/* /root/venv; \
    echo 'PATH=/root/venv/bin:$PATH' > /root/.bashrc
ENV SHELL=bash
ENV BASH_ENV=/root/.bashrc
ENV PATH=/root/venv/bin:$PATH
CMD ["bash"]
