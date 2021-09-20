FROM python:3.8
RUN set -ex; \
    apt-get update; \
    apt-get install -y less vim-nox locate screen rsync jq colordiff libglib2.0-0; \
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -; \
    apt-get install -y nodejs; \
    npm install -g @bazel/bazelisk; \
    pip install --upgrade pip; \
    pip install poetry; \
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
    poetry install; \
    poetry run jupyter lab build; \
    ln -s /root/.cache/pypoetry/virtualenvs/* /root/venv; \
    echo 'PATH=/root/venv/bin:$PATH' > /root/.bashrc
ENV SHELL=bash
ENV BASH_ENV=/root/.bashrc
ENV PATH=/root/venv/bin:$PATH
CMD ["bash"]
