FROM python:3.9.7

# where external files are mapped to
ARG APP_DIR="/app"

# build directory
WORKDIR /tmp

# for vs code, see https://aka.ms/vscode-docker-python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# use bash as a default shell
ENV SHELL=/bin/bash
ENV force_color_prompt=yes

# path to app modules
ENV PYTHONPATH=${APP_DIR}:${APP_DIR}/lib:${APP_DIR}/src

# apt setup
RUN apt-get update
ENV TZ=America/Los_Angles
ENV DEBIAN_FRONTEND=noninteractive

# node js (required by for jupyterlab_code_formatter and plotly)
RUN apt-get install -y nodejs npm
RUN npm install -g n && n lts

# pip
RUN pip install --upgrade pip

# jupyter lab
RUN pip install jupyterlab==v3.1.12

# jupyterlab_code_formatter extension
RUN jupyter labextension install @ryantam626/jupyterlab_code_formatter@v1.4.10 --no-build
RUN pip install jupyterlab_code_formatter==v1.4.10
RUN jupyter server extension enable --py jupyterlab_code_formatter
RUN pip install isort yapf black

# plotly jl extension
RUN pip install "ipywidgets>=7.5"
RUN jupyter labextension install jupyterlab-plotly --no-build
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget --no-build
RUN pip install plotly

# lsp jl extension
# https://github.com/krassowski/jupyterlab-lsp
RUN pip install jupyterlab-lsp
RUN pip install python-language-server[all]

# jl debugger extension
# RUN pip install xeus-python
# RUN jupyter labextension install @jupyterlab/debugger --no-build

# build jupyter lab assets
RUN jupyter lab build

# jupyter config saving
RUN echo "cp -r /root/.jupyter ${APP_DIR}/docker/res/root/ && \
    rm -rf ${APP_DIR}/docker/res/root/.jupyter/lab/workspaces && echo \"done\"" \
    >> /usr/bin/save-jupyter-config
RUN chmod +x /usr/bin/save-jupyter-config

# startup commands
ARG JUPYTER_TOKEN="mytoken"
RUN echo "jupyter lab --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token='${JUPYTER_TOKEN}' &" >> /cmd
RUN echo "sleep infinity" >> /cmd
CMD ["sh", "/cmd"]

# client sdk for TRITON
RUN apt install -y libb64-dev
RUN pip install tritonclient[all]

# python packages
RUN pip install numpy
RUN pip install Pillow
RUN pip install tqdm
RUN pip install click
RUN pip install matplotlib

# geo packages
RUN pip install rasterio
RUN pip install fiona
RUN pip install geopandas
RUN pip install geopy
RUN pip install sentinelhub --upgrade
RUN pip install scikit-image

# torch
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install git+https://github.com/qubvel/segmentation_models.pytorch.git

# onnx
RUN pip install onnx

# copy external resources
COPY docker/res /

# cleanup
# RUN rm -rf /var/lib/apt/lists/*
# RUN rm -rf /tmp/*
# RUN jupyter lab clean
# RUN jlpm cache clean
# RUN npm cache clean --force

WORKDIR ${APP_DIR}
