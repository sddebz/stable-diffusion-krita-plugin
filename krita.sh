#!/usr/bin/env bash

PYTHON=python
GIT=git
COMMANDLINE_ARGS=
VENV_DIR=venv

mkdir tmp

TORCH_COMMAND=pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
REQS_FILE=requirements_versions.txt

source activate stable-diffusion-krita-plugin
python -m $TORCH_COMMAND
python -c "import omegaconf"
python -m pip install -r $REQS_FILE
mkdir repositories
git clone https://github.com/CompVis/stable-diffusion.git repositories\stable-diffusion
git clone https://github.com/CompVis/taming-transformers.git repositories\taming-transformers