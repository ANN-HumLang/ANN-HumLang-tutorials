#!/bin/sh

set -e

conda create -n h_interpret python=3.9


source activate h_interpret

cd codebase

pip install -e .

cd ..

jupyter lab