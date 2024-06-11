# Interpretability with Soft Entropy [![CC BY 4.0][cc-by-shield]][cc-by]



As part of the Using ANNs to Study Human Langauge Learning and Processing Workshop at the University of Amsterdam.

The analysis used here is described in greater detail in [this paper](https://arxiv.org/pdf/2406.02449)

## Setup

There are three steps to the setup process.

#### 1. Copy this repo to your machine

You can either copy at the command line with SSH by running ```git clone git``` or by downloading it from the repo page.

#### 2. Install Miniconda

if you already have a Conda version (or other venv) installed you can skip ahead to the next step. We've provided scripts to do this in the ```conda_install``` directory. This should work for MacOS and Linux based machines, by navigating to that directory and running ```bash name_of_scripy.sh``` installing on windows may prove

#### 3. Install Dependencies

In the terminal navigate to the directory where you downloaded the repo from github. It contains a file called setup.sh - this file will handle the rest of the setup. You can run ```bash setup.sh``` and the script will:

- create a new conda environment called ```h_interpret``` with python version 3.9, and activate that envionment
- install the package for performing entropy analyses of bert models, and all related packages we'll need for today
- launch an instance of jupyterlab, an integrated editor for jupyter notebooks.

You should now be all set!


## Get Started
You're all set! Now look in the notebooks ```0_analysis.ipynb``` for starter code to run analyses. ```1_resources.ipynb``` contains a listing of supported models and datasets. ```2_entropy_explanation.ipynb``` includes a primer on entropy and the measures used here

## Additional Models and Datasets
Take a look on hugging face for other models, and datasets. The analysis code works with any BERT model so feel free to try some other, than the ones listed here (the code will throw an error if it can't work the model you load).

For additional datasets, you need to define a ```get_example``` method that formats batches of examples to be fed into the model (ask Henry for tips!).

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg