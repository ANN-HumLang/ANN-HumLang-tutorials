# Interpretability with Soft Entropy

As part of the Using ANNs to Study Human Langauge Learning and Processing Workshop at the University of Amsterdam.

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
