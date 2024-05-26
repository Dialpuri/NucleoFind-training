# NucleoFind-training

This repository contains a collection of scripts used to generate the machine-learning models for NucleoFind - A Deep-Learning Network for Interpreting Nucleic Acid Electron Density.
All scripts that are necessary to create the models are in this repository, although for the moment they are not runnable with one command and require multiple steps. If you come across this repository and 
believe that would be helpful to your work, do not hesitate to get in touch and that can be created for you. 

## Directory Structure

The repository is organised into the following directories:

- `data`: This directory contains all the data needed for training the machine learning models, after running the dataset_creation scripts.
- `dataset_creation`: This directory hosts scripts and notebooks that facilitate the creation of the dataset. 
- `scripts`: This directory contains the scripts used to generate the machine learning models. The folder, dataset, is not used but retained for reference.
- `unrefined_scripts`: This directory contains scripts to recalculate maps of protein-nucleic acids using only the protein atoms. This generates maps with additional phase errors which contribute to better training data.

## Contact
Any questions feel free to email jordan.dialpuri (at) york.ac.uk
