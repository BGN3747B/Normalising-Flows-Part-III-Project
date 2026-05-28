# Normalising-Flows-Part-III-Project
This repository contains the codes and supplementary files for the Part III project in normalising flows.

This contains the model and analysis codes for the part III Project on Normalising Flows for Particle Detector Acceptance Modelling. There are also optuna run CSV outputs.

.py files in flowlib are the main scripts (e.g. metrics.py contains the 5D energy distance and Sliced Wasserstein Distance implementations)

.iypnb files are Notebooks used for analysis

initial_project: initial notebook used for building the flow on the original dataset, and includes some of the plots used in the December report

final: the final analysis notebook containing investigation of accuracy metrics (seed, sample, permutation, projection dependence etc.)

bdt: the bdt functions used, from the existing Legendre code, available at https://gitlab.cern.ch/LHCb-BnoC/detector-acceptances legendre/-/tree/MattsStudents/Main_Scripts?ref_type=heads

optuna_run: contains the codes used for the Optuna hyperparameter optimisations.

checkpoint and model files are too large to upload here.
