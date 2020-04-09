# Gimme Signals

This repository contains the action recognition approach as presented in the Gimme Signals paper.

## Prequisites

* Pytorch Lightning
* Pytorch
* Hydra

## Generate Representation


### Precalculated representations

We provide precalculated representations for intermediate result reproduction:

* [Simitate (MoCap)](https://agas.uni-koblenz.de/gimme_signals/precal_representation_simitate.zip)
* [UTD-MHAD (Inertial and Skeleton)](https://agas.uni-koblenz.de/gimme_signals/precal_representation_utdmhad.zip)
* [ARIL (Wi-Fi)](https://agas.uni-koblenz.de/gimme_signals/precal_representation_aril.zip)

## Train

Example:

`train.py dataset=simitate model_name=efficientnet learning_rate=0.1 net="efficientnet"`

Exemplary, this command trains using the simitate dataset.

