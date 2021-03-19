# Gimme Signals

This repository contains the action recognition approach as presented in the Gimme Signals paper.

![Gimme Signals Overview](images/gimme_signals_overview.png)

* A preprint can be found on [arxiv](https://arxiv.org/abs/2003.06156).
* A video abstract is available on [youtube](https://youtu.be/oDAtim_nJEg).

<!--<section id="video" class="bg-light">-->
## Video
<video width=100% controls>
<source src="video/gimme_signals.mp4">
</video>

<!--[![Gimme Signals Video](images/gimme_signals_video_preview.png)](https://userpages.uni-koblenz.de/~raphael/videos/gimme_signals.mp4)-->
[![Gimme Signal Video](images/gimme_signals_video_preview.png)](https://youtu.be/oDAtim_nJEg)

In case the video does not play you can download it [here](https://userpages.uni-koblenz.de/~raphael/videos/gimme_signals.mp4)

## Citation


```
@inproceedings{Memmesheimer2020GSD, 
   author = {Memmesheimer, Raphael and Theisen, Nick and Paulus, Dietrich}, 
   title = {Gimme Signals: Discriminative signal encoding for multimodal activity recognition}, 
   year = {2020}, 
   booktitle = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
   address = {Las Vegas, NV, USA}, 
   publisher = {IEEE}, 
   doi = {10.1109/IROS45743.2020.9341699}, 
   isbn = {978-1-7281-6213-3}, 
 } 
```

## Requirements

* pytorch, torchvision, pytorch-lightning, hydra-core
* `pip intall -r requirements.txt`

## Generate Representation

Example code to generate representations for the NTU dataset:

```
python generate_representation_ntu.py <ntu_skeleton_dir> $DATASET_FOLDER <split>
```
where split is either "cross_subject", "cross_setup", "one_shot"

Representations must be placed inside a `$DATASET_FOLDER` that an environment variable points to.

### Precalculated representations

We provide precalculated representations for intermediate result reproduction:

* [NTU RGB+D 120 Cross Subject](https://agas.uni-koblenz.de/gimme_signals/ntu_120_cross_subject.tar.gz)
* [Simitate (MoCap)](https://agas.uni-koblenz.de/gimme_signals/precal_representation_simitate.zip)
* [UTD-MHAD (Inertial and Skeleton)](https://agas.uni-koblenz.de/gimme_signals/precal_representation_utdmhad.zip)
* [ARIL (Wi-Fi)](https://agas.uni-koblenz.de/gimme_signals/precal_representation_aril.zip)

## Train

Example:

### Simitate

`python train.py dataset=simitate model_name=efficientnet learning_rate=0.1 net="efficientnet"`

Exemplary, this command trains using the simitate dataset.

