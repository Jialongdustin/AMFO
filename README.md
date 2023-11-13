# Few-Shot Class-Incremental Audio Classification with Adaptive Mitigation of Forgetting and Overfitting

This repository contains the introductions to the datasets and codes used in our paper, titled "Few-Shot Class-Incremental Audio Classification with Adaptive Mitigation of Forgetting and Overfitting
" (as shown in the section of Citation).

## Datasets

To study the Few-shot Class-incremental Audio Classification (FCAC) problem, three datasets of LS-100 dataset, NSynth-100 dataset and FSC-89 dataset are constructed by 
choosing samples from audio corpora of the [Librispeech](https://www.openslr.org/12/) dataset, the [NSynth](https://magenta.tensorflow.org/datasets/nsynth) dataset and the [FSD-MIX-CLIPS](https://zenodo.org/record/5574135#.YWyINEbMIWo) dataset respectively.

Wei Xie, one of our team members, constructed the NSynth-100 dataset and FSC-89 dataset. The detailed information of these two datasets is [here](https://github.com/chester-w-xie/FCAC_datasets).

The detailed information of the LS-100 dataset is given below.

### Statistics on the LS-100 dataset

|                                                                 | LS-100                                        |
|:---------------------------------------------------------------:|:---------------------------------------------:|
| Type of audio                                                   | Speech                                        |
| Num. of classes                                                 | 100 (60 of base classes, 40 of novel classes) |
| Num. of training / validation / testing samples per base class  | 500 / 150 / 100                               |
| Num. of training / validation / testing samples per novel class | 500 / 150 / 100                               |
| Duration of the sample                                          | All in 2 seconds                              |
| Sampling frequency                                              | All in 16K Hz                                 |

### a use example of the LS-100 dataset

LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned. We find that the subset ``train-clean-100`` of   Librispeech is enough for our study, so we constructed the LS-100 dataset using partial samples from the Librispeech as the source materials. To be specific, we first concatenate all the speakers' speech clips into a long speech, and then select the 100 speakers with the longest duration to cut their voices into two second speech. You can download the Librispeech from [here](https://www.openslr.org/12/).

1. you can download [dataset](https://www.openslr.org/resources/12/train-clean-100.tar.gz) and extract the files.

2. Also, after you download the dataset, you can run the `features.py` to extracted mel-features from audio sample and stored in h5py files in the directory--dataset.

- `train.h5` stores the base session training data

- `new_train.h5` stores the incremental session training data

- `test.h5` stores the test data of all sessions

- `val.h5` stores the validate data of all sessions
   


## Overview
- `train.py` is the code for base training session;
- `Inc_train.py` is the code for incremental training session;
- `exemplar.py` is the code for reconstrcut the embedding;
- `features.py` is the code for extracting mel-features;
- `models/` contains the implementation of the AMFO(`AMFO.py`) and the backbone network;
- `dataloader/` is the code to load the training and testing data;
- `data_list/` contains the list of data;
- `config/` contains the setting file;
- `utils/` contains the log and restore files;


## Training
```bash
bash run_librispeech.sh
```


## Contact

Yanxiong Li (eeyxli@scut.edu.cn) and Jialong Li (lijialongjy@163.com)

School of Electronic and Information Engineering, South China University of Technology, Guangzhou, China

## Citation

Please cite our paper if you find the codes and datasets are useful for your research.

Y. Li, J. Li, Y. Si, J. Tan and Q. He, " Few-Shot Class-Incremental Audio Classification with Adaptive Mitigation of Forgetting and Overfitting," IEEE-ACM TASLP, 2023. Submitted.


