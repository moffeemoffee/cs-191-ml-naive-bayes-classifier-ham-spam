# Setup

## Folder structure

```
├── dataset
|  └── trec07p
|     ├── data
|     ├── delay
|     ├── full
|     ├── partial
|     └── README.txt
├── log.txt
├── naivebayes.py
├── preprocess.py
├── processed.csv
├── README.md
├── requirements.txt
└── train.py
```

Get the data set from https://plg.uwaterloo.ca/~gvcormac/treccorpus07/ and extract it into the dataset folder, following the provided structure.

## Pre-requisites

Python 3.7.2 (un-tested on other versions)

## Installation

```
pip install -r requirements.txt
```

## Usage

Run `train.py`, it uses `processed.csv`:
```
python train.py
```

Example usage of `NaiveBayes` from `naivebayes.py` can be seen in `train.py` as well.

Also, you can generate a new `processed.csv` if you want with `preprocess.py`:
```
python preprocess.py
```