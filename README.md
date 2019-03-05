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

First, generate the `processed.csv` if needed:
```
python preprocess.py
```

Then, run `train.py` on the generated `processed.csv`:
```
python train.py
```