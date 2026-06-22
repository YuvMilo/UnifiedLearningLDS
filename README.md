Code for reproducing the experiment in the paper "A Memory Efficient Unified Algorithm for Online Learning of Linear Dynamical Systems".

## Setup

```bash
conda create -n unified-lds python=3.10 -y
conda activate unified-lds
pip install -r requirements.txt
```

## Reproducing the Experiment

Reproduces the results in Table 1: trains four online predictors (FIR, AR, SF,
and the unified UF) with the same parameter budget and prints their normalized
mean-squared error.

```bash
python main.py
```


